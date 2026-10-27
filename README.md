# glasspane

[![CI](https://github.com/oneilltomhq/glasspane/actions/workflows/ci.yml/badge.svg)](https://github.com/oneilltomhq/glasspane/actions/workflows/ci.yml)

**A tenant Wayland compositor as a library.** It speaks Wayland to real clients
— terminals, browsers — and its entire output is one `wgpu::Texture` per client
plus an input-injection API.

It was built as the compositor layer of
[Upfront](https://github.com/oneilltomhq/upfront), a Wayland shell, and
depends on nothing in it.

```
  socket ─ wl_compositor + xdg_shell + zwp_linux_dmabuf_v1 + wl_shm
         └─ commit ─ import-once ─ blit ─▶ Tenant::texture   (your wgpu::Device)
  tick() ───────────────────────────────▶ wl_callback.done ─ client paints again
  pointer_motion / keyboard_key ────────▶ wl_pointer / wl_keyboard
```

```rust
let mut pane = Glasspane::new(device, queue, Caps::new().with_adapter(&adapter))?;
let _child = pane.spawn(Command::new("foot"))?;

loop {
    pane.dispatch(Some(Duration::from_millis(4)))?;
    pane.tick(Instant::now());          // hand out frame callbacks: see below
    for event in pane.events() { /* Mapped, Committed, Resized, … */ }
    for tenant in pane.tenants() {
        if let Some(texture) = tenant.texture() { /* sample it in your scene */ }
    }
}
```

## What it is not

No placement policy: glasspane never decides where a tenant is, how large it is
drawn, or what is in front of what. No scene graph, no renderer, no window. No
scanout: nothing here talks to KMS. The shell owns the `wgpu::Device`, the
camera, the frame rate and the screen; glasspane borrows the first and is told
about the third.

There is one mode and it is headless — no display, no session, no seat of its
own. That is not a limitation to be lifted later; a caller that wants a window
makes one.

**Never links GL or EGL.** The only GPU API in the dependency tree is Vulkan,
through `wgpu` and [`wgpu-dmabuf`](https://github.com/oneilltomhq/wgpu-dmabuf).

## Requirements

* Linux. Wayland, DRM and dma-buf are Linux interfaces; nothing here builds
  for anything else.
* A Vulkan driver with the dma-buf extensions: `VK_KHR_external_memory_fd`,
  `VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier`, and
  `VK_KHR_external_semaphore_fd` for the acquire fence. Mesa's ANV and RADV
  have them.
* A DRM render node, `/dev/dri/renderD*`, that your user can open.
* To build: Rust 1.87 or later and libxkbcommon (`libxkbcommon-dev`,
  `libxkbcommon-devel`); the tests and examples also link libgbm
  (`libgbm-dev`, `mesa-libgbm-devel`). libwayland-client is loaded at run
  time rather than linked; the library itself uses it only for
  `Glasspane::bridge_clipboard`, the tests for their client.
* For the examples, `foot` and a Chromium (`chromium`, `chromium-browser` or
  `google-chrome`) on `PATH`.

No display server is needed: glasspane is the server.

## Running the examples

```
cargo run --example spawn_foot
cargo run --example chromium_tenant -- https://example.com
```

The first starts `foot` on a private socket, runs the frame clock for two
seconds, and writes what it drew to `foot.png`: a shell prompt, 800x480. The
second does the same with Chromium and writes the page to `chromium.png`.
Both print what the tenant did on the way: mapped, its title, its size, how
many commits and how many dma-buf imports.

## Three rules that are load-bearing

### The frame clock is yours to drive

A Wayland client paints **only** when it has been handed a frame callback. A
compositor that answers callbacks "when something needs one" collapses to "when
someone asks", and its tenants stall, or stop altogether when nothing is
watching.

So glasspane never answers a callback on its own. `Glasspane::tick(now)` does,
and you call it **once per frame of your clock, unconditionally** — whether or
not anything looks like it needs it, whether or not the tenant is on screen,
whether or not anyone is looking. A tenant you stop ticking is a tenant you
have stopped.

### Import once per `wl_buffer`

Every double-buffered client re-commits the same two `wl_buffer`s forever.
Importing a dma-buf is a `vkAllocateMemory` and a `vkCreateImage`; doing it per
frame is a per-frame allocation for a resource that has not changed. glasspane
keeps a registry keyed by `wl_buffer` identity, so N commits of one buffer
produce one import and N−1 cache hits, and the entry is dropped when the client
destroys the buffer.

That rule and the next one pull in opposite directions — a client alternating
buffers would alternate textures — so there are two levels: the registry caches
the *imported* image per `wl_buffer`, each tenant owns one *destination*
texture, and a dma-buf commit is a blit between them. shm commits are a
`write_texture` straight into the destination and never enter the registry;
there is nothing to amortise.

### The texture handle is stable while the size is

`Tenant::texture()` is replaced only when the client's buffer size changes,
which is announced as `Event::Resized` *before* any `Event::Committed` against
the new handle. Bind it once, index it by `TenantId`, and do not rebind per
frame.

`TenantId` is the identity, always. Never identify a tenant by its position in
`tenants()`: map order is not insertion order, a tenant can unmap between two
frames, and "the new window is the last one in the list" is a bug that produces
real pixels under the wrong label.

## Popups

An `xdg_popup` — a `<select>` menu, an autocomplete list, a context menu, a
permission bubble — is handed over exactly as a toplevel is: its own
`PopupId`, its own texture, its own generation. What it has that a toplevel
has not is a parent and a place.

```rust
for popup in pane.popups() {
    let (x, y) = popup.position();   // top-left in popup.root()'s logical pixels
    let (w, h) = popup.size();
    let _ = popup.texture();         // draw it over the pane you drew root() on
}
```

`popups()` is a **separate list from `tenants()`**: a popup is not a tenant
and a shell that keeps a row of windows must not find a menu in it. The
position is always relative to the *root tenant*, never to an intermediate
popup, so a submenu three deep still hangs off the one quad you drew the
window on.

The positioner is resolved here — anchor, gravity, offset, and the client's
own `flip`/`slide`/`resize` permissions — against the root tenant's own
rectangle rather than an output, because a tenant in someone's scene has no
output and its rectangle is the only one both sides know the size of. So a
`<select>` near the foot of a page opens upwards. What is *not* decided here
is where the tenant is drawn, at what scale, or how far off its plane the
popup is lifted: that is placement, and it stays the shell's.

A grabbing popup is modal over its tenant. There is no seat grab to install —
this crate has no input devices, the shell injects every event — so what a
grab means is the one thing the client is waiting on: a pointer button pressed
outside the popup's chain sends `popup_done` and is swallowed, exactly as a
click outside a menu on a desktop closes the menu without also pressing what
was under it. Without that a Chromium `<select>` hangs waiting for an event
that never comes. `dismiss_popup` does the same for the cases that are not a
click.

## The input contract

Coordinates are **tenant-local logical pixels, origin top-left** — the space of
`Tenant::size()`, which is the size the tenant was configured to. A shell that
hit-tests a quad in its own scene converts to `[0, width) × [0, height)` of
that size and passes that; glasspane applies no transform, no scale and no
clamp. At scale 1 the texture is the same size. Above it
(`Glasspane::set_scale`) the texture is larger than `size` by the scale, so
sample it by UV and hit-test by `size`.

```rust
pane.pointer_enter(id, x, y)?;   // and pointer_motion / pointer_leave
pane.pointer_button(glasspane::input::button::LEFT, KeyState::Pressed);
pane.pointer_axis(0.0, 10.0, Some((0, 120)));
pane.set_keyboard_focus(Some(id))?;
pane.keyboard_key(Some(id), 30 /* KEY_A */, KeyState::Pressed)?;
```

A popup is aimed at the same way, through a `Target`, in **its own** logical
pixels rather than its parent's:

```rust
pane.pointer_enter_at(popup_id.into(), x, y)?;
pane.set_keyboard_focus_at(Some(popup_id.into()))?;
```

Keycodes are **evdev** codes, exactly as `linux/input-event-codes.h` and
`libinput` give them (`KEY_A` is 30). The +8 offset xkb wants is applied inside,
once. Passing an xkb keysym or an already-offset code is the classic way to get
a keyboard that types one key to the left.

## Testing

```
cargo test
```

Tests that need a GPU **skip** (they do not fail) when there is no
`/dev/dri/renderD*` or no Vulkan adapter — "there is no GPU here" and "the GPU
got it wrong" must not look alike. No display server is needed for any of them:
glasspane *is* the server, on a private socket, and the test client is an
in-process Wayland client on a thread.

The test instance keeps Vulkan validation on but turns off wgpu's object names
(`InstanceFlags::DEBUG`, which a debug build sets): with them, tests creating
devices in parallel crashed inside the Vulkan loader
([#6](https://github.com/oneilltomhq/glasspane/issues/6)).

## Measured here

Intel Iris Xe Graphics (RPL-U), Mesa 25.3.6, kernel 7.1.9, `/dev/dri/renderD128`.
17 unit tests, 29 integration tests and 2 doctests, all passing; the
integration tests take 4–6 s in parallel.

* shm and dma-buf tenant textures both **byte-exact over all 65 536 pixels** of
  a 256x256 lattice.
* One `wl_buffer` re-committed six times: **1 import, 5 cache hits**, and the
  acquire ledger reads `exports=6 waits=6 carried=6 superseded=0
  live_semaphores=1` — one semaphore per live buffer, not per frame.
* An unticked client: **1 commit, ever**, against 3 s of dispatch.
* `foot`: maps, is configured to 800x480, paints through `wl_shm`.
* Chromium 151 with `--ozone-platform=wayland`: maps in ~2 s, is configured to
  1280x800, and commits **dma-buf** — 3 imports and every later commit a cache
  hit, which is triple buffering, each one fenced. The page comes back
  pixel-correct through `examples/chromium_tenant.rs`. It logs
  `'--ozone-platform=wayland' is not compatible with Vulkan` from its own GPU
  process, which is about Chromium's internal rasteriser and not about this
  compositor; it renders through GL and shares the result as a dma-buf anyway.

## Status

v0.1. shm and single-plane `ARGB8888`/`XRGB8888`/`ABGR8888`/`XBGR8888`
dma-buf, `xdg_shell` toplevels and popups, server-side decorations refused
politely, pointer, keyboard and axis injection.

The clipboard (`wl_data_device`) and the primary selection
(`zwp_primary_selection_device_manager_v1`) are relayed between tenants: copy
in one and paste in another, or select in one and middle-click in another. The
selection follows keyboard focus, so a tenant is offered it once it has the
keyboard.
Data control (`ext_data_control_manager_v1` and `zwlr_data_control_manager_v1`)
reads and sets both without focus, so `wl-copy`, `wl-paste` and clipboard
managers work from any tenant. Each set is announced as
`Event::SelectionChanged`, saying which selection and nothing else.

A shell that is itself a Wayland window can hand glasspane its `wl_display`
with `Glasspane::bridge_clipboard`, and the clipboard then crosses to and from
the desktop: copy in a desktop app and paste in a tenant, or the other way.
It rides the window's own connection, so the desktop treats it as the window
copying and pasting, and it works only while the window has the keyboard.

Subsurfaces are composed into their toplevel's texture and take the pointer;
the keyboard stays on the toplevel.

A tenant draws at the scale it is asked (`Glasspane::set_scale`, sent as
`wp_fractional_scale_v1` and `wl_surface.preferred_buffer_scale`), and a
`wp_viewport` source rectangle is honoured.

Not yet: damage tracking (an shm commit uploads the whole buffer), multi-plane
formats, drag-and-drop beyond the plumbing to accept it, and KMS scanout of the
composited result.

## License

MIT
