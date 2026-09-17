# glasspane

**A tenant Wayland compositor as a library.** It speaks Wayland to real clients
— terminals, browsers — and its entire output is one `wgpu::Texture` per client
plus an input-injection API.

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
through `wgpu` and [`wgpu-dmabuf`](../wgpu-dmabuf).

## Three rules that are load-bearing

### The frame clock is yours to drive

A Wayland client paints **only** when it has been handed a frame callback. A
compositor that answers callbacks "when something needs one" collapses to "when
someone asks", and its tenants stall — measured in the prior art at 0.3–11
seconds per frame, and stopping altogether when nothing was watching.

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
    let (x, y) = popup.position();   // top-left in popup.root()'s buffer pixels
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

Coordinates are **tenant-local buffer pixels, origin top-left** — the same
space as the texture you sampled. A shell that hit-tests a quad in its own
scene converts to `[0, width) × [0, height)` and passes that; glasspane applies
no transform, no scale and no clamp.

```rust
pane.pointer_enter(id, x, y)?;   // and pointer_motion / pointer_leave
pane.pointer_button(glasspane::input::button::LEFT, KeyState::Pressed);
pane.pointer_axis(0.0, 10.0, Some((0, 120)));
pane.set_keyboard_focus(Some(id))?;
pane.keyboard_key(Some(id), 30 /* KEY_A */, KeyState::Pressed)?;
```

A popup is aimed at the same way, through a `Target`, in **its own** buffer
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

## Measured here

Intel Iris Xe Graphics (RPL-U), Mesa 25.3.6, kernel 7.1.9, `/dev/dri/renderD128`.
7 integration tests + 6 unit tests, all passing, 5.6 s.

* shm and dma-buf tenant textures both **byte-exact over all 65 536 pixels** of
  a 256x256 lattice.
* One `wl_buffer` re-committed six times: **1 import, 5 cache hits**, and the
  acquire ledger reads `exports=6 waits=6 carried=6 superseded=0
  live_semaphores=1` — one semaphore per live buffer, not per frame.
* An unticked client: **1 commit, ever**, against 3 s of dispatch.
* `foot`: maps, is configured to 798x480, paints through `wl_shm`.
* Chromium 142 with `--ozone-platform=wayland`: maps in ~3 s, is configured to
  1280x800, and commits **dma-buf** — 3 imports and 1 cache hit over 4 commits,
  which is triple buffering, each one fenced. The page comes back pixel-correct
  through `examples/chromium_tenant.rs`. It logs
  `'--ozone-platform=wayland' is not compatible with Vulkan` from its own GPU
  process, which is about Chromium's internal rasteriser and not about this
  compositor; it renders through GL and shares the result as a dma-buf anyway.

## Status

v0.1. shm and single-plane `ARGB8888`/`XRGB8888` dma-buf, `xdg_shell` toplevels
and popups, server-side decorations refused politely, pointer, keyboard and
axis injection.

Not yet: subsurfaces, damage tracking
(an shm commit uploads the whole buffer), multi-plane formats, clipboard and
drag-and-drop beyond the plumbing to accept them, `wp_viewport`, fractional
scale, and KMS scanout of the composited result.

## Publishing

`glasspane` depends on `wgpu-dmabuf` by path and by version. Until that crate
is on crates.io, `cargo package` stops at "no matching package named
`wgpu-dmabuf`"; everything before that step — the manifest, the file list,
`cargo doc`, `cargo clippy --all-targets` — is clean.

## License

MIT
