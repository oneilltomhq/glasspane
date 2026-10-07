//! What glasspane claims, checked against a real client on a real socket.
//!
//! No Wayland session is needed to run any of this: glasspane *is* the server.
//! What is needed is a GPU, and the tests that need one **skip** rather than
//! fail when there is no DRM render node or no Vulkan adapter — "there is no
//! GPU here" and "the GPU got it wrong" must not look alike in a log.

mod fixture;

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use fixture::{Fixture, Report, Transport};
use glasspane::{Caps, Event, Glasspane, KeyState, TenantId};

/// A GPU that can import a dma-buf *and* import an acquire fence, or `None`.
fn gpu() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    if wgpu_dmabuf::testkit::render_node().is_none() {
        eprintln!("SKIP: no /dev/dri/renderD* on this machine");
        return None;
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .ok()?;
    match wgpu_dmabuf::device::open_device(&adapter, &Default::default()) {
        Ok((device, queue, caps)) => {
            if !caps.dmabuf {
                eprintln!("SKIP: adapter has no VULKAN_EXTERNAL_MEMORY_DMA_BUF");
                return None;
            }
            Some((adapter, device, queue))
        }
        Err(e) => {
            eprintln!("SKIP: cannot open a dma-buf capable device: {e}");
            None
        }
    }
}

/// A socket name no other test in this process (or run) will take.
fn socket_name(tag: &str) -> String {
    static N: AtomicU32 = AtomicU32::new(0);
    format!(
        "glasspane-test-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    )
}

/// Pump the compositor, driving the frame clock, until `done` says stop.
///
/// The tick is unconditional and on every turn, which is the contract: a
/// client paints only when handed a frame callback, so a pump that ticked
/// "when something needed it" would never tick at all.
fn pump(
    pane: &mut Glasspane,
    events: &mut Vec<Event>,
    timeout: Duration,
    mut done: impl FnMut(&Glasspane, &[Event]) -> bool,
) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        pane.dispatch(Some(Duration::from_millis(4))).unwrap();
        pane.tick(Instant::now());
        events.extend(pane.events());
        if done(pane, events) {
            return true;
        }
    }
    false
}

/// Read a tenant's texture back into a tightly packed `w * h * 4` vector, in
/// memory order.
fn read_back(pane: &Glasspane, id: TenantId) -> Vec<u8> {
    let tenant = pane.tenant(id).expect("tenant");
    let texture = tenant.texture().expect("texture");
    let (w, h) = (texture.width(), texture.height());
    let row = w * 4;
    assert_eq!(
        row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        0,
        "this helper does not unpad rows; the fixture picks widths that need no padding"
    );
    let device = pane.device();
    let queue = pane.queue();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("glasspane readback"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let out = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    out
}

/// The identity of a tenant's texture, as a shell that cached a handle would
/// see it: the allocation itself, not its contents.
fn handle(pane: &Glasspane, id: TenantId) -> *const wgpu::Texture {
    std::sync::Arc::as_ptr(pane.tenant(id).unwrap().texture_arc().unwrap())
}

/// Where the first mismatching pixel is, or `None`.
fn first_difference(got: &[u8], want: &[u8], width: u32) -> Option<String> {
    assert_eq!(got.len(), want.len(), "readback length");
    for (i, (g, w)) in got.chunks_exact(4).zip(want.chunks_exact(4)).enumerate() {
        if g != w {
            let (x, y) = (i as u32 % width, i as u32 / width);
            return Some(format!("at ({x},{y}): got {g:?}, want {w:?}"));
        }
    }
    None
}

fn start(tag: &str, adapter: Option<&wgpu::Adapter>, device: wgpu::Device, queue: wgpu::Queue)
    -> Glasspane
{
    let name = socket_name(tag);
    let mut caps = Caps::new().with_socket_name(name).with_size(256, 256);
    if let Some(a) = adapter {
        caps = caps.with_adapter(a);
    }
    Glasspane::new(device, queue, caps).expect("glasspane")
}

/// Bring a fixture up and return its tenant id once it has committed.
fn map_tenant(
    pane: &mut Glasspane,
    transport: Transport,
    events: &mut Vec<Event>,
) -> (Fixture, TenantId) {
    let client = Fixture::start(pane.socket_name(), transport);
    // Only events from here on: an earlier tenant's `Mapped` is still in the
    // log, and picking it up would make two clients look like one.
    let from = events.len();
    let mut id = None;
    let up = pump(pane, events, Duration::from_secs(10), |_, evs| {
        let evs = &evs[from..];
        if id.is_none() {
            id = evs.iter().find_map(|e| match e {
                Event::Mapped { id, .. } => Some(*id),
                _ => None,
            });
        }
        match id {
            Some(id) => evs.iter().any(|e| {
                matches!(e, Event::Committed { id: c, generation } if *c == id && *generation >= 1)
            }),
            None => false,
        }
    });
    assert!(
        up,
        "the fixture never mapped and committed; it reported {:?}; compositor events {events:?}",
        client.drain()
    );
    (client, id.expect("mapped"))
}

// ---------------------------------------------------------------------------

/// **The shm path, byte for byte.**
///
/// A client paints a lattice that is a pure function of `(x, y)`, commits it
/// through `wl_shm`, and the texture the shell would sample is read back and
/// compared over all 65 536 pixels. A wrong stride, a swapped channel order or
/// an off-by-one row each produce a texture that samples cleanly and shows the
/// wrong picture, and each one fails here.
#[test]
fn an_shm_tenant_texture_is_byte_exact() {
    let Some((_adapter, device, queue)) = gpu() else {
        return;
    };
    let mut pane = start("shm", None, device, queue);
    assert!(
        !pane.advertises_dmabuf(),
        "no adapter was given, so there must be no dma-buf global"
    );
    let mut events = Vec::new();
    let (client, id) = map_tenant(&mut pane, Transport::Shm, &mut events);

    let (w, h) = pane.tenant(id).unwrap().size();
    assert_eq!((w, h), (256, 256));
    let got = read_back(&pane, id);
    assert_eq!(
        first_difference(&got, &fixture::expected(w, h), w),
        None,
        "the tenant texture is not what the client painted"
    );

    // And the client is painting because it is being ticked, not because
    // anybody asked: it has been handed frame callbacks and has used them.
    let (callbacks, ticks) = pane.frame_stats();
    assert!(callbacks > 0, "no frame callbacks were answered in {ticks} ticks");
    assert!(
        client.drain().iter().any(|r| matches!(r, Report::Committed { n, .. } if *n >= 2)),
        "the client never repainted, so the frame clock drove nothing"
    );
}

/// **The dma-buf path, byte for byte**, through a real GBM buffer object.
#[test]
fn a_dmabuf_tenant_texture_is_byte_exact() {
    let Some((adapter, device, queue)) = gpu() else {
        return;
    };
    let mut pane = start("dmabuf", Some(&adapter), device, queue);
    if !pane.advertises_dmabuf() {
        eprintln!("SKIP: this adapter shares no single-plane ARGB8888 modifiers");
        return;
    }
    let mut events = Vec::new();
    let (_client, id) = map_tenant(&mut pane, Transport::Dmabuf, &mut events);

    let (w, h) = pane.tenant(id).unwrap().size();
    let got = read_back(&pane, id);
    assert_eq!(
        first_difference(&got, &fixture::expected(w, h), w),
        None,
        "the imported dma-buf is not what the client filled"
    );
    assert!(
        pane.tenant(id).unwrap().acquire_staged(),
        "a dma-buf commit must have staged an acquire wait"
    );

    // **Import once.** The client re-commits the same wl_buffer on every frame
    // callback, so after N commits there must still be exactly one import.
    let generation = pane.tenant(id).unwrap().generation();
    pump(&mut pane, &mut events, Duration::from_secs(3), |p, _| {
        p.tenant(id).map(|t| t.generation()).unwrap_or(0) >= generation + 5
    });
    let (imports, hits) = pane.import_stats();
    assert_eq!(imports, 1, "one wl_buffer must produce exactly one import");
    assert!(hits >= 5, "the re-commits must have hit the cache, got {hits}");

    let report = pane.acquire_report().expect("this device can fence");
    assert!(
        report.exports >= 6,
        "every dma-buf commit exports a fence: {report}"
    );
    assert_eq!(
        report.live_semaphores, 1,
        "the semaphore pool is bounded by live buffers, not by frames: {report}"
    );
    eprintln!("{report}");
}

/// **The other byte order, byte for byte.** An `ABGR8888` dma-buf, which is
/// what Android commits, lands in the tenant texture as the same lattice an
/// `ARGB8888` one does: drawn across rather than copied, and not
/// channel-swapped on the way.
#[test]
fn an_abgr_dmabuf_tenant_texture_is_byte_exact() {
    let Some((adapter, device, queue)) = gpu() else {
        return;
    };
    let mut pane = start("dmabuf-abgr", Some(&adapter), device, queue);
    if !pane.advertises_dmabuf() {
        eprintln!("SKIP: this adapter shares no single-plane 8888 modifiers");
        return;
    }
    let mut events = Vec::new();
    let (_client, id) = map_tenant(&mut pane, Transport::DmabufAbgr, &mut events);

    let tenant = pane.tenant(id).unwrap();
    assert_eq!(tenant.texture().unwrap().format(), glasspane::texture::TENANT_FORMAT);
    let (w, h) = tenant.size();
    let got = read_back(&pane, id);
    assert_eq!(
        first_difference(&got, &fixture::expected(w, h), w),
        None,
        "the ABGR8888 dma-buf did not land as the lattice the client filled"
    );
}

/// **The texture handle is stable while the size is.**
///
/// Ten commits of a client that never resizes must land in one texture. A
/// shell that indexed this handle and never rebound has to stay correct.
#[test]
fn the_texture_handle_survives_recommits() {
    let Some((_adapter, device, queue)) = gpu() else {
        return;
    };
    let mut pane = start("stable", None, device, queue);
    let mut events = Vec::new();
    let (_client, id) = map_tenant(&mut pane, Transport::Shm, &mut events);
    let first = handle(&pane, id);

    pump(&mut pane, &mut events, Duration::from_secs(5), |p, _| {
        p.tenant(id).map(|t| t.generation()).unwrap_or(0) >= 10
    });
    assert!(
        pane.tenant(id).unwrap().generation() >= 10,
        "the client did not reach ten commits"
    );
    assert_eq!(
        handle(&pane, id),
        first,
        "the texture was replaced without the size changing"
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Event::Resized { id: r, .. } if *r == id))
            .count(),
        1,
        "exactly one Resized, for the first commit that established the size"
    );
}

/// **A click reaches the client in tenant-local coordinates.**
///
/// The compositor is told `(12.5, 7.5)` inside a tenant; the *client* must
/// report those exact numbers, and then a button. Both are checked on the
/// client's side of the socket, because the compositor's own record of what it
/// sent is not evidence that anything was sent.
#[test]
fn a_pointer_click_arrives_where_it_was_aimed() {
    let Some((_adapter, device, queue)) = gpu() else {
        return;
    };
    let mut pane = start("pointer", None, device, queue);
    let mut events = Vec::new();
    let (client, id) = map_tenant(&mut pane, Transport::Shm, &mut events);

    pane.pointer_enter(id, 12.5, 7.5).unwrap();
    pane.pointer_button(glasspane::input::button::LEFT, KeyState::Pressed);
    pane.pointer_button(glasspane::input::button::LEFT, KeyState::Released);
    pump(&mut pane, &mut events, Duration::from_millis(500), |_, _| false);

    // wl_fixed carries 1/256ths, so .5 is exact and a mismatch here is a
    // transform, not a rounding error.
    client.expect(Duration::from_secs(2), |r| {
        matches!(r, Report::PointerEnter(x, y) if *x == 12.5 && *y == 7.5)
    });
    client.expect(Duration::from_secs(2), |r| {
        matches!(r, Report::PointerButton(b, true) if *b == glasspane::input::button::LEFT)
    });
    client.expect(Duration::from_secs(2), |r| {
        matches!(r, Report::PointerButton(b, false) if *b == glasspane::input::button::LEFT)
    });
}

/// **A key arrives as the same evdev code it was sent as.**
#[test]
fn a_key_survives_the_xkb_offset() {
    let Some((_adapter, device, queue)) = gpu() else {
        return;
    };
    let mut pane = start("keyboard", None, device, queue);
    let mut events = Vec::new();
    let (client, id) = map_tenant(&mut pane, Transport::Shm, &mut events);

    // KEY_A is 30 in linux/input-event-codes.h, and 30 is what wl_keyboard.key
    // must carry. 38 would be the same key with the xkb offset applied twice.
    pane.keyboard_key(Some(id), 30, KeyState::Pressed).unwrap();
    pane.keyboard_key(Some(id), 30, KeyState::Released).unwrap();
    pump(&mut pane, &mut events, Duration::from_millis(500), |_, _| false);

    client.expect(Duration::from_secs(2), |r| *r == Report::Key(30, true));
    client.expect(Duration::from_secs(2), |r| *r == Report::Key(30, false));
    assert_eq!(pane.keyboard_focus(), Some(id));
}

/// **A resize is reported by id, never by list position.**
///
/// Two tenants, and the *second* one is resized. The assertion is on the
/// tenant with that id — the one that was told — and on the other one being
/// untouched. Identifying "the resized one" as the last entry of a listing is
/// the bug this is written against: map order is not insertion order, and that
/// mistake produces real pixels under the wrong label.
#[test]
fn a_resize_is_reported_by_id_not_by_list_position() {
    let Some((_adapter, device, queue)) = gpu() else {
        return;
    };
    let mut pane = start("resize", None, device, queue);
    let mut events = Vec::new();
    let (_a, first) = map_tenant(&mut pane, Transport::Shm, &mut events);
    let (_b, second) = map_tenant(&mut pane, Transport::Shm, &mut events);
    assert_ne!(first, second);
    assert_eq!(pane.tenants().len(), 2);

    pane.configure(second, (512, 128)).unwrap();
    assert_eq!(
        pane.tenant(second).unwrap().size(),
        (256, 256),
        "configure is a request, not a fact: the size is still the painted one"
    );
    assert_eq!(pane.tenant(second).unwrap().configured(), Some((512, 128)));

    let resized = pump(&mut pane, &mut events, Duration::from_secs(10), |_, evs| {
        evs.iter()
            .any(|e| matches!(e, Event::Resized { id, size } if *id == second && *size == (512, 128)))
    });
    assert!(resized, "the client never painted at the new size: {events:?}");

    assert_eq!(pane.tenant(second).unwrap().size(), (512, 128));
    assert_eq!(
        pane.tenant(first).unwrap().size(),
        (256, 256),
        "the tenant that was not configured must not have moved"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Event::Resized { id, size } if *id == first && *size != (256, 256))),
        "the untouched tenant was reported as resized"
    );

    // And the new size is really in the texture, not just in the bookkeeping.
    let got = read_back(&pane, second);
    assert_eq!(first_difference(&got, &fixture::expected(512, 128), 512), None);
}

/// **Nothing paints without a tick.**
///
/// The frame-starvation trap, as an assertion. The same client, dispatched
/// exactly as hard but never ticked, must stall after its first commit —
/// because a Wayland client paints only when handed a frame callback, and the
/// only thing that hands one out is [`Glasspane::tick`].
#[test]
fn a_client_that_is_never_ticked_stalls_after_one_frame() {
    let Some((_adapter, device, queue)) = gpu() else {
        return;
    };
    let mut pane = start("starve", None, device, queue);
    let client = Fixture::start(pane.socket_name(), Transport::Shm);

    let deadline = Instant::now() + Duration::from_secs(3);
    let mut id = None;
    while Instant::now() < deadline {
        pane.dispatch(Some(Duration::from_millis(4))).unwrap();
        // No tick. That is the experiment.
        for e in pane.events() {
            if let Event::Mapped { id: m, .. } = e {
                id = Some(m);
            }
        }
    }
    let id = id.expect("the toplevel still maps — only painting is starved");
    assert_eq!(
        pane.tenant(id).unwrap().generation(),
        1,
        "an unticked client painted more than the one frame it starts with"
    );
    assert_eq!(pane.frame_stats().0, 0, "no callbacks can have been answered");
    let commits = client
        .drain()
        .iter()
        .filter(|r| matches!(r, Report::Committed { .. }))
        .count();
    assert_eq!(commits, 1, "the client committed {commits} times unticked");
}

/// `wl-copy` and `wl-paste` from wl-clipboard, or `None` if either is missing.
///
/// Neither is a dev-dependency a crate can declare, so a machine without them
/// skips, and says so, rather than failing.
fn wl_clipboard() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let find = |name: &str| {
        std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|d| d.join(name))
                .find(|p| p.is_file())
        })
    };
    match (find("wl-copy"), find("wl-paste")) {
        (Some(c), Some(p)) => Some((c, p)),
        _ => {
            eprintln!("SKIP: wl-clipboard (wl-copy, wl-paste) is not installed");
            None
        }
    }
}

/// A wl-clipboard process on this pane's socket, and nothing of the session's.
fn wl_clipboard_command(program: &std::path::Path, pane: &Glasspane) -> std::process::Command {
    let mut cmd = std::process::Command::new(program);
    cmd.env("WAYLAND_DISPLAY", pane.socket_name())
        .env_remove("WAYLAND_SOCKET")
        .env_remove("DISPLAY")
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    cmd
}

/// Pump until a new toplevel maps, and give it the keyboard.
///
/// wl-clipboard has no data-control protocol to use here, so it does what any
/// client must: map a surface, wait for keyboard focus, and use that serial.
fn map_and_focus(pane: &mut Glasspane, events: &mut Vec<Event>, what: &str) -> TenantId {
    let from = events.len();
    let mut id = None;
    pump(pane, events, Duration::from_secs(10), |_, evs| {
        id = evs[from..].iter().find_map(|e| match e {
            Event::Mapped { id, .. } => Some(*id),
            _ => None,
        });
        id.is_some()
    });
    let id = id.unwrap_or_else(|| panic!("{what} never mapped a toplevel: {events:?}"));
    pane.set_keyboard_focus(Some(id)).unwrap();
    id
}

/// Copy with one wl-clipboard client, paste with another, and return what the
/// paste printed.
fn copy_then_paste(tag: &str, flag: &[&str], text: &str) -> Option<String> {
    let (copy, paste) = wl_clipboard()?;
    let (_adapter, device, queue) = gpu()?;
    let mut pane = start(tag, None, device, queue);
    let mut events = Vec::new();

    // --foreground so the process is ours to kill; by default wl-copy forks
    // and serves the selection from a child this test would leak.
    let mut copier = wl_clipboard_command(&copy, &pane)
        .arg("--foreground")
        .args(flag)
        .arg(text)
        .stdout(std::process::Stdio::null())
        .spawn()
        .expect("spawn wl-copy");
    map_and_focus(&mut pane, &mut events, "wl-copy");
    // wl-copy sets the selection on keyboard enter, then unmaps its surface.
    // The selection outlives the surface; the source is what serves it.
    pump(&mut pane, &mut events, Duration::from_secs(5), |_, evs| {
        evs.iter().any(|e| matches!(e, Event::Unmapped { .. }))
    });

    let mut paster = wl_clipboard_command(&paste, &pane)
        .arg("--no-newline")
        .args(flag)
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("spawn wl-paste");
    let mut out = paster.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut s = String::new();
        let _ = std::io::Read::read_to_string(&mut out, &mut s);
        let _ = tx.send(s);
    });
    map_and_focus(&mut pane, &mut events, "wl-paste");
    // The transfer is client to client, but every request of it goes through
    // the compositor, so the pane must keep dispatching until the pipe closes.
    let mut got = None;
    pump(&mut pane, &mut events, Duration::from_secs(10), |_, _| {
        got = rx.try_recv().ok();
        got.is_some()
    });

    let _ = copier.kill();
    let _ = copier.wait();
    let _ = paster.kill();
    let status = paster.wait().unwrap();
    let mut err = String::new();
    let _ = std::io::Read::read_to_string(paster.stderr.as_mut().unwrap(), &mut err);
    Some(got.unwrap_or_else(|| panic!("wl-paste never finished ({status}): {err}; {events:?}")))
}

/// **A primary selection crosses from one tenant to another.**
///
/// Select-then-middle-click, as two wl-clipboard processes: one owns the
/// `zwp_primary_selection_v1` selection, a second one, focused afterwards, is
/// offered it and reads it. Without the primary selection global wl-paste
/// refuses outright; without `set_primary_focus` on a focus change it is
/// never offered anything.
#[test]
fn a_primary_selection_crosses_between_tenants() {
    if let Some(got) = copy_then_paste("primary", &["--primary"], "hello") {
        assert_eq!(got, "hello");
    }
}

/// **The clipboard crosses from one tenant to another**, the same way.
#[test]
fn a_clipboard_selection_crosses_between_tenants() {
    if let Some(got) = copy_then_paste("clipboard", &[], "hello clipboard") {
        assert_eq!(got, "hello clipboard");
    }
}

/// `foot` on `PATH`, or `None`, saying so.
fn foot() -> Option<std::path::PathBuf> {
    let found = std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join("foot"))
            .find(|p| p.is_file())
    });
    if found.is_none() {
        eprintln!("SKIP: foot is not installed");
    }
    found
}

/// **A tenant asked to draw at 2 paints twice the pixels at the same size.**
///
/// Against a real client that speaks `wp_fractional_scale_v1` and
/// `wp_viewporter`, because the claim is about what a client does with the
/// ask: the texture doubles in each direction, and the logical size — the
/// space input and popups are in — does not move. Then a fractional scale,
/// which only the fractional protocol can carry, and back to 1.
#[test]
fn a_scaled_tenant_paints_more_pixels_at_the_same_size() {
    let Some(foot) = foot() else {
        return;
    };
    let Some((adapter, device, queue)) = gpu() else {
        return;
    };
    let mut pane = start("scale", Some(&adapter), device, queue);
    let mut events = Vec::new();
    let mut cmd = std::process::Command::new(foot);
    // No user config: a font size or a `dpi-aware` from the machine running
    // the test is not what is under test. Its default `resize-by-cells=yes`
    // is kept on purpose: a floating foot snaps 520 to whole cells, and on a
    // scale change keeps its grid and resizes around the new font. A tiled
    // one takes the size it is given, and every tenant here is tiled.
    cmd.args(["--config", "/dev/null"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    let mut child = pane.spawn(cmd).expect("spawn foot");
    let id = map_and_focus(&mut pane, &mut events, "foot");
    pane.configure(id, (390, 520)).unwrap();

    let settle = |pane: &mut Glasspane, events: &mut Vec<Event>, buffer: (u32, u32)| {
        let ok = pump(pane, events, Duration::from_secs(10), |pane, _| {
            pane.tenant(id)
                .is_some_and(|t| t.size() == (390, 520) && t.buffer_size() == buffer)
        });
        let t = pane.tenant(id).unwrap();
        assert!(
            ok,
            "wanted 390x520 over a {buffer:?} buffer, got {:?} over {:?}",
            t.size(),
            t.buffer_size()
        );
        let tex = t.texture().unwrap();
        assert_eq!((tex.width(), tex.height()), buffer, "the texture is the buffer");
    };

    settle(&mut pane, &mut events, (390, 520));
    pane.set_scale(id, 2.0).unwrap();
    assert_eq!(pane.tenant(id).unwrap().scale(), 2.0);
    settle(&mut pane, &mut events, (780, 1040));
    let from = events.len();
    pane.set_scale(id, 3.0).unwrap();
    settle(&mut pane, &mut events, (1170, 1560));
    assert!(
        events[from..]
            .iter()
            .any(|e| matches!(e, Event::Resized { id: r, size: (390, 520) } if *r == id)),
        "the new buffer, at the same size, was not announced: {events:?}"
    );
    pane.set_scale(id, 2.1).unwrap();
    settle(&mut pane, &mut events, (819, 1092));
    pane.set_scale(id, 1.0).unwrap();
    settle(&mut pane, &mut events, (390, 520));

    assert!(matches!(
        pane.set_scale(id, 0.0),
        Err(glasspane::Error::BadScale(_))
    ));
    let _ = child.kill();
    let _ = child.wait();
}

/// **Two tenants, two clocks.** `tick_where` hands callbacks to the tenants
/// it picks and holds the rest: a tenant left out stops painting with its
/// callback pending, the one picked goes on, and the tenant left out paints
/// again the first tick it is picked, on the callback it was holding. The
/// per-tenant count says which got what. What a shell pacing the panes it
/// cannot see stands on.
#[test]
fn two_tenants_callbacks_are_released_apart() {
    let Some((_adapter, device, queue)) = gpu() else {
        return;
    };
    let mut pane = start("pace", None, device, queue);
    let mut events = Vec::new();
    let (_a_client, a) = map_tenant(&mut pane, Transport::Shm, &mut events);
    let (_b_client, b) = map_tenant(&mut pane, Transport::Shm, &mut events);
    // Both are painting, on `tick`; now only `a` is picked, and every tenant
    // asked about is one holding a callback.
    let turn = |pane: &mut Glasspane, pick: TenantId, asked: &mut Vec<TenantId>| {
        pane.dispatch(Some(Duration::from_millis(4))).unwrap();
        pane.tick_where(Instant::now(), |id| {
            asked.push(id);
            id == pick
        });
        let _ = pane.events().count();
    };
    let mut asked = Vec::new();
    // Let `b`'s last paid frame land, so what follows starts from rest.
    for _ in 0..20 {
        turn(&mut pane, a, &mut asked);
    }
    let gen = |pane: &Glasspane, id: TenantId| pane.tenant(id).unwrap().generation();
    let sent = |pane: &Glasspane, id: TenantId| pane.tenant(id).unwrap().callbacks_sent();
    let (a0, b0, a_sent, b_sent) = (gen(&pane, a), gen(&pane, b), sent(&pane, a), sent(&pane, b));
    // Until the picked one has painted ten more frames: long enough for the
    // other to have painted as many, had it been paid.
    let deadline = Instant::now() + Duration::from_secs(5);
    while gen(&pane, a) < a0 + 10 && Instant::now() < deadline {
        turn(&mut pane, a, &mut asked);
    }
    assert!(gen(&pane, a) >= a0 + 10, "the picked tenant painted {} frames in 5 s", gen(&pane, a) - a0);
    assert!(sent(&pane, a) >= a_sent + 10, "the picked tenant was handed {} callbacks", sent(&pane, a) - a_sent);
    assert_eq!(gen(&pane, b), b0, "the tenant left out painted");
    assert_eq!(sent(&pane, b), b_sent, "the tenant left out was handed a callback");
    assert!(asked.contains(&b), "the tenant left out was never asked about, so it held no callback");
    let (callbacks, _) = pane.frame_stats();
    assert_eq!(callbacks, sent(&pane, a) + sent(&pane, b), "the run's total is the tenants' sum");

    // Picked once: the held callback is answered, and `b` paints again.
    let b1 = gen(&pane, b);
    turn(&mut pane, b, &mut asked);
    assert_eq!(sent(&pane, b), b_sent + 1, "the held callback, answered on the first tick picked");
    let deadline = Instant::now() + Duration::from_secs(2);
    while gen(&pane, b) == b1 && Instant::now() < deadline {
        turn(&mut pane, b, &mut asked);
    }
    assert!(gen(&pane, b) > b1, "the tenant picked again never painted");
}
