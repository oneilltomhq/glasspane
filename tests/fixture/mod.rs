//! The test fixture: a real Wayland client, on a real socket, in a thread.
//!
//! glasspane *is* the server, so this suite needs no Wayland session of its
//! own — but it does need a client, and the client has to be one whose pixels
//! the test knows in advance. This is that client: it paints a lattice that is
//! a pure function of `(x, y)` so any committed frame is byte-identical to any
//! other, and it reports every input event it receives down a channel so the
//! test can assert on what the client *saw* rather than on what the compositor
//! believes it sent.
//!
//! It runs in a thread rather than a child process on purpose. A child would
//! have to report through a pipe and be parsed; a thread reports through an
//! `mpsc::Sender`, which cannot be half-written or interleaved. It is no less
//! a real client for it: it connects over the same unix socket with the same
//! `wayland-client` any other client uses, and glasspane knows nothing about
//! it.
//!
//! Stopping it is by socket closure: dropping the `Glasspane` closes the
//! listening socket and the client's connection with it, the dispatch loop
//! errors, and the thread ends. There is no stop flag to forget to set.

#![allow(dead_code)]

use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use wgpu_dmabuf::format::Fourcc;

use wayland_client::protocol::{
    wl_buffer::WlBuffer, wl_callback, wl_compositor::WlCompositor, wl_keyboard, wl_pointer,
    wl_registry, wl_seat, wl_shm, wl_shm_pool, wl_surface,
};
use wayland_client::{Connection, Dispatch, Proxy, QueueHandle, delegate_noop};
use wayland_protocols::wp::linux_dmabuf::zv1::client::{
    zwp_linux_buffer_params_v1, zwp_linux_dmabuf_v1,
};
use wayland_protocols::xdg::shell::client::{
    xdg_surface::{self, XdgSurface},
    xdg_toplevel::{self, XdgToplevel},
    xdg_wm_base::{self, XdgWmBase},
};

/// Which transport the fixture commits through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// `wl_shm`: a memfd pool the client repaints in place.
    Shm,
    /// `zwp_linux_dmabuf_v1`: one GBM `LINEAR` buffer object, filled on the
    /// CPU and committed.
    Dmabuf,
    /// The same, in `ABGR8888`: the lattice with its bytes in the other
    /// order, as Android's gralloc commits.
    DmabufAbgr,
}

/// What the fixture tells the test.
#[derive(Debug, Clone, PartialEq)]
pub enum Report {
    /// `xdg_surface.configure`, with whatever size the compositor asked for
    /// (`(0, 0)` means "you choose").
    Configured(i32, i32),
    /// A commit went out, carrying a buffer of this size. `n` counts them.
    Committed { n: u64, w: u32, h: u32 },
    /// `wl_pointer.enter`, in **surface-local** coordinates.
    PointerEnter(f64, f64),
    /// `wl_pointer.motion`.
    PointerMotion(f64, f64),
    /// `wl_pointer.button`: evdev code, and whether it went down.
    PointerButton(u32, bool),
    /// `wl_pointer.axis`: horizontal, vertical.
    PointerAxis(f64, f64),
    /// `wl_keyboard.key`: the **evdev** code as it arrived on the wire, and
    /// whether it went down.
    Key(u32, bool),
    /// `xdg_toplevel.close`.
    CloseRequested,
    /// The client could not start. The string is why.
    Failed(String),
}

/// The lattice the fixture paints, in **memory order** for `ARGB8888`, i.e.
/// `[B, G, R, A]`.
///
/// A pure function of position and nothing else: no frame counter, no seed, no
/// animation. That is what makes "read the texture back and compare all of it"
/// an identity rather than a race — any committed frame is every committed
/// frame. The channels are deliberately unequal functions of x and y, so a
/// transposed, mirrored or channel-swapped readback fails rather than passing
/// on a symmetry.
pub fn pattern(x: u32, y: u32) -> [u8; 4] {
    let b = (x.wrapping_mul(7).wrapping_add(11) % 251) as u8;
    let g = (y.wrapping_mul(13).wrapping_add(29) % 241) as u8;
    let r = ((x ^ y).wrapping_mul(5).wrapping_add(3) % 239) as u8;
    [b, g, r, 255]
}

/// The whole expected buffer, tightly packed, for a size.
pub fn expected(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            out.extend_from_slice(&pattern(x, y));
        }
    }
    out
}

/// A running fixture client.
pub struct Fixture {
    /// Everything the client saw, in order.
    pub reports: Receiver<Report>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Fixture {
    /// Connect a client to `socket_name` and start painting.
    ///
    /// The connection is made by path rather than through `WAYLAND_DISPLAY`:
    /// the environment is process-wide, and two tests running in parallel that
    /// both set it are two tests that can hand a client the wrong compositor.
    pub fn start(socket_name: &std::ffi::OsStr, transport: Transport) -> Self {
        let path = socket_path(socket_name);
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            if let Err(why) = run(path, transport, tx.clone(), &flag) {
                let _ = tx.send(Report::Failed(why));
            }
        });
        Fixture {
            reports: rx,
            stop,
            handle: Some(handle),
        }
    }

    /// Wait for a report matching `want`, or panic saying what did arrive.
    pub fn expect(&self, timeout: Duration, want: impl Fn(&Report) -> bool) -> Report {
        let deadline = std::time::Instant::now() + timeout;
        let mut seen = Vec::new();
        loop {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            match self.reports.recv_timeout(left.max(Duration::from_millis(1))) {
                Ok(r) => {
                    if want(&r) {
                        return r;
                    }
                    seen.push(r);
                }
                Err(_) => panic!("fixture never reported it; it did report {seen:?}"),
            }
            if std::time::Instant::now() >= deadline {
                panic!("fixture never reported it; it did report {seen:?}");
            }
        }
    }

    /// Drain whatever has arrived so far.
    pub fn drain(&self) -> Vec<Report> {
        self.reports.try_iter().collect()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // Ask, then wait. The client thread is not blocked in a dispatch that
        // only the compositor's death could end — it polls with a timeout
        // precisely so that a test can drop its client *before* its
        // `Glasspane` and still finish.
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn socket_path(socket_name: &std::ffi::OsStr) -> PathBuf {
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        .unwrap_or_else(|| format!("/run/user/{}", unsafe { libc::getuid() }).into());
    PathBuf::from(runtime).join(socket_name)
}

// ---------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------

struct Client {
    tx: Sender<Report>,
    transport: Transport,
    qh: QueueHandle<Client>,

    compositor: Option<WlCompositor>,
    shm: Option<wl_shm::WlShm>,
    dmabuf: Option<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1>,
    wm_base: Option<XdgWmBase>,
    seat: Option<wl_seat::WlSeat>,

    surface: Option<wl_surface::WlSurface>,
    xdg_surface: Option<XdgSurface>,
    toplevel: Option<XdgToplevel>,

    /// The current buffer and the size it was made at. Both are needed: a
    /// pool big enough is not a buffer the right shape — 512x128 and 256x256
    /// are the same number of bytes.
    buffer: Option<WlBuffer>,
    buffer_size: (u32, u32),
    size: (u32, u32),
    /// The shm pool's mapping, so a repaint is a memcpy rather than a new pool.
    pool: Option<ShmPool>,
    /// Held for its lifetime: the exported fd keeps the memory alive, but the
    /// bo is what gbm needs to still exist.
    gbm: Option<wgpu_dmabuf::testkit::GbmBuffer>,

    commits: u64,
    configured: bool,
    running: bool,
}

struct ShmPool {
    pool: wl_shm_pool::WlShmPool,
    map: *mut u8,
    len: usize,
}

impl Drop for ShmPool {
    fn drop(&mut self) {
        // SAFETY: `map`/`len` are exactly what mmap returned for this pool.
        unsafe { libc::munmap(self.map.cast(), self.len) };
        self.pool.destroy();
    }
}

fn run(
    path: PathBuf,
    transport: Transport,
    tx: Sender<Report>,
    stop: &AtomicBool,
) -> Result<(), String> {
    let stream = UnixStream::connect(&path)
        .map_err(|e| format!("connect {}: {e}", path.display()))?;
    let conn = Connection::from_socket(stream).map_err(|e| format!("wayland connection: {e}"))?;
    let display = conn.display();
    let mut queue = conn.new_event_queue();
    let qh = queue.handle();
    display.get_registry(&qh, ());

    let mut client = Client {
        tx,
        transport,
        qh: qh.clone(),
        compositor: None,
        shm: None,
        dmabuf: None,
        wm_base: None,
        seat: None,
        surface: None,
        xdg_surface: None,
        toplevel: None,
        buffer: None,
        buffer_size: (0, 0),
        size: (0, 0),
        pool: None,
        gbm: None,
        commits: 0,
        configured: false,
        running: true,
    };

    // Two roundtrips: the first delivers the globals, the second the
    // `wl_shm.format` and `zwp_linux_dmabuf_v1.modifier` events that follow
    // from binding them.
    queue
        .roundtrip(&mut client)
        .map_err(|e| format!("registry roundtrip: {e}"))?;
    if transport != Transport::Shm && client.dmabuf.is_none() {
        return Err("the compositor advertised no zwp_linux_dmabuf_v1".into());
    }
    client.make_toplevel()?;
    queue
        .roundtrip(&mut client)
        .map_err(|e| format!("toplevel roundtrip: {e}"))?;

    while client.running && !stop.load(Ordering::Relaxed) {
        queue.flush().map_err(|e| format!("flush: {e}"))?;
        queue
            .dispatch_pending(&mut client)
            .map_err(|e| format!("dispatch: {e}"))?;
        let Some(guard) = conn.prepare_read() else {
            continue;
        };
        // Poll rather than block: a blocking read ends only when the
        // compositor says something, and a test that stops ticking is a test
        // that has arranged for the compositor to say nothing.
        let mut pfd = libc::pollfd {
            fd: conn.as_fd().as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one initialised pollfd, and a timeout in milliseconds.
        let ready = unsafe { libc::poll(&mut pfd, 1, 20) };
        if ready > 0 {
            match guard.read() {
                Ok(_) => {}
                Err(wayland_client::backend::WaylandError::Io(e))
                    if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(format!("read: {e}")),
            }
        }
    }
    Ok(())
}

impl Client {
    fn make_toplevel(&mut self) -> Result<(), String> {
        let compositor = self.compositor.clone().ok_or("no wl_compositor")?;
        let wm_base = self.wm_base.clone().ok_or("no xdg_wm_base")?;
        let surface = compositor.create_surface(&self.qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &self.qh, ());
        let toplevel = xdg_surface.get_toplevel(&self.qh, ());
        toplevel.set_title("glasspane-fixture".into());
        toplevel.set_app_id("glasspane.fixture".into());
        surface.commit();
        self.surface = Some(surface);
        self.xdg_surface = Some(xdg_surface);
        self.toplevel = Some(toplevel);
        Ok(())
    }

    /// Paint and commit one frame, with a frame callback requested so the
    /// compositor's clock can drive the next one.
    fn paint(&mut self) {
        let (w, h) = self.size;
        if w == 0 || h == 0 {
            return;
        }
        let ok = match self.transport {
            Transport::Shm => self.paint_shm(w, h),
            Transport::Dmabuf => self.paint_dmabuf(w, h, Fourcc::Argb8888),
            Transport::DmabufAbgr => self.paint_dmabuf(w, h, Fourcc::Abgr8888),
        };
        if let Err(why) = ok {
            let _ = self.tx.send(Report::Failed(why));
            self.running = false;
            return;
        }
        let surface = self.surface.clone().expect("surface");
        surface.attach(self.buffer.as_ref(), 0, 0);
        surface.damage_buffer(0, 0, w as i32, h as i32);
        surface.frame(&self.qh, ());
        surface.commit();
        self.commits += 1;
        let _ = self.tx.send(Report::Committed {
            n: self.commits,
            w,
            h,
        });
    }

    fn paint_shm(&mut self, w: u32, h: u32) -> Result<(), String> {
        let stride = w * 4;
        let len = (stride * h) as usize;
        let fits =
            self.buffer.is_some() && self.buffer_size == (w, h) && self.pool.as_ref().is_some_and(|p| p.len >= len);
        if !fits {
            self.buffer = None;
            self.pool = None;
            let fd = memfd(len)?;
            // SAFETY: `fd` is a memfd of exactly `len` bytes, just created.
            let map = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd.as_raw_fd(),
                    0,
                )
            };
            if map == libc::MAP_FAILED {
                return Err(format!("mmap: {}", std::io::Error::last_os_error()));
            }
            let shm = self.shm.clone().ok_or("no wl_shm")?;
            let pool = shm.create_pool(fd.as_fd(), len as i32, &self.qh, ());
            let buffer = pool.create_buffer(
                0,
                w as i32,
                h as i32,
                stride as i32,
                wl_shm::Format::Argb8888,
                &self.qh,
                (),
            );
            self.pool = Some(ShmPool {
                pool,
                map: map.cast(),
                len,
            });
            self.buffer = Some(buffer);
            self.buffer_size = (w, h);
        }
        let pool = self.pool.as_ref().expect("pool");
        // SAFETY: the mapping is `len` bytes and is written row-major inside
        // it; the compositor only reads it between our commits.
        let bytes = unsafe { std::slice::from_raw_parts_mut(pool.map, len) };
        bytes.copy_from_slice(&expected(w, h));
        Ok(())
    }

    fn paint_dmabuf(&mut self, w: u32, h: u32, fourcc: Fourcc) -> Result<(), String> {
        if self.gbm.is_some() && self.buffer.is_some() && self.buffer_size == (w, h) {
            // A dma-buf's pixels are the client's own memory: re-committing
            // the same buffer is the whole point of the import-once law, and
            // the content is already what it will be.
            return Ok(());
        }
        let allocator = wgpu_dmabuf::testkit::GbmAllocator::open_default()
            .ok_or("no DRM render node to allocate a dma-buf on")?;
        let bo = allocator
            .allocate_linear(w, h, fourcc, |x, y| {
                let [b, g, r, a] = pattern(x, y);
                match fourcc {
                    Fourcc::Abgr8888 => [r, g, b, a],
                    _ => [b, g, r, a],
                }
            })
            .map_err(|e| format!("gbm allocate: {e}"))?;
        let plane = bo.dmabuf.sole_plane().map_err(|e| format!("plane: {e}"))?;
        let dmabuf = self.dmabuf.clone().ok_or("no zwp_linux_dmabuf_v1")?;
        let params = dmabuf.create_params(&self.qh, ());
        let modifier = bo.dmabuf.modifier;
        params.add(
            plane.as_fd(),
            0,
            plane.offset,
            plane.stride,
            (modifier >> 32) as u32,
            (modifier & 0xffff_ffff) as u32,
        );
        let buffer = params.create_immed(
            w as i32,
            h as i32,
            fourcc as u32,
            zwp_linux_buffer_params_v1::Flags::empty(),
            &self.qh,
            (),
        );
        params.destroy();
        self.buffer = Some(buffer);
        self.buffer_size = (w, h);
        self.gbm = Some(bo);
        Ok(())
    }
}

fn memfd(len: usize) -> Result<OwnedFd, String> {
    // SAFETY: a NUL-terminated name and no flags that change ownership.
    let fd = unsafe { libc::memfd_create(c"glasspane-fixture".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(format!("memfd_create: {}", std::io::Error::last_os_error()));
    }
    // SAFETY: `fd` was just returned by memfd_create and is owned here.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    // SAFETY: `fd` is a live memfd.
    if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } < 0 {
        return Err(format!("ftruncate: {}", std::io::Error::last_os_error()));
    }
    Ok(fd)
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

impl Dispatch<wl_registry::WlRegistry, ()> for Client {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        else {
            return;
        };
        match interface.as_str() {
            "wl_compositor" => {
                state.compositor = Some(registry.bind(name, version.min(4), qh, ()));
            }
            "wl_shm" => {
                state.shm = Some(registry.bind(name, 1, qh, ()));
            }
            "xdg_wm_base" => {
                state.wm_base = Some(registry.bind(name, version.min(3), qh, ()));
            }
            "wl_seat" => {
                let seat: wl_seat::WlSeat = registry.bind(name, version.min(5), qh, ());
                state.seat = Some(seat);
            }
            "zwp_linux_dmabuf_v1" => {
                // Version 3 on purpose: v4 replaces the format/modifier events
                // with per-surface feedback, which this fixture has no use for
                // — it allocates LINEAR and nothing else.
                state.dmabuf = Some(registry.bind(name, version.min(3), qh, ()));
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for Client {
    fn event(
        state: &mut Self,
        seat: &wl_seat::WlSeat,
        event: wl_seat::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities {
            capabilities: wayland_client::WEnum::Value(caps),
        } = event
        {
            if caps.contains(wl_seat::Capability::Pointer) {
                seat.get_pointer(qh, ());
            }
            if caps.contains(wl_seat::Capability::Keyboard) {
                seat.get_keyboard(qh, ());
            }
            let _ = state;
        }
    }
}

impl Dispatch<wl_pointer::WlPointer, ()> for Client {
    fn event(
        state: &mut Self,
        _: &wl_pointer::WlPointer,
        event: wl_pointer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let report = match event {
            wl_pointer::Event::Enter {
                surface_x,
                surface_y,
                ..
            } => Report::PointerEnter(surface_x, surface_y),
            wl_pointer::Event::Motion {
                surface_x,
                surface_y,
                ..
            } => Report::PointerMotion(surface_x, surface_y),
            wl_pointer::Event::Button { button, state: s, .. } => Report::PointerButton(
                button,
                matches!(s, wayland_client::WEnum::Value(wl_pointer::ButtonState::Pressed)),
            ),
            wl_pointer::Event::Axis { axis, value, .. } => match axis {
                wayland_client::WEnum::Value(wl_pointer::Axis::HorizontalScroll) => {
                    Report::PointerAxis(value, 0.0)
                }
                _ => Report::PointerAxis(0.0, value),
            },
            _ => return,
        };
        let _ = state.tx.send(report);
    }
}

impl Dispatch<wl_keyboard::WlKeyboard, ()> for Client {
    fn event(
        state: &mut Self,
        _: &wl_keyboard::WlKeyboard,
        event: wl_keyboard::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_keyboard::Event::Key { key, state: s, .. } = event {
            let _ = state.tx.send(Report::Key(
                key,
                matches!(
                    s,
                    wayland_client::WEnum::Value(wl_keyboard::KeyState::Pressed)
                ),
            ));
        }
    }
}

impl Dispatch<XdgWmBase, ()> for Client {
    fn event(
        _: &mut Self,
        wm_base: &XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            wm_base.pong(serial);
        }
    }
}

impl Dispatch<XdgSurface, ()> for Client {
    fn event(
        state: &mut Self,
        xdg_surface: &XdgSurface,
        event: xdg_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            xdg_surface.ack_configure(serial);
            state.configured = true;
            // Paint on the ack, which is the only moment the client knows the
            // size it was asked for is the size it is allowed to be.
            state.paint();
        }
    }
}

impl Dispatch<XdgToplevel, ()> for Client {
    fn event(
        state: &mut Self,
        _: &XdgToplevel,
        event: xdg_toplevel::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            xdg_toplevel::Event::Configure { width, height, .. } => {
                let _ = state.tx.send(Report::Configured(width, height));
                if width > 0 && height > 0 {
                    state.size = (width as u32, height as u32);
                } else if state.size == (0, 0) {
                    // "You choose" — and this client chooses a size whose row
                    // stride is a multiple of 256, so a readback needs no
                    // padding arithmetic to be wrong about.
                    state.size = (256, 256);
                }
            }
            xdg_toplevel::Event::Close => {
                let _ = state.tx.send(Report::CloseRequested);
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_callback::WlCallback, ()> for Client {
    fn event(
        state: &mut Self,
        _: &wl_callback::WlCallback,
        event: wl_callback::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_callback::Event::Done { .. } = event {
            // The frame callback is the only thing that makes this client
            // paint again. That is the contract under test.
            state.paint();
        }
    }
}

delegate_noop!(Client: ignore WlCompositor);
delegate_noop!(Client: ignore wl_surface::WlSurface);
delegate_noop!(Client: ignore wl_shm::WlShm);
delegate_noop!(Client: ignore wl_shm_pool::WlShmPool);
delegate_noop!(Client: ignore WlBuffer);
delegate_noop!(Client: ignore zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1);
delegate_noop!(Client: ignore zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1);

/// Unused, but it keeps `Proxy` in scope for the `bind` calls above.
#[allow(unused)]
fn _proxy_in_scope<T: Proxy>() {}

/// A borrowed fd, spelled once so the imports above are used in both builds.
#[allow(unused)]
fn _fd(f: &OwnedFd) -> BorrowedFd<'_> {
    f.as_fd()
}
