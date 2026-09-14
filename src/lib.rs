//! **A tenant Wayland compositor as a library.** It speaks Wayland to real
//! clients — terminals, browsers — and its entire output is one
//! [`wgpu::Texture`] per client plus an input-injection API.
//!
//! ```text
//!   socket ─ wl_compositor + xdg_shell + zwp_linux_dmabuf_v1 + wl_shm
//!          └─ commit ─ import-once ─ blit ─▶ Tenant::texture   (your wgpu::Device)
//!   tick() ───────────────────────────────▶ wl_callback.done ─ client paints again
//!   pointer_motion / keyboard_key ────────▶ wl_pointer / wl_keyboard
//! ```
//!
//! # What it is not
//!
//! No placement policy: glasspane never decides where a tenant is, how large
//! it is drawn, or what is in front of what. No scene graph, no renderer, no
//! window. No scanout: nothing here talks to KMS. The shell owns the
//! `wgpu::Device`, the camera, the frame rate and the screen; glasspane
//! borrows the first and is told about the third.
//!
//! There is one mode, and it is headless. A caller that wants a window makes
//! one.
//!
//! # Three rules that are load-bearing
//!
//! **The frame clock is yours to drive.** A Wayland client paints *only* when
//! handed a frame callback. A compositor that answers callbacks "on demand"
//! collapses to "when someone asks", and its tenants stall — measured
//! elsewhere at 0.3–11 s per frame, and stopping entirely when nothing was
//! watching. So glasspane never answers a callback by itself: [`Glasspane::tick`]
//! does, and you call it once per frame of *your* clock, whether or not
//! anything looks like it needs it.
//!
//! **Import once per `wl_buffer`.** A client that re-commits the same buffer —
//! every double-buffered client, forever — must produce exactly one dma-buf
//! import and then cache hits. See [`registry`].
//!
//! **The texture handle is stable while the size is.** Each tenant owns one
//! texture; commits are copied into it. It is replaced only when the client's
//! buffer size changes, which is announced as [`Event::Resized`] before any
//! [`Event::Committed`] against the new handle. Index tenants by [`TenantId`],
//! sample their textures, and never rebind per frame.
//!
//! # Coordinates
//!
//! Input is in tenant-local buffer pixels, origin top-left. See [`input`].
//!
//! # A whole session
//!
//! ```no_run
//! use std::time::{Duration, Instant};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
//!     backends: wgpu::Backends::VULKAN,
//!     ..Default::default()
//! });
//! let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
//! // Stock `request_device` cannot import an acquire fence; this can.
//! let (device, queue, _caps) =
//!     wgpu_dmabuf::device::open_device(&adapter, &Default::default())?;
//!
//! let mut pane = glasspane::Glasspane::new(
//!     device,
//!     queue,
//!     glasspane::Caps::new().with_adapter(&adapter).with_size(1280, 800),
//! )?;
//! let _child = pane.spawn(std::process::Command::new("foot"))?;
//!
//! let start = Instant::now();
//! while start.elapsed() < Duration::from_secs(2) {
//!     pane.dispatch(Some(Duration::from_millis(4)))?;
//!     pane.tick(Instant::now());
//!     for event in pane.events() {
//!         println!("{event:?}");
//!     }
//!     for tenant in pane.tenants() {
//!         if let Some(tex) = tenant.texture() {
//!             let _ = tex; // sample it in your scene
//!         }
//!     }
//! }
//! # Ok(()) }
//! ```

#![cfg(target_os = "linux")]
#![deny(missing_debug_implementations)]
#![warn(missing_docs)]
#![doc(html_root_url = "https://docs.rs/glasspane/0.1.0")]

pub mod acquire;
mod convert;
mod error;
pub mod event;
pub mod input;
pub mod registry;
mod state;
pub mod tenant;
pub mod texture;

pub use acquire::AcquireReport;
pub use error::{Error, Result};
pub use event::Event;
pub use input::KeyState;
pub use tenant::{Tenant, TenantId};

/// The smithay this crate is built against, re-exported so a caller naming
/// `ToplevelSurface` or `calloop::EventLoop` names *the* smithay rather than a
/// semver-compatible second copy.
pub use smithay;

use std::ffi::OsString;
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::{Duration, Instant};

use smithay::input::keyboard::XkbConfig;
use smithay::output::{Mode as OutputMode, Output, PhysicalProperties, Subpixel};
use smithay::reexports::calloop::{EventLoop, Interest, Mode, PostAction, generic::Generic};
use smithay::reexports::wayland_server::Display;
use smithay::utils::Transform;
use smithay::wayland::compositor::CompositorState;
use smithay::wayland::dmabuf::DmabufState;
use smithay::wayland::selection::data_device::DataDeviceState;
use smithay::wayland::shell::xdg::XdgShellState;
use smithay::wayland::shell::xdg::decoration::XdgDecorationState;
use smithay::wayland::shm::ShmState;
use smithay::wayland::socket::ListeningSocketSource;

use state::{ClientState, State};

/// What kind of compositor to be.
///
/// Everything here has a default that works; the one field worth setting
/// deliberately is [`Caps::with_adapter`], without which no dma-buf global is
/// advertised and every tenant falls back to `wl_shm`.
#[derive(Debug, Clone)]
pub struct Caps<'a> {
    /// The adapter behind the device, used to enumerate the DRM format
    /// modifiers this GPU will actually share.
    ///
    /// `None` means **no `zwp_linux_dmabuf_v1` global at all**, so clients use
    /// `wl_shm`. That is a real mode — it is what a software-rendered tenant
    /// does anyway — but it is a CPU copy per frame per tenant.
    pub adapter: Option<&'a wgpu::Adapter>,
    /// The socket to listen on, e.g. `"glasspane-0"`. `None` takes the next
    /// free `wayland-N`.
    ///
    /// Name it in a test. Two tests that both take "the next free" one on a
    /// machine with a live session are two tests that can hand a client the
    /// wrong compositor.
    pub socket_name: Option<String>,
    /// The size every new tenant is configured with, in pixels, before it has
    /// said anything about what it wants.
    pub default_size: (u32, u32),
    /// The seat's name, as `wl_seat.name` carries it.
    pub seat_name: String,
    /// The xkb layout, as in `XkbConfig`. `None` is xkb's own default.
    pub xkb_layout: Option<String>,
    /// Advertise dma-buf even if this device cannot import an acquire fence.
    ///
    /// Off by default, and the default is the honest one: without
    /// `VK_KHR_external_semaphore_fd` there is no way to order a client's GPU
    /// writes before this compositor's reads, and a compositor that samples
    /// client memory with nothing ordering it is not a degraded mode, it is a
    /// silent-wrong-output bug that passes its own tests on the driver it was
    /// written on. Open the device with [`wgpu_dmabuf::device::open_device`]
    /// and this never comes up.
    pub allow_unfenced_dmabuf: bool,
}

impl Default for Caps<'_> {
    fn default() -> Self {
        Self {
            adapter: None,
            socket_name: None,
            default_size: (1280, 800),
            seat_name: "glasspane".into(),
            xkb_layout: None,
            allow_unfenced_dmabuf: false,
        }
    }
}

impl<'a> Caps<'a> {
    /// The defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advertise the dma-buf formats this adapter can share.
    pub fn with_adapter(mut self, adapter: &'a wgpu::Adapter) -> Self {
        self.adapter = Some(adapter);
        self
    }

    /// Listen on a named socket rather than the next free one.
    pub fn with_socket_name(mut self, name: impl Into<String>) -> Self {
        self.socket_name = Some(name.into());
        self
    }

    /// The size new tenants are configured with.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.default_size = (width, height);
        self
    }

    /// The seat name.
    pub fn with_seat_name(mut self, name: impl Into<String>) -> Self {
        self.seat_name = name.into();
        self
    }

    /// The xkb layout, e.g. `"gb"`.
    pub fn with_xkb_layout(mut self, layout: impl Into<String>) -> Self {
        self.xkb_layout = Some(layout.into());
        self
    }
}

/// A running compositor: a socket, an event loop, and a texture per tenant.
///
/// Owns its calloop [`EventLoop`]; [`Glasspane::dispatch`] runs one turn of it.
/// It is not `Send`: the Wayland display, the seat and the wgpu queue are all
/// driven from the thread that built them.
pub struct Glasspane {
    event_loop: EventLoop<'static, State>,
    state: State,
    socket_name: OsString,
}

impl std::fmt::Debug for Glasspane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Glasspane")
            .field("socket_name", &self.socket_name)
            .field("tenants", &self.state.tenants.len())
            .finish_non_exhaustive()
    }
}

impl Glasspane {
    /// Open a socket and start listening.
    ///
    /// `device` and `queue` are the shell's: they are cloned (both are handles
    /// to the same underlying device) and every texture this crate produces
    /// belongs to them, so the shell can sample them with no sharing, no
    /// second device and no cross-device fence.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, caps: Caps<'_>) -> Result<Self> {
        let event_loop: EventLoop<'static, State> =
            EventLoop::try_new().map_err(|e| Error::Socket(format!("calloop: {e}")))?;
        let display: Display<State> =
            Display::new().map_err(|e| Error::Socket(format!("wl_display: {e}")))?;
        let dh = display.handle();

        let compositor_state = CompositorState::new::<State>(&dh);
        let xdg_shell_state = XdgShellState::new::<State>(&dh);
        let xdg_decoration_state = XdgDecorationState::new::<State>(&dh);
        let shm_state = ShmState::new::<State>(&dh, vec![]);
        let data_device_state = DataDeviceState::new::<State>(&dh);
        let mut dmabuf_state = DmabufState::new();

        // The advertisement is derived from the device, never hardcoded: a
        // modifier list that is a guess is a list clients will allocate
        // against and this compositor will then refuse.
        let acquire = match acquire::AcquireGate::new(&device, &queue) {
            Ok(gate) => Some(gate),
            Err(e) => {
                if !caps.allow_unfenced_dmabuf && caps.adapter.is_some() {
                    return Err(Error::Device(format!(
                        "this device cannot import an acquire fence ({e}), so dma-buf tenants \
                         could not be ordered against this compositor's reads. Open it with \
                         wgpu_dmabuf::device::open_device, or set Caps::allow_unfenced_dmabuf \
                         to accept unordered sampling."
                    )));
                }
                None
            }
        };
        let fenced = acquire.is_some();
        let dmabuf_global = match caps.adapter {
            Some(adapter) if fenced || caps.allow_unfenced_dmabuf => {
                let formats = shareable_formats(adapter)?;
                if formats.is_empty() {
                    None
                } else {
                    Some(dmabuf_state.create_global::<State>(&dh, formats))
                }
            }
            _ => None,
        };

        // One output, sized to the default tenant size. It exists because a
        // client with no `wl_output` at all cannot work out its scale and some
        // toolkits refuse to map; its geometry is not placement policy,
        // because nothing is ever placed on it.
        let output = Output::new(
            "glasspane".to_string(),
            PhysicalProperties {
                size: (0, 0).into(),
                subpixel: Subpixel::Unknown,
                make: "glasspane".into(),
                model: "headless".into(),
            },
        );
        let mode = OutputMode {
            size: (caps.default_size.0 as i32, caps.default_size.1 as i32).into(),
            refresh: 60_000,
        };
        output.change_current_state(Some(mode), Some(Transform::Normal), None, Some((0, 0).into()));
        output.set_preferred(mode);
        output.create_global::<State>(&dh);

        let mut seat_state = smithay::input::SeatState::<State>::new();
        let xkb = match caps.xkb_layout.as_deref() {
            Some(layout) => XkbConfig {
                layout,
                ..Default::default()
            },
            None => XkbConfig::default(),
        };
        let (seat, keyboard, pointer) =
            State::make_seat(&mut seat_state, &dh, &caps.seat_name, xkb)?;

        let socket = match &caps.socket_name {
            Some(name) => ListeningSocketSource::with_name(name),
            None => ListeningSocketSource::new_auto(),
        }
        .map_err(|e| Error::Socket(format!("bind: {e}")))?;
        let socket_name = socket.socket_name().to_os_string();

        let handle = event_loop.handle();
        handle
            .insert_source(socket, move |stream, _, state: &mut State| {
                // The pid is read here, once, from the socket's peer
                // credentials: it is the only thing that ties a tenant back to
                // the Child the caller holds.
                let pid = peer_pid(&stream);
                let data = Arc::new(ClientState {
                    pid,
                    ..Default::default()
                });
                if let Err(e) = state.dh.insert_client(stream, data) {
                    state.reject(None, format!("insert_client: {e}"));
                }
            })
            .map_err(|e| Error::Socket(format!("insert socket source: {e}")))?;

        handle
            .insert_source(
                Generic::new(display, Interest::READ, Mode::Level),
                |_, display, state: &mut State| {
                    // SAFETY: the display is never dropped out of the source.
                    unsafe { display.get_mut().dispatch_clients(state)? };
                    // Every answer to a client request — a registry global, a
                    // `wl_display.sync` reply, an `xdg_surface.configure` — is
                    // a buffered write, and the next thing this loop does is
                    // block in poll. A client whose entire startup is
                    // roundtrips (Chromium is one) reaches its timeout with
                    // zero toplevels without this line.
                    state.flush();
                    Ok(PostAction::Continue)
                },
            )
            .map_err(|e| Error::Socket(format!("insert display source: {e}")))?;

        let state = State {
            start: Instant::now(),
            dh,
            device,
            queue,
            compositor_state,
            xdg_shell_state,
            xdg_decoration_state,
            shm_state,
            dmabuf_state,
            dmabuf_global,
            data_device_state,
            seat_state,
            seat,
            keyboard,
            pointer,
            output,
            registry: registry::Registry::new(),
            acquire,
            tenants: Vec::new(),
            next_tenant: 0,
            events: Default::default(),
            pending_callbacks: Vec::new(),
            default_size: caps.default_size,
            keyboard_focus: None,
            pointer_focus: None,
            ticks: 0,
            callbacks_sent: 0,
        };

        Ok(Self {
            event_loop,
            state,
            socket_name,
        })
    }

    /// The socket this compositor is listening on, i.e. the value of
    /// `WAYLAND_DISPLAY` a client needs.
    pub fn socket_name(&self) -> &std::ffi::OsStr {
        &self.socket_name
    }

    /// Whether a `zwp_linux_dmabuf_v1` global was advertised. `false` means
    /// every tenant will be an `wl_shm` tenant.
    pub fn advertises_dmabuf(&self) -> bool {
        self.state.dmabuf_global.is_some()
    }

    /// Spawn a client on this compositor's socket.
    ///
    /// `WAYLAND_DISPLAY` is set on the command, and `DISPLAY` and
    /// `WAYLAND_SOCKET` are removed from it. Both removals are load-bearing: a
    /// toolkit that can see an X display will often prefer it, and
    /// `WAYLAND_SOCKET` names an already-connected fd that takes precedence
    /// over `WAYLAND_DISPLAY` — so a process that inherited one would hand the
    /// child somebody else's compositor while looking correctly configured.
    ///
    /// The returned [`Child`] is the caller's to wait on and to kill; glasspane
    /// does not reap it. The tenant itself arrives later, as
    /// [`Event::Mapped`], carrying the pid so the two can be matched — a
    /// process may map several toplevels, or none, and may do so seconds after
    /// it started, so there is no [`TenantId`] to return here and this crate
    /// does not invent one.
    pub fn spawn(&mut self, mut command: Command) -> Result<Child> {
        command
            .env("WAYLAND_DISPLAY", &self.socket_name)
            .env_remove("DISPLAY")
            .env_remove("WAYLAND_SOCKET");
        command.spawn().map_err(Error::Spawn)
    }

    /// Run one turn of the event loop: accept connections, read client
    /// requests, turn commits into texels.
    ///
    /// `timeout` is how long to block if nothing is ready; `None` blocks until
    /// something is. A shell with its own frame budget passes a slice of it.
    pub fn dispatch(&mut self, timeout: Option<Duration>) -> Result<()> {
        self.event_loop
            .dispatch(timeout, &mut self.state)
            .map_err(|e| Error::Dispatch(e.to_string()))?;
        self.state.flush();
        Ok(())
    }

    /// **Drive the frame clock**: answer every frame callback harvested since
    /// the last tick, and say how many were answered.
    ///
    /// Call this once per frame of your own clock, unconditionally. Not when a
    /// tenant looks like it wants something, not when a texture changed — a
    /// client paints only when handed a callback, so "when it needs it" is a
    /// circular condition that resolves to "never".
    ///
    /// Zero is not an error: it means no tenant was waiting. A run in which it
    /// is *always* zero is a run in which nothing is painting.
    pub fn tick(&mut self, now: Instant) -> usize {
        let ms = now.saturating_duration_since(self.state.start).as_millis() as u32;
        self.state.release_frame_callbacks(ms)
    }

    /// Every live tenant, in no particular order.
    ///
    /// Not insertion order, and not stable across calls. Identify tenants by
    /// [`TenantId`] — a "the last one is the newest" reading produces real
    /// pixels under the wrong label, which is worse than an error.
    pub fn tenants(&self) -> impl ExactSizeIterator<Item = &Tenant> {
        self.state.tenants.iter()
    }

    /// One tenant by id.
    pub fn tenant(&self, id: TenantId) -> Option<&Tenant> {
        self.state.tenants.iter().find(|t| t.id == id)
    }

    /// Drain the event queue.
    ///
    /// Events are facts about tenants, in the order they happened. The queue
    /// is bounded; a caller that never drains loses the *oldest* events.
    pub fn events(&mut self) -> impl ExactSizeIterator<Item = Event> + '_ {
        std::mem::take(&mut self.state.events).into_iter()
    }

    /// Ask a tenant to be a different size.
    ///
    /// A resize is a negotiation — configure, ack, commit — so this is a
    /// request and nothing more. [`Tenant::size`] does not change here; it
    /// changes when the client paints at the new size, which is announced as
    /// [`Event::Resized`]. A starved client (one that is not being ticked)
    /// keeps its old rect indefinitely, and a caller that assumes otherwise
    /// gets N identical textures filed under N different sizes.
    pub fn configure(&mut self, id: TenantId, size: (u32, u32)) -> Result<()> {
        let index = self.state.index_of(id).ok_or(Error::NoSuchTenant(id))?;
        let toplevel = self.state.tenants[index].toplevel.clone();
        toplevel.with_pending_state(|state| {
            state.size = Some((size.0 as i32, size.1 as i32).into());
        });
        toplevel.send_pending_configure();
        self.state.tenants[index].configured = Some(size);
        self.state.flush();
        Ok(())
    }

    /// Ask a tenant to close.
    ///
    /// `xdg_toplevel.close` is a request the client may refuse or delay (a
    /// terminal with a running job will). The tenant is gone when
    /// [`Event::Unmapped`] says so, not when this returns.
    pub fn close(&mut self, id: TenantId) -> Result<()> {
        let index = self.state.index_of(id).ok_or(Error::NoSuchTenant(id))?;
        self.state.tenants[index].toplevel.send_close();
        self.state.flush();
        Ok(())
    }

    /// The acquire ledger: how many client fences were exported, how many
    /// waits a submit actually carried, and how many of those could not
    /// possibly have blocked.
    ///
    /// `None` if this device cannot fence (see [`Caps::allow_unfenced_dmabuf`]).
    /// A harness that reports `waits` without `already_signalled` is reporting
    /// a gate that passes for a reason unrelated to what it measures.
    pub fn acquire_report(&self) -> Option<AcquireReport> {
        self.state.acquire.as_ref().map(|g| g.report())
    }

    /// How many dma-buf imports have happened, and how many commits hit the
    /// cache instead.
    ///
    /// The import-once law expressed as two numbers: a client re-committing
    /// one buffer must move the second and not the first.
    pub fn import_stats(&self) -> (u64, u64) {
        (
            self.state.registry.import_count(),
            self.state.registry.cache_hit_count(),
        )
    }

    /// Frame callbacks answered over this run, and ticks taken.
    pub fn frame_stats(&self) -> (u64, u64) {
        (self.state.callbacks_sent, self.state.ticks)
    }

    /// The `wgpu::Device` every tenant texture belongs to.
    pub fn device(&self) -> &wgpu::Device {
        &self.state.device
    }

    /// The queue the acquire waits are staged on.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.state.queue
    }
}

impl Drop for Glasspane {
    /// The acquire pool holds raw `VkSemaphore`s on a device this struct also
    /// holds a handle to. Releasing them in an explicit `Drop` rather than
    /// leaving it to field order is the difference between a clean exit and
    /// destroying semaphores on a dead device.
    fn drop(&mut self) {
        if let Some(gate) = self.state.acquire.as_mut() {
            gate.shutdown();
        }
    }
}

/// The client's pid, from `SO_PEERCRED`.
///
/// `UnixStream::peer_cred` is still unstable, and this is the one fact that
/// lets a caller match an [`Event::Mapped`] against the [`Child`] it holds, so
/// the two-line `getsockopt` is worth having rather than doing without.
fn peer_pid(stream: &std::os::unix::net::UnixStream) -> Option<u32> {
    use std::os::fd::AsRawFd;
    let mut cred = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: `cred` and `len` are live, correctly sized, and `stream` is an
    // open socket for the duration of the call.
    let rc = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&raw mut cred).cast(),
            &raw mut len,
        )
    };
    (rc == 0 && cred.pid > 0).then_some(cred.pid as u32)
}

/// The DRM formats and modifiers this adapter can both import from and export
/// to a dma-buf, in the shape `DmabufState::create_global` wants.
fn shareable_formats(
    adapter: &wgpu::Adapter,
) -> Result<Vec<smithay::backend::allocator::Format>> {
    use smithay::backend::allocator::Format;
    let mut out = Vec::new();
    for fourcc in texture::ACCEPTED_FOURCC {
        for m in wgpu_dmabuf::shareable(adapter, fourcc, texture::IMPORT_USAGE)? {
            out.push(Format {
                code: fourcc,
                modifier: m.value.into(),
            });
        }
    }
    Ok(out)
}
