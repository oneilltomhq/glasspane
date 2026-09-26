//! **The shell side of the daemon**: [`Glasspane`] with the in-process
//! facade's method names and [`Event`] enum, over a unix socket to a
//! `glasspane-daemon` instead of over a Wayland socket to tenants.
//!
//! ```rust,no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let adapter: wgpu::Adapter = todo!();
//! let (device, queue, _) = wgpu_dmabuf::device::open_device(&adapter, &Default::default())?;
//! let mut pane = glasspane::remote::Glasspane::new(device, queue)?; // default socket
//! loop {
//!     pane.dispatch(Some(std::time::Duration::from_millis(4)))?;
//!     pane.tick(std::time::Instant::now());
//!     for event in pane.events() { /* Mapped, Committed, Resized, … */ }
//!     for tenant in pane.tenants() {
//!         if let Some(texture) = tenant.texture() { /* sample it */ }
//!     }
//! }
//! # }
//! ```
//!
//! The import, the acquire wait and the blit happen here, on the caller's
//! device, through the same [`crate::registry`], [`crate::acquire`] and
//! [`crate::texture`] the in-process path uses: the daemon hands over each
//! commit's plane fd, its metadata, a stable buffer id and (normally) the
//! acquire `sync_file` it exported at commit time. A buffer id is imported
//! once and re-blitted on later commits; the daemon's `BufferGone` drops the
//! import. The two texture laws hold unchanged: [`Tenant::texture`] is stable
//! while the size is, and a [`Event::Resized`] precedes any
//! [`Event::Committed`] against a new handle.
//!
//! # Differences from [`crate::Glasspane`]
//!
//! * **Constructor**: [`Glasspane::new`]`(device, queue)` connects to the
//!   default socket; [`Glasspane::connect`]`(device, queue, path)` to another.
//!   There is no `Caps`: the daemon owns the Wayland side. The device must be
//!   able to import an acquire fence (open it with
//!   [`wgpu_dmabuf::device::open_device`]), or connecting fails.
//! * **[`Glasspane::tick`]** tells the daemon the shell presented, and the
//!   daemon answers frame callbacks then. It returns `0`: the count is the
//!   daemon's, not known here.
//! * **[`Glasspane::spawn`]** asks the daemon to run the command on its
//!   socket and returns the **pid**, not a [`std::process::Child`]: the
//!   daemon owns and reaps the process, which is the point — it outlives this
//!   shell.
//! * **On connect**, every existing tenant arrives as a fresh `Mapped` (with
//!   its current [`Tenant::size`] already set), its title, and a `Committed`
//!   of its last buffer. A shell cannot tell a replay from a new tenant, and
//!   does not need to.
//! * **[`Tenant`]** is this module's own type, with the same accessors minus
//!   `surface()` and `toplevel()` (the protocol objects are in another
//!   process), plus [`Tenant::buffer`].
//! * **Not in the spike**: popups ([`Glasspane::popups`] is always empty),
//!   pointer and keyboard (the `Result` methods return an error, the others
//!   do nothing), and `wl_shm` tenants (the daemon rejects them).

use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::os::fd::{AsFd, OwnedFd};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::acquire::{AcquireGate, AcquireReport};
use crate::input::{KeyState, Target};
use crate::popup::{Popup, PopupId};
use crate::registry::{BufferKey, Registry};
use crate::tenant::TenantId;
use crate::texture::{self, Skin};
use crate::wire::{self, Conn, Fence, ToDaemon, ToShell};
use crate::{Error, Event, Result};

/// One tenant, as a remote shell sees it.
#[derive(Debug)]
pub struct Tenant {
    id: TenantId,
    pid: Option<u32>,
    title: Option<String>,
    app_id: Option<String>,
    configured: Option<(u32, u32)>,
    skin: Skin,
    buffer: Option<u64>,
}

impl Tenant {
    /// This tenant's identity. Stable across shell restarts: it is the
    /// daemon's.
    pub fn id(&self) -> TenantId {
        self.id
    }

    /// The client process.
    pub fn pid(&self) -> Option<u32> {
        self.pid
    }

    /// `xdg_toplevel.set_title`, if the client has sent one.
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// `xdg_toplevel.set_app_id`, if the client has sent one.
    pub fn app_id(&self) -> Option<&str> {
        self.app_id.as_deref()
    }

    /// The size of the last committed buffer — set from `Mapped` before the
    /// first commit, so it is right on the `Mapped` event, as in-process.
    pub fn size(&self) -> (u32, u32) {
        self.skin.size
    }

    /// The size this tenant was last told to be.
    pub fn configured(&self) -> Option<(u32, u32)> {
        self.configured
    }

    /// The texture holding this tenant's pixels, or `None` before its first
    /// commit. Stable while [`Tenant::size`] is.
    pub fn texture(&self) -> Option<&wgpu::Texture> {
        self.skin.texture.as_deref()
    }

    /// The texture as a shared handle.
    pub fn texture_arc(&self) -> Option<&Arc<wgpu::Texture>> {
        self.skin.texture.as_ref()
    }

    /// The daemon's commit counter as of the last commit this shell blitted.
    pub fn generation(&self) -> u64 {
        self.skin.generation
    }

    /// Whether the last commit's acquire fence was staged as a wait.
    pub fn acquire_staged(&self) -> bool {
        self.skin.acquire_staged
    }

    /// The daemon's id for the `wl_buffer` last blitted: equal across two
    /// shells means the same client buffer, not a repaint.
    pub fn buffer(&self) -> Option<u64> {
        self.buffer
    }
}

/// A connection to a `glasspane-daemon`, and a texture per tenant.
pub struct Glasspane {
    conn: Conn,
    device: wgpu::Device,
    queue: wgpu::Queue,
    registry: Registry,
    acquire: Option<AcquireGate>,
    tenants: Vec<Tenant>,
    events: VecDeque<Event>,
    wayland_display: OsString,
    dmabuf: bool,
    spawn_reply: Option<std::result::Result<u32, String>>,
}

impl std::fmt::Debug for Glasspane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("remote::Glasspane")
            .field("wayland_display", &self.wayland_display)
            .field("tenants", &self.tenants.len())
            .finish_non_exhaustive()
    }
}

fn remote(e: impl std::fmt::Display) -> Error {
    Error::Remote(e.to_string())
}

impl Glasspane {
    /// Connect to the daemon at `$XDG_RUNTIME_DIR/plurality/glasspane.sock`.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self> {
        let path = default_path();
        Self::connect(device, queue, &path)
    }

    /// Connect to the daemon at `path`, and wait for its hello.
    ///
    /// The replay of existing tenants follows the hello and arrives through
    /// [`Glasspane::dispatch`] like anything else.
    pub fn connect(device: wgpu::Device, queue: wgpu::Queue, path: &Path) -> Result<Self> {
        let acquire = AcquireGate::new(&device, &queue).map_err(|e| {
            Error::Device(format!(
                "this device cannot import an acquire fence ({e}); open it with \
                 wgpu_dmabuf::device::open_device"
            ))
        })?;
        let conn = Conn::connect(path)
            .map_err(|e| Error::Remote(format!("connect {}: {e}", path.display())))?;
        if !conn
            .wait_readable(Some(Duration::from_secs(5)))
            .map_err(remote)?
        {
            return Err(Error::Remote("the daemon sent no hello within 5 s".into()));
        }
        let (bytes, _) = conn
            .recv()
            .map_err(remote)?
            .ok_or_else(|| Error::Remote("no hello".into()))?;
        let (wayland_display, dmabuf) = match ToShell::decode(&bytes).map_err(remote)? {
            ToShell::Hello {
                version,
                wayland_display,
                dmabuf,
            } if version == wire::VERSION => (wayland_display, dmabuf),
            ToShell::Hello { version, .. } => {
                return Err(Error::Remote(format!(
                    "daemon speaks wire version {version}, this shell {}",
                    wire::VERSION
                )));
            }
            other => return Err(Error::Remote(format!("expected hello, got {other:?}"))),
        };
        Ok(Self {
            conn,
            device,
            queue,
            registry: Registry::new(),
            acquire: Some(acquire),
            tenants: Vec::new(),
            events: VecDeque::new(),
            wayland_display: wayland_display.into(),
            dmabuf,
            spawn_reply: None,
        })
    }

    /// The daemon's Wayland socket name.
    pub fn socket_name(&self) -> &OsStr {
        &self.wayland_display
    }

    /// Whether the daemon advertises dma-buf.
    pub fn advertises_dmabuf(&self) -> bool {
        self.dmabuf
    }

    /// Ask the daemon to run `command` on its Wayland socket. Returns the pid;
    /// the daemon owns the process.
    ///
    /// The program, arguments and environment changes cross; the working
    /// directory and stdio do not.
    pub fn spawn(&mut self, command: Command) -> Result<u32> {
        let msg = ToDaemon::Spawn {
            program: command.get_program().to_string_lossy().into_owned(),
            args: command
                .get_args()
                .map(|a| a.to_string_lossy().into_owned())
                .collect(),
            env: command
                .get_envs()
                .map(|(k, v)| {
                    (
                        k.to_string_lossy().into_owned(),
                        v.map(|v| v.to_string_lossy().into_owned()),
                    )
                })
                .collect(),
        };
        self.send(&msg)?;
        self.spawn_reply = None;
        let deadline = Instant::now() + Duration::from_secs(10);
        while self.spawn_reply.is_none() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(Error::Remote("the daemon did not answer the spawn".into()));
            }
            self.dispatch(Some(left))?;
        }
        match self.spawn_reply.take().expect("loop condition") {
            Ok(pid) => Ok(pid),
            Err(why) => Err(Error::Spawn(std::io::Error::other(why))),
        }
    }

    /// Wait up to `timeout` for the daemon, then handle everything it sent:
    /// imports, acquire waits and blits happen here.
    ///
    /// An error means the daemon went away (or spoke nonsense). The textures
    /// already held stay valid; reconnecting is a new [`Glasspane`].
    pub fn dispatch(&mut self, timeout: Option<Duration>) -> Result<()> {
        self.conn.wait_readable(timeout).map_err(remote)?;
        while let Some((bytes, fds)) = self.conn.recv().map_err(remote)? {
            let msg = ToShell::decode(&bytes).map_err(remote)?;
            self.handle(msg, fds);
        }
        Ok(())
    }

    /// Tell the daemon this shell presented a frame; it answers the tenants'
    /// frame callbacks on receipt. Call it once per frame, unconditionally,
    /// exactly as in-process.
    ///
    /// Returns `0`: how many callbacks were answered is known only to the
    /// daemon.
    pub fn tick(&mut self, _now: Instant) -> usize {
        let _ = self.send(&ToDaemon::Presented);
        0
    }

    /// Every live tenant, in no particular order.
    pub fn tenants(&self) -> impl ExactSizeIterator<Item = &Tenant> {
        self.tenants.iter()
    }

    /// One tenant by id.
    pub fn tenant(&self, id: TenantId) -> Option<&Tenant> {
        self.tenants.iter().find(|t| t.id == id)
    }

    /// Always empty: popups do not cross the socket in the spike.
    pub fn popups(&self) -> impl ExactSizeIterator<Item = &Popup> {
        [].iter()
    }

    /// Always `None`: popups do not cross the socket in the spike.
    pub fn popup(&self, _id: PopupId) -> Option<&Popup> {
        None
    }

    /// Always `None`.
    pub fn grabbing_popup(&self) -> Option<PopupId> {
        None
    }

    /// Always [`Error::NoSuchPopup`]: popups do not cross the socket.
    pub fn dismiss_popup(&mut self, id: PopupId) -> Result<()> {
        Err(Error::NoSuchPopup(id))
    }

    /// Does nothing: popups do not cross the socket.
    pub fn dismiss_popups_of(&mut self, _id: TenantId) {}

    /// Drain the event queue.
    pub fn events(&mut self) -> impl ExactSizeIterator<Item = Event> + '_ {
        std::mem::take(&mut self.events).into_iter()
    }

    /// Ask a tenant to be a different size. As in-process, a request only:
    /// [`Tenant::size`] changes when the client paints at it.
    pub fn configure(&mut self, id: TenantId, size: (u32, u32)) -> Result<()> {
        let tenant = self
            .tenants
            .iter_mut()
            .find(|t| t.id == id)
            .ok_or(Error::NoSuchTenant(id))?;
        tenant.configured = Some(size);
        self.send(&ToDaemon::Configure {
            id: id.0,
            width: size.0,
            height: size.1,
        })
    }

    /// Ask a tenant to close.
    pub fn close(&mut self, id: TenantId) -> Result<()> {
        if self.tenant(id).is_none() {
            return Err(Error::NoSuchTenant(id));
        }
        self.send(&ToDaemon::Close { id: id.0 })
    }

    /// The acquire ledger, as in-process.
    pub fn acquire_report(&self) -> Option<AcquireReport> {
        self.acquire.as_ref().map(|g| g.report())
    }

    /// Imports and cache hits, as in-process.
    pub fn import_stats(&self) -> (u64, u64) {
        (
            self.registry.import_count(),
            self.registry.cache_hit_count(),
        )
    }

    /// The device every tenant texture belongs to.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue the acquire waits are staged on.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    // --- input: not in the spike --------------------------------------------

    fn no_input() -> Error {
        Error::Remote("input does not cross the daemon socket in the spike".into())
    }

    /// Not in the spike: always an error.
    pub fn pointer_enter(&mut self, _id: TenantId, _x: f64, _y: f64) -> Result<()> {
        Err(Self::no_input())
    }
    /// Not in the spike: always an error.
    pub fn pointer_enter_at(&mut self, _t: Target, _x: f64, _y: f64) -> Result<()> {
        Err(Self::no_input())
    }
    /// Not in the spike: always an error.
    pub fn pointer_motion(&mut self, _id: TenantId, _x: f64, _y: f64) -> Result<()> {
        Err(Self::no_input())
    }
    /// Not in the spike: always an error.
    pub fn pointer_motion_at(&mut self, _t: Target, _x: f64, _y: f64) -> Result<()> {
        Err(Self::no_input())
    }
    /// Not in the spike: does nothing.
    pub fn pointer_leave(&mut self) {}
    /// Not in the spike: does nothing.
    pub fn pointer_button(&mut self, _code: u32, _state: KeyState) {}
    /// Not in the spike: does nothing.
    pub fn pointer_axis(&mut self, _h: f64, _v: f64, _v120: Option<(i32, i32)>) {}
    /// Not in the spike: always an error.
    pub fn set_keyboard_focus(&mut self, _id: Option<TenantId>) -> Result<()> {
        Err(Self::no_input())
    }
    /// Not in the spike: always an error.
    pub fn set_keyboard_focus_at(&mut self, _t: Option<Target>) -> Result<()> {
        Err(Self::no_input())
    }
    /// Not in the spike: always `None`.
    pub fn keyboard_focus(&self) -> Option<TenantId> {
        None
    }
    /// Not in the spike: always an error.
    pub fn keyboard_key(&mut self, _to: Option<TenantId>, _evdev: u32, _s: KeyState) -> Result<()> {
        Err(Self::no_input())
    }
    /// Not in the spike: always `None`.
    pub fn keyboard_focus_at(&self) -> Option<Target> {
        None
    }
    /// Not in the spike: always `None`.
    pub fn pointer_focus_at(&self) -> Option<Target> {
        None
    }
    /// Not in the spike: always an error.
    pub fn keyboard_key_at(&mut self, _to: Option<Target>, _evdev: u32, _s: KeyState) -> Result<()> {
        Err(Self::no_input())
    }

    // --- internals ------------------------------------------------------------

    fn send(&self, msg: &ToDaemon) -> Result<()> {
        self.conn.send(&msg.encode(), &[]).map_err(remote)
    }

    fn emit(&mut self, event: Event) {
        const CAP: usize = 4096;
        if self.events.len() >= CAP {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    fn index_of(&self, id: TenantId) -> Option<usize> {
        self.tenants.iter().position(|t| t.id == id)
    }

    fn handle(&mut self, msg: ToShell, fds: Vec<OwnedFd>) {
        match msg {
            ToShell::Hello { .. } => {}
            ToShell::Mapped {
                id,
                pid,
                size,
                configured,
            } => {
                let id = TenantId(id);
                if self.index_of(id).is_some() {
                    return;
                }
                self.tenants.push(Tenant {
                    id,
                    pid,
                    title: None,
                    app_id: None,
                    configured,
                    skin: Skin {
                        size,
                        ..Default::default()
                    },
                    buffer: None,
                });
                self.emit(Event::Mapped { id, pid });
            }
            ToShell::Unmapped { id } => {
                let id = TenantId(id);
                if let Some(i) = self.index_of(id) {
                    self.tenants.remove(i);
                    self.emit(Event::Unmapped { id });
                }
            }
            ToShell::Title { id, title } => {
                let id = TenantId(id);
                if let Some(i) = self.index_of(id) {
                    if self.tenants[i].title.as_deref() != Some(&title) {
                        self.tenants[i].title = Some(title.clone());
                        self.emit(Event::TitleChanged { id, title });
                    }
                }
            }
            ToShell::AppId { id, app_id } => {
                let id = TenantId(id);
                if let Some(i) = self.index_of(id) {
                    if self.tenants[i].app_id.as_deref() != Some(&app_id) {
                        self.tenants[i].app_id = Some(app_id.clone());
                        self.emit(Event::AppIdChanged { id, app_id });
                    }
                }
            }
            ToShell::Commit {
                id,
                generation,
                buffer,
                width,
                height,
                fourcc,
                modifier,
                offset,
                stride,
                fence,
            } => {
                let id = TenantId(id);
                let c = CommitIn {
                    generation,
                    key: BufferKey(buffer),
                    width,
                    height,
                    fourcc,
                    modifier,
                    offset,
                    stride,
                    fence,
                };
                if let Err(why) = self.absorb(id, c, fds) {
                    self.emit(Event::Rejected { id: Some(id), why });
                }
            }
            ToShell::BufferGone { buffer } => {
                let key = BufferKey(buffer);
                self.registry.forget(key);
                if let Some(gate) = self.acquire.as_mut() {
                    gate.forget(key);
                }
            }
            ToShell::Rejected { id, why } => {
                self.emit(Event::Rejected {
                    id: id.map(TenantId),
                    why,
                });
            }
            ToShell::Spawned { pid } => self.spawn_reply = Some(Ok(pid)),
            ToShell::SpawnFailed { why } => self.spawn_reply = Some(Err(why)),
        }
    }

    /// The in-process `absorb_dmabuf`, with the dma-buf described by the wire
    /// instead of by smithay: acquire wait first, import once per buffer id,
    /// blit into the tenant's stable texture.
    fn absorb(
        &mut self,
        id: TenantId,
        c: CommitIn,
        fds: Vec<OwnedFd>,
    ) -> std::result::Result<(), String> {
        let index = self
            .index_of(id)
            .ok_or_else(|| format!("commit for unknown {id}"))?;
        let mut fds = fds.into_iter();
        let plane = fds.next().ok_or("commit carried no plane fd")?;
        let fourcc = wgpu_dmabuf::format::Fourcc::try_from(c.fourcc)
            .map_err(|_| format!("unknown fourcc 0x{:08x}", c.fourcc))?;
        if !texture::ACCEPTED_FOURCC.contains(&fourcc) {
            return Err(format!(
                "fourcc {fourcc:?} is not one of {:?}",
                texture::ACCEPTED_FOURCC
            ));
        }

        // **The acquire wait goes first**, before anything on this queue can
        // touch the buffer: the next submit carries it, and the next submit
        // is the blit below.
        let gate = self
            .acquire
            .as_mut()
            .expect("connect refuses a device without one");
        match (c.fence, fds.next()) {
            (Fence::Pending | Fence::Signalled, Some(fd)) => gate.wait_on(
                c.key,
                wgpu_dmabuf::sync::SyncFile::from_fd(fd),
                c.fence == Fence::Signalled,
            )?,
            _ => gate.wait_for(c.key, plane.as_fd())?,
        }
        self.tenants[index].skin.acquire_staged = true;

        let dmabuf = wgpu_dmabuf::Dmabuf::single_plane(
            c.width, c.height, fourcc, c.modifier, plane, c.offset, c.stride,
        );
        let device = self.device.clone();
        let (_tex_id, src, _imported) = self
            .registry
            .get_or_import(c.key, || {
                wgpu_dmabuf::import(&device, &dmabuf, texture::IMPORT_USAGE)
            })
            .map_err(|e| format!("dmabuf import failed: {e}"))?;

        // Same law as in-process: the handle changes only with the size, and
        // the change is announced before the commit that lands in it.
        let tenant = &mut self.tenants[index];
        let fits = tenant
            .skin
            .texture
            .as_ref()
            .is_some_and(|t| t.width() == c.width && t.height() == c.height);
        let resized = !fits;
        if resized {
            let tex = texture::create_surface_texture(
                &self.device,
                &format!("tenant#{}", id.0),
                c.width,
                c.height,
            );
            tenant.skin.texture = Some(Arc::new(tex));
        }
        tenant.skin.size = (c.width, c.height);
        let dst = tenant.skin.texture.clone().expect("just ensured");
        texture::blit(&self.device, &self.queue, &src, &dst, c.width, c.height);
        tenant.skin.generation = c.generation;
        tenant.buffer = Some(c.key.0);
        if resized {
            self.emit(Event::Resized {
                id,
                size: (c.width, c.height),
            });
        }
        self.emit(Event::Committed {
            id,
            generation: c.generation,
        });
        Ok(())
    }
}

struct CommitIn {
    generation: u64,
    key: BufferKey,
    width: u32,
    height: u32,
    fourcc: u32,
    modifier: u64,
    offset: u32,
    stride: u32,
    fence: Fence,
}

impl Drop for Glasspane {
    /// As in-process: withdraw staged waits and drain before the semaphores
    /// go.
    fn drop(&mut self) {
        if let Some(gate) = self.acquire.as_mut() {
            gate.shutdown();
        }
    }
}

fn default_path() -> std::path::PathBuf {
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        // SAFETY: getuid cannot fail.
        .unwrap_or_else(|| format!("/run/user/{}", unsafe { libc::getuid() }).into());
    std::path::PathBuf::from(runtime)
        .join("plurality")
        .join("glasspane.sock")
}
