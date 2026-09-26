//! **glasspane as a process**: the compositor with no device, owning the
//! Wayland socket, the tenants and their processes, with the shell a client of
//! it over a unix socket.
//!
//! The point is survival. A browser pane is a Wayland client of whatever
//! holds the socket; if that is the shell, a shell restart or crash takes
//! every pane with it. Here the socket is held by this process, which does
//! not render, and a shell that goes away and comes back finds every window
//! where it was — same tenant id, same buffer, no reload.
//!
//! ```text
//!   tenants ── wayland ──▶ Daemon (State, Sink::Export) ── seqpacket ──▶ shell
//!                           configure, xdg-shell, seat,     Commit + fds    remote::Glasspane
//!                           frame callbacks                  ◀── Presented   import, acquire, blit
//! ```
//!
//! # What changed, semantically
//!
//! **The shell is no longer there from the first commit.** In-process
//! glasspane blits each commit and forgets it. The daemon keeps, per tenant,
//! a dup of the last committed buffer's fds and metadata, and a shell that
//! connects is sent — for every tenant — a `Mapped` carrying the current
//! size, the title and app id, and a `Commit` of that buffer with a freshly
//! exported acquire fence. That replay is the new piece.
//!
//! **Frame callbacks.** With a shell connected they are answered when the
//! shell says it presented (the remote `tick`), exactly as in-process. With no
//! shell they are answered on a timer ([`Config::blind_interval`], 60 Hz by
//! default) so tenants keep running blind rather than stalling until a shell
//! returns.
//!
//! # Release is still smithay's, and that is a known hole
//!
//! smithay releases a `wl_buffer` when the next commit supersedes it
//! (`merge_into` in its compositor handlers). In-process that is safe
//! because the blit was submitted inside the commit handler, before the
//! release; here the blit happens in another process, later. So a client can
//! be handed a buffer back, and start drawing into it, while the shell is
//! still reading it. For a double-buffered client at a steady rate this is a
//! torn frame at worst; it is accepted for the spike. The fix, if it is ever
//! needed, is a smithay pre-commit blocker (`add_pre_commit_hook` /
//! `add_blocker`) cleared by a blit ack from the shell — not a fork, and not
//! built here. The replay buffer is only a dup of the fd, so keeping it does
//! not change when smithay releases anything either.
//!
//! # The spike's limits, stated
//!
//! * **dma-buf only.** `wl_shm` commits are rejected in daemon mode.
//! * **No popups, no input.** Popups are still hosted (configured, placed,
//!   dismissed on disconnect) but their buffers are not exported and their
//!   events are not forwarded; pointer, keyboard and the seat do not cross.
//! * **One shell.** A second connection replaces the first.

use std::collections::HashMap;
use std::ffi::OsString;
use std::os::fd::AsFd;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use smithay::reexports::calloop::EventLoop;

use crate::registry::BufferKey;
use crate::state::{Sink, State};
use crate::tenant::TenantId;
use crate::wire::{self, Conn, Fence, Listener, ToDaemon, ToShell};
use crate::{Caps, Error, Event, Result};

/// One item of the daemon's single ordered outbox. [`State`] pushes these in
/// export mode instead of queueing events and blitting.
#[derive(Debug)]
pub(crate) enum Outgoing {
    /// An event, in order with the commits around it.
    Event(Event),
    /// A committed dma-buf.
    Commit(Commit),
    /// A `wl_buffer` was destroyed: the shell drops its import.
    BufferGone(BufferKey),
}

/// Everything a shell needs to import one commit.
#[derive(Debug)]
pub(crate) struct Commit {
    pub(crate) id: TenantId,
    pub(crate) generation: u64,
    pub(crate) key: BufferKey,
    /// Owns a dup of the plane fd.
    pub(crate) dmabuf: wgpu_dmabuf::Dmabuf,
    pub(crate) fence: Option<wgpu_dmabuf::sync::ExportedFence>,
}

/// Export the buffer's implicit write fences, or `None` if the kernel will
/// not. `None` is not a pass: the shell then exports one itself from the
/// plane, and rejects the commit if that fails too.
pub(crate) fn export_fence(d: &wgpu_dmabuf::Dmabuf) -> Option<wgpu_dmabuf::sync::ExportedFence> {
    let plane = d.sole_plane().ok()?;
    wgpu_dmabuf::sync::export_sync_file(plane.as_fd(), wgpu_dmabuf::sync::SyncDirection::Read).ok()
}

/// How to run.
#[derive(Debug, Clone)]
pub struct Config {
    /// The Wayland socket name (`WAYLAND_DISPLAY`), or `None` for the next
    /// free `wayland-N` in `$XDG_RUNTIME_DIR`.
    pub socket_name: Option<String>,
    /// Where shells connect. See [`Config::default_control_path`].
    pub control_path: PathBuf,
    /// The size new tenants are configured with.
    pub default_size: (u32, u32),
    /// How often frame callbacks are answered while no shell is connected.
    pub blind_interval: Duration,
}

impl Config {
    /// `$XDG_RUNTIME_DIR/plurality/glasspane.sock`.
    pub fn default_control_path() -> PathBuf {
        let runtime = std::env::var_os("XDG_RUNTIME_DIR")
            // SAFETY: getuid cannot fail.
            .unwrap_or_else(|| format!("/run/user/{}", unsafe { libc::getuid() }).into());
        PathBuf::from(runtime)
            .join("plurality")
            .join("glasspane.sock")
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            socket_name: None,
            control_path: Self::default_control_path(),
            default_size: (1280, 800),
            blind_interval: Duration::from_micros(16_667),
        }
    }
}

/// The last buffer a tenant committed, kept for a shell that connects later.
#[derive(Debug)]
struct Last {
    key: BufferKey,
    generation: u64,
    dmabuf: wgpu_dmabuf::Dmabuf,
    /// The client destroyed the `wl_buffer` after committing it. The memory is
    /// still ours through the dup, so it is replayed, and followed by a
    /// `BufferGone` so the shell does not keep the import.
    orphan: bool,
}

/// Counters, for a harness.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// Shells that have connected.
    pub shells: u64,
    /// Shells that have gone away.
    pub shells_gone: u64,
    /// Frame callbacks answered on the blind timer.
    pub blind_callbacks: u64,
    /// Frame callbacks answered on a shell's `Presented`.
    pub presented_callbacks: u64,
    /// Commits exported.
    pub commits: u64,
    /// Commits sent in replays.
    pub replayed: u64,
}

/// The daemon: a compositor with no device, a control socket, and the
/// processes it spawned.
pub struct Daemon {
    event_loop: EventLoop<'static, State>,
    state: State,
    socket_name: OsString,
    control_path: PathBuf,
    listener: Listener,
    shell: Option<Conn>,
    last: HashMap<TenantId, Last>,
    children: Vec<Child>,
    blind_interval: Duration,
    last_blind_tick: Instant,
    dmabuf: bool,
    stats: Stats,
}

impl std::fmt::Debug for Daemon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Daemon")
            .field("socket_name", &self.socket_name)
            .field("control_path", &self.control_path)
            .field("tenants", &self.state.tenants.len())
            .field("shell", &self.shell.is_some())
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl Daemon {
    /// Open the Wayland socket and the control socket.
    ///
    /// `adapter` is used only to enumerate the dma-buf formats worth
    /// advertising — no device is opened. `None` advertises no dma-buf at
    /// all, which in daemon mode means no tenant can show anything.
    pub fn new(adapter: Option<&wgpu::Adapter>, config: Config) -> Result<Self> {
        let mut caps = Caps::new().with_size(config.default_size.0, config.default_size.1);
        caps.adapter = adapter;
        caps.socket_name = config.socket_name.clone();
        let sink = Sink::Export(Default::default());
        let (event_loop, state, socket_name) = crate::build(&caps, sink, adapter.is_some())?;
        let dmabuf = state.dmabuf_global.is_some();

        if let Some(dir) = config.control_path.parent() {
            use std::os::unix::fs::DirBuilderExt;
            let _ = std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(dir);
        }
        let listener = Listener::bind(&config.control_path).map_err(|e| {
            Error::Socket(format!(
                "control socket {}: {e}",
                config.control_path.display()
            ))
        })?;
        Ok(Self {
            event_loop,
            state,
            socket_name,
            control_path: config.control_path,
            listener,
            shell: None,
            last: HashMap::new(),
            children: Vec::new(),
            blind_interval: config.blind_interval,
            last_blind_tick: Instant::now(),
            dmabuf,
            stats: Stats::default(),
        })
    }

    /// The Wayland socket name, i.e. `WAYLAND_DISPLAY` for a tenant.
    pub fn socket_name(&self) -> &std::ffi::OsStr {
        &self.socket_name
    }

    /// Where shells connect.
    pub fn control_path(&self) -> &Path {
        &self.control_path
    }

    /// Whether dma-buf is advertised. In daemon mode, `false` means nothing
    /// can be shown.
    pub fn advertises_dmabuf(&self) -> bool {
        self.dmabuf
    }

    /// Whether a shell is connected right now.
    pub fn shell_connected(&self) -> bool {
        self.shell.is_some()
    }

    /// Counters.
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Spawn a client on the Wayland socket. The daemon owns the child: it
    /// is reaped here and killed when the daemon is dropped. Returns its pid.
    pub fn spawn(&mut self, mut command: Command) -> Result<u32> {
        command
            .env("WAYLAND_DISPLAY", &self.socket_name)
            .env_remove("DISPLAY")
            .env_remove("WAYLAND_SOCKET");
        // The socket lives in *this* process's runtime dir; a command that
        // arrived over the wire with some other one would look for it there.
        if let Some(dir) = std::env::var_os("XDG_RUNTIME_DIR") {
            command.env("XDG_RUNTIME_DIR", dir);
        }
        let child = command.spawn().map_err(Error::Spawn)?;
        let pid = child.id();
        self.children.push(child);
        Ok(pid)
    }

    /// Run until `stop` is set.
    pub fn run(&mut self, stop: &AtomicBool) -> Result<()> {
        while !stop.load(Ordering::Relaxed) {
            self.turn(Duration::from_millis(50))?;
        }
        Ok(())
    }

    /// One turn: Wayland dispatch, the outbox, the control socket, and the
    /// blind clock. Blocks at most `timeout`, less if the blind clock is due.
    pub fn turn(&mut self, timeout: Duration) -> Result<()> {
        let timeout = if self.shell.is_none() {
            let due = self.last_blind_tick + self.blind_interval;
            timeout.min(due.saturating_duration_since(Instant::now()))
        } else {
            timeout
        };

        // The control fds are not calloop sources (they belong to this
        // struct, not to State), so one poll covers all three and the Wayland
        // dispatch below then runs without blocking.
        let mut fds = vec![self.event_loop_fd(), self.listener.as_fd()];
        if let Some(shell) = &self.shell {
            fds.push(shell.as_fd());
        }
        let _ = wire::poll_readable(&fds, Some(timeout));
        drop(fds);

        self.event_loop
            .dispatch(Some(Duration::ZERO), &mut self.state)
            .map_err(|e| Error::Dispatch(e.to_string()))?;
        self.state.prune_popups();
        self.state.flush();

        // Drain before accepting, so a new shell's replay is built from state
        // that already includes everything the outbox said.
        self.drain_outbox();
        self.accept();
        self.read_shell();
        self.drain_outbox();

        if self.shell.is_none() && self.last_blind_tick.elapsed() >= self.blind_interval {
            self.last_blind_tick = Instant::now();
            let ms = self.state.start.elapsed().as_millis() as u32;
            self.stats.blind_callbacks += self.state.release_frame_callbacks(ms) as u64;
        }
        self.reap();
        Ok(())
    }

    fn event_loop_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        use std::os::fd::AsFd;
        self.event_loop.as_fd()
    }

    fn outbox(&mut self) -> Vec<Outgoing> {
        match &mut self.state.sink {
            Sink::Export(out) => out.drain(..).collect(),
            Sink::Gpu(_) => Vec::new(),
        }
    }

    fn drain_outbox(&mut self) {
        for item in self.outbox() {
            match item {
                Outgoing::Event(event) => self.forward_event(event),
                Outgoing::Commit(commit) => {
                    self.stats.commits += 1;
                    let msg = commit_message(&commit);
                    if let Some(shell) = &self.shell {
                        let mut fds = vec![commit.dmabuf.planes[0].fd.as_fd()];
                        if let Some(f) = &commit.fence {
                            fds.push(f.sync_file.as_fd());
                        }
                        let r = shell.send(&msg.encode(), &fds);
                        drop(fds);
                        self.check(r);
                    }
                    self.last.insert(
                        commit.id,
                        Last {
                            key: commit.key,
                            generation: commit.generation,
                            dmabuf: commit.dmabuf,
                            orphan: false,
                        },
                    );
                }
                Outgoing::BufferGone(key) => {
                    for last in self.last.values_mut().filter(|l| l.key == key) {
                        last.orphan = true;
                    }
                    self.send(ToShell::BufferGone { buffer: key.0 });
                }
            }
        }
    }

    fn mapped_message(&self, id: TenantId, pid: Option<u32>) -> ToShell {
        let (size, configured) = match self.state.index_of(id) {
            Some(i) => (
                self.state.tenants[i].skin.size,
                self.state.tenants[i].configured,
            ),
            None => (self.state.default_size, None),
        };
        ToShell::Mapped {
            id: id.0,
            pid,
            size,
            configured,
        }
    }

    fn forward_event(&mut self, event: Event) {
        let msg = match event {
            Event::Mapped { id, pid } => self.mapped_message(id, pid),
            Event::Unmapped { id } => {
                self.last.remove(&id);
                ToShell::Unmapped { id: id.0 }
            }
            Event::TitleChanged { id, title } => ToShell::Title { id: id.0, title },
            Event::AppIdChanged { id, app_id } => ToShell::AppId { id: id.0, app_id },
            Event::Rejected { id, why } => {
                eprintln!("glasspane-daemon: rejected {id:?}: {why}");
                ToShell::Rejected {
                    id: id.map(|i| i.0),
                    why,
                }
            }
            // Resized and Committed are the shell's to derive from Commit;
            // popups are not in the spike.
            _ => return,
        };
        self.send(msg);
    }

    fn send(&mut self, msg: ToShell) {
        if let Some(shell) = &self.shell {
            let r = shell.send(&msg.encode(), &[]);
            self.check(r);
        }
    }

    /// A failed send is a shell that went away.
    fn check(&mut self, r: std::io::Result<()>) {
        if let Err(e) = r {
            self.drop_shell(&format!("send: {e}"));
        }
    }

    fn drop_shell(&mut self, why: &str) {
        if self.shell.take().is_some() {
            self.stats.shells_gone += 1;
            self.last_blind_tick = Instant::now();
            eprintln!("glasspane-daemon: shell gone ({why}); tenants run blind");
        }
    }

    fn accept(&mut self) {
        loop {
            match self.listener.accept() {
                Ok(Some(conn)) => {
                    if self.shell.is_some() {
                        self.drop_shell("replaced by a new connection");
                    }
                    self.shell = Some(conn);
                    self.stats.shells += 1;
                    eprintln!(
                        "glasspane-daemon: shell connected; replaying {} tenant(s)",
                        self.state.tenants.len()
                    );
                    self.replay();
                }
                Ok(None) => return,
                Err(e) => {
                    eprintln!("glasspane-daemon: accept: {e}");
                    return;
                }
            }
        }
    }

    /// Everything a shell that was not here needs: hello, then per tenant a
    /// Mapped with its current size, its title and app id, and its last
    /// buffer.
    fn replay(&mut self) {
        self.send(ToShell::Hello {
            version: wire::VERSION,
            wayland_display: self.socket_name.to_string_lossy().into_owned(),
            dmabuf: self.dmabuf,
        });
        let tenants: Vec<_> = self
            .state
            .tenants
            .iter()
            .map(|t| (t.id, t.pid, t.title.clone(), t.app_id.clone()))
            .collect();
        for (id, pid, title, app_id) in tenants {
            let mapped = self.mapped_message(id, pid);
            self.send(mapped);
            if let Some(title) = title {
                self.send(ToShell::Title { id: id.0, title });
            }
            if let Some(app_id) = app_id {
                self.send(ToShell::AppId { id: id.0, app_id });
            }
            let Some(last) = self.last.get(&id) else {
                continue;
            };
            // A fresh fence, exported now: the one from the original commit
            // was consumed by whichever shell received it, if any.
            let fence = export_fence(&last.dmabuf);
            let msg =
                commit_message_parts(id, last.generation, last.key, &last.dmabuf, fence.as_ref());
            let orphan = last.orphan.then_some(last.key);
            if let Some(shell) = &self.shell {
                let mut fds = vec![last.dmabuf.planes[0].fd.as_fd()];
                if let Some(f) = &fence {
                    fds.push(f.sync_file.as_fd());
                }
                let r = shell.send(&msg.encode(), &fds);
                drop(fds);
                self.stats.replayed += 1;
                self.check(r);
            }
            if let Some(key) = orphan {
                self.send(ToShell::BufferGone { buffer: key.0 });
            }
        }
    }

    fn read_shell(&mut self) {
        loop {
            let Some(shell) = &self.shell else {
                return;
            };
            let msg = match shell.recv() {
                Ok(Some((bytes, _fds))) => ToDaemon::decode(&bytes),
                Ok(None) => return,
                Err(e) => {
                    self.drop_shell(&format!("recv: {e}"));
                    return;
                }
            };
            match msg {
                Ok(msg) => self.handle(msg),
                Err(e) => {
                    self.drop_shell(&format!("decode: {e}"));
                    return;
                }
            }
        }
    }

    fn handle(&mut self, msg: ToDaemon) {
        match msg {
            ToDaemon::Presented => {
                let ms = self.state.start.elapsed().as_millis() as u32;
                self.stats.presented_callbacks += self.state.release_frame_callbacks(ms) as u64;
            }
            ToDaemon::Configure { id, width, height } => {
                let id = TenantId(id);
                if let Some(index) = self.state.index_of(id) {
                    let toplevel = self.state.tenants[index].toplevel.clone();
                    toplevel.with_pending_state(|s| {
                        s.size = Some((width as i32, height as i32).into());
                    });
                    toplevel.send_pending_configure();
                    self.state.tenants[index].configured = Some((width, height));
                    self.state.flush();
                }
            }
            ToDaemon::Close { id } => {
                if let Some(index) = self.state.index_of(TenantId(id)) {
                    self.state.tenants[index].toplevel.send_close();
                    self.state.flush();
                }
            }
            ToDaemon::Spawn { program, args, env } => {
                let mut cmd = Command::new(&program);
                cmd.args(&args);
                for (k, v) in env {
                    match v {
                        Some(v) => cmd.env(k, v),
                        None => cmd.env_remove(k),
                    };
                }
                let reply = match self.spawn(cmd) {
                    Ok(pid) => {
                        eprintln!("glasspane-daemon: spawned {program} (pid {pid})");
                        ToShell::Spawned { pid }
                    }
                    Err(e) => ToShell::SpawnFailed { why: e.to_string() },
                };
                self.send(reply);
            }
        }
    }

    fn reap(&mut self) {
        self.children.retain_mut(|c| match c.try_wait() {
            Ok(Some(status)) => {
                eprintln!("glasspane-daemon: child {} exited: {status}", c.id());
                false
            }
            _ => true,
        });
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = std::fs::remove_file(&self.control_path);
    }
}

fn commit_message(c: &Commit) -> ToShell {
    commit_message_parts(c.id, c.generation, c.key, &c.dmabuf, c.fence.as_ref())
}

fn commit_message_parts(
    id: TenantId,
    generation: u64,
    key: BufferKey,
    d: &wgpu_dmabuf::Dmabuf,
    fence: Option<&wgpu_dmabuf::sync::ExportedFence>,
) -> ToShell {
    let plane = &d.planes[0];
    ToShell::Commit {
        id: id.0,
        generation,
        buffer: key.0,
        width: d.width,
        height: d.height,
        fourcc: d.fourcc as u32,
        modifier: d.modifier,
        offset: plane.offset,
        stride: plane.stride,
        fence: match fence {
            None => Fence::None,
            Some(f) if f.already_signalled => Fence::Signalled,
            Some(_) => Fence::Pending,
        },
    }
}
