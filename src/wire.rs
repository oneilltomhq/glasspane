//! The daemon ⇄ shell wire: one `SOCK_SEQPACKET` unix socket, one message per
//! packet, fds alongside with `SCM_RIGHTS`.
//!
//! `SEQPACKET` rather than a stream because it keeps message boundaries, so an
//! fd is attached to exactly the message that names it and there is no framing
//! to get wrong. Each packet is a one-byte tag followed by little-endian
//! fields; a string is a `u32` length and UTF-8 bytes; an `Option<T>` is a
//! `u8` presence flag and then `T`. Nothing is versioned beyond
//! [`VERSION`] in the hello: both ends are this crate.
//!
//! ```text
//!  daemon → shell                       shell → daemon
//!  Hello{version,wayland,dmabuf}        Presented          (tick: answer frame callbacks)
//!  Mapped{id,pid,size,configured}       Configure{id,w,h}
//!  Unmapped{id}                         Close{id}
//!  Title{id,s} AppId{id,s}              Spawn{program,args,env}
//!  Commit{id,gen,buffer,w,h,fourcc,
//!         modifier,offset,stride,fence} + fds [plane, sync_file?]
//!  BufferGone{buffer}
//!  Rejected{id?,why}
//!  Spawned{pid} SpawnFailed{why}
//! ```
//!
//! There is **no release message**. smithay releases a `wl_buffer` when the
//! next commit supersedes it (`merge_into` in its compositor handlers), which
//! in the daemon is before the shell has necessarily blitted it; see the
//! daemon's module doc for what that costs and how it would be fixed.

// Each half is used by one side only; a build with one side is not dead code.
#![cfg_attr(not(all(feature = "daemon", feature = "remote")), allow(dead_code))]

use std::io;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::path::Path;
use std::time::Duration;

/// Bumped when a message changes shape.
pub(crate) const VERSION: u32 = 1;

/// The most fds any one message carries (a plane and a fence), plus slack.
const MAX_FDS: usize = 4;
/// The largest packet either end accepts. A title is the only unbounded
/// field, and a 64 KiB title is a client bug.
const MAX_PACKET: usize = 64 * 1024;

/// What a Commit's fence field says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fence {
    /// No sync_file came with it; the shell exports one from the plane itself.
    None,
    /// A sync_file is the second fd.
    Pending,
    /// A sync_file is the second fd, and it had already signalled when the
    /// daemon exported it.
    Signalled,
}

/// Daemon to shell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToShell {
    Hello {
        version: u32,
        wayland_display: String,
        dmabuf: bool,
    },
    Mapped {
        id: u64,
        pid: Option<u32>,
        size: (u32, u32),
        configured: Option<(u32, u32)>,
    },
    Unmapped {
        id: u64,
    },
    Title {
        id: u64,
        title: String,
    },
    AppId {
        id: u64,
        app_id: String,
    },
    /// One committed dma-buf. fds: `[plane]` or `[plane, sync_file]`.
    Commit {
        id: u64,
        generation: u64,
        buffer: u64,
        width: u32,
        height: u32,
        fourcc: u32,
        modifier: u64,
        offset: u32,
        stride: u32,
        fence: Fence,
    },
    BufferGone {
        buffer: u64,
    },
    Rejected {
        id: Option<u64>,
        why: String,
    },
    Spawned {
        pid: u32,
    },
    SpawnFailed {
        why: String,
    },
}

/// Shell to daemon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToDaemon {
    /// The shell presented a frame: answer the frame callbacks harvested since
    /// the last one. The remote `tick`.
    Presented,
    Configure {
        id: u64,
        width: u32,
        height: u32,
    },
    Close {
        id: u64,
    },
    /// Run a command on the daemon's Wayland socket; the daemon owns and
    /// reaps it. `env` entries with `None` are removals.
    Spawn {
        program: String,
        args: Vec<String>,
        env: Vec<(String, Option<String>)>,
    },
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

#[derive(Default)]
struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) -> &mut Self {
        self.0.push(v);
        self
    }
    fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn u64(&mut self, v: u64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn str(&mut self, v: &str) -> &mut Self {
        self.u32(v.len() as u32);
        self.0.extend_from_slice(v.as_bytes());
        self
    }
    fn opt_u32(&mut self, v: Option<u32>) -> &mut Self {
        match v {
            Some(v) => self.u8(1).u32(v),
            None => self.u8(0),
        }
    }
    fn opt_u64(&mut self, v: Option<u64>) -> &mut Self {
        match v {
            Some(v) => self.u8(1).u64(v),
            None => self.u8(0),
        }
    }
    fn opt_size(&mut self, v: Option<(u32, u32)>) -> &mut Self {
        match v {
            Some((w, h)) => self.u8(1).u32(w).u32(h),
            None => self.u8(0),
        }
    }
}

struct R<'a>(&'a [u8]);

fn short() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "glasspane wire: truncated message",
    )
}

impl R<'_> {
    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        if self.0.len() < n {
            return Err(short());
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> io::Result<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "glasspane wire: bad utf-8"))
    }
    fn opt_u32(&mut self) -> io::Result<Option<u32>> {
        Ok(if self.u8()? != 0 {
            Some(self.u32()?)
        } else {
            None
        })
    }
    fn opt_u64(&mut self) -> io::Result<Option<u64>> {
        Ok(if self.u8()? != 0 {
            Some(self.u64()?)
        } else {
            None
        })
    }
    fn opt_size(&mut self) -> io::Result<Option<(u32, u32)>> {
        Ok(if self.u8()? != 0 {
            Some((self.u32()?, self.u32()?))
        } else {
            None
        })
    }
}

fn bad_tag(t: u8) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("glasspane wire: unknown tag {t}"),
    )
}

impl ToShell {
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut w = W::default();
        match self {
            ToShell::Hello {
                version,
                wayland_display,
                dmabuf,
            } => {
                w.u8(1).u32(*version).str(wayland_display).u8(*dmabuf as u8);
            }
            ToShell::Mapped {
                id,
                pid,
                size,
                configured,
            } => {
                w.u8(2)
                    .u64(*id)
                    .opt_u32(*pid)
                    .u32(size.0)
                    .u32(size.1)
                    .opt_size(*configured);
            }
            ToShell::Unmapped { id } => {
                w.u8(3).u64(*id);
            }
            ToShell::Title { id, title } => {
                w.u8(4).u64(*id).str(title);
            }
            ToShell::AppId { id, app_id } => {
                w.u8(5).u64(*id).str(app_id);
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
                let f = match fence {
                    Fence::None => 0,
                    Fence::Pending => 1,
                    Fence::Signalled => 2,
                };
                w.u8(6)
                    .u64(*id)
                    .u64(*generation)
                    .u64(*buffer)
                    .u32(*width)
                    .u32(*height)
                    .u32(*fourcc)
                    .u64(*modifier)
                    .u32(*offset)
                    .u32(*stride)
                    .u8(f);
            }
            ToShell::BufferGone { buffer } => {
                w.u8(7).u64(*buffer);
            }
            ToShell::Rejected { id, why } => {
                w.u8(8).opt_u64(*id).str(why);
            }
            ToShell::Spawned { pid } => {
                w.u8(9).u32(*pid);
            }
            ToShell::SpawnFailed { why } => {
                w.u8(10).str(why);
            }
        }
        w.0
    }

    pub(crate) fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut r = R(bytes);
        Ok(match r.u8()? {
            1 => ToShell::Hello {
                version: r.u32()?,
                wayland_display: r.str()?,
                dmabuf: r.u8()? != 0,
            },
            2 => ToShell::Mapped {
                id: r.u64()?,
                pid: r.opt_u32()?,
                size: (r.u32()?, r.u32()?),
                configured: r.opt_size()?,
            },
            3 => ToShell::Unmapped { id: r.u64()? },
            4 => ToShell::Title {
                id: r.u64()?,
                title: r.str()?,
            },
            5 => ToShell::AppId {
                id: r.u64()?,
                app_id: r.str()?,
            },
            6 => ToShell::Commit {
                id: r.u64()?,
                generation: r.u64()?,
                buffer: r.u64()?,
                width: r.u32()?,
                height: r.u32()?,
                fourcc: r.u32()?,
                modifier: r.u64()?,
                offset: r.u32()?,
                stride: r.u32()?,
                fence: match r.u8()? {
                    0 => Fence::None,
                    1 => Fence::Pending,
                    2 => Fence::Signalled,
                    t => return Err(bad_tag(t)),
                },
            },
            7 => ToShell::BufferGone { buffer: r.u64()? },
            8 => ToShell::Rejected {
                id: r.opt_u64()?,
                why: r.str()?,
            },
            9 => ToShell::Spawned { pid: r.u32()? },
            10 => ToShell::SpawnFailed { why: r.str()? },
            t => return Err(bad_tag(t)),
        })
    }
}

impl ToDaemon {
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut w = W::default();
        match self {
            ToDaemon::Presented => {
                w.u8(1);
            }
            ToDaemon::Configure { id, width, height } => {
                w.u8(2).u64(*id).u32(*width).u32(*height);
            }
            ToDaemon::Close { id } => {
                w.u8(3).u64(*id);
            }
            ToDaemon::Spawn { program, args, env } => {
                w.u8(4).str(program).u32(args.len() as u32);
                for a in args {
                    w.str(a);
                }
                w.u32(env.len() as u32);
                for (k, v) in env {
                    w.str(k);
                    match v {
                        Some(v) => w.u8(1).str(v),
                        None => w.u8(0),
                    };
                }
            }
        }
        w.0
    }

    pub(crate) fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut r = R(bytes);
        Ok(match r.u8()? {
            1 => ToDaemon::Presented,
            2 => ToDaemon::Configure {
                id: r.u64()?,
                width: r.u32()?,
                height: r.u32()?,
            },
            3 => ToDaemon::Close { id: r.u64()? },
            4 => {
                let program = r.str()?;
                let n = r.u32()? as usize;
                let mut args = Vec::with_capacity(n.min(1024));
                for _ in 0..n {
                    args.push(r.str()?);
                }
                let n = r.u32()? as usize;
                let mut env = Vec::with_capacity(n.min(1024));
                for _ in 0..n {
                    let k = r.str()?;
                    let v = if r.u8()? != 0 { Some(r.str()?) } else { None };
                    env.push((k, v));
                }
                ToDaemon::Spawn { program, args, env }
            }
            t => return Err(bad_tag(t)),
        })
    }
}

// ---------------------------------------------------------------------------
// The socket
// ---------------------------------------------------------------------------

fn cvt(rc: libc::c_int) -> io::Result<libc::c_int> {
    if rc < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(rc)
    }
}

fn sockaddr(path: &Path) -> io::Result<(libc::sockaddr_un, libc::socklen_t)> {
    use std::os::unix::ffi::OsStrExt;
    // SAFETY: an all-zero sockaddr_un is a valid value.
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let bytes = path.as_os_str().as_bytes();
    if bytes.len() >= addr.sun_path.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "socket path is {} bytes, the limit is {}: {}",
                bytes.len(),
                addr.sun_path.len() - 1,
                path.display()
            ),
        ));
    }
    for (d, s) in addr.sun_path.iter_mut().zip(bytes) {
        *d = *s as libc::c_char;
    }
    let len = std::mem::size_of::<libc::sa_family_t>() + bytes.len() + 1;
    Ok((addr, len as libc::socklen_t))
}

fn seqpacket() -> io::Result<OwnedFd> {
    // SAFETY: plain socket(2); the fd is owned from here.
    let fd =
        cvt(unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0) })?;
    // SAFETY: just returned by socket(2) and owned by nobody else.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// A listening `SOCK_SEQPACKET` socket, non-blocking.
pub(crate) struct Listener {
    fd: OwnedFd,
}

impl Listener {
    /// Bind `path`, replacing a stale socket file left by a daemon that died.
    pub(crate) fn bind(path: &Path) -> io::Result<Self> {
        let (addr, len) = sockaddr(path)?;
        let fd = seqpacket()?;
        let _ = std::fs::remove_file(path);
        // SAFETY: `addr` is a live, initialised sockaddr_un of length `len`.
        cvt(unsafe { libc::bind(fd.as_raw_fd(), (&raw const addr).cast(), len) })?;
        // SAFETY: a bound socket.
        cvt(unsafe { libc::listen(fd.as_raw_fd(), 4) })?;
        set_nonblocking(fd.as_raw_fd())?;
        Ok(Self { fd })
    }

    /// Accept one pending connection, or `None` if there is none.
    pub(crate) fn accept(&self) -> io::Result<Option<Conn>> {
        // SAFETY: accept4 on a listening socket with null address out-params.
        let rc = unsafe {
            libc::accept4(
                self.fd.as_raw_fd(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                libc::SOCK_CLOEXEC,
            )
        };
        if rc < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::WouldBlock {
                return Ok(None);
            }
            return Err(e);
        }
        // SAFETY: just returned by accept4.
        let fd = unsafe { OwnedFd::from_raw_fd(rc) };
        Conn::new(fd).map(Some)
    }

    pub(crate) fn as_fd(&self) -> BorrowedFd<'_> {
        use std::os::fd::AsFd;
        self.fd.as_fd()
    }
}

fn set_nonblocking(fd: RawFd) -> io::Result<()> {
    // SAFETY: fcntl on an fd we own.
    let flags = cvt(unsafe { libc::fcntl(fd, libc::F_GETFL) })?;
    // SAFETY: as above.
    cvt(unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) })?;
    Ok(())
}

/// One end of a connection.
///
/// Sends block, with a timeout, so a peer that stops reading is dropped
/// rather than queued against forever; receives never block (the caller
/// polls).
pub(crate) struct Conn {
    fd: OwnedFd,
}

impl std::fmt::Debug for Conn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Conn({})", self.fd.as_raw_fd())
    }
}

impl Conn {
    fn new(fd: OwnedFd) -> io::Result<Self> {
        let tv = libc::timeval {
            tv_sec: 2,
            tv_usec: 0,
        };
        // SAFETY: a live timeval of the size given.
        cvt(unsafe {
            libc::setsockopt(
                fd.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_SNDTIMEO,
                (&raw const tv).cast(),
                std::mem::size_of::<libc::timeval>() as libc::socklen_t,
            )
        })?;
        Ok(Self { fd })
    }

    /// Connect to a daemon's socket.
    pub(crate) fn connect(path: &Path) -> io::Result<Self> {
        let (addr, len) = sockaddr(path)?;
        let fd = seqpacket()?;
        // SAFETY: `addr` is a live, initialised sockaddr_un of length `len`.
        cvt(unsafe { libc::connect(fd.as_raw_fd(), (&raw const addr).cast(), len) })?;
        Self::new(fd)
    }

    pub(crate) fn as_fd(&self) -> BorrowedFd<'_> {
        use std::os::fd::AsFd;
        self.fd.as_fd()
    }

    /// Send one packet with `fds` attached.
    pub(crate) fn send(&self, bytes: &[u8], fds: &[BorrowedFd<'_>]) -> io::Result<()> {
        assert!(fds.len() <= MAX_FDS);
        let mut iov = libc::iovec {
            iov_base: bytes.as_ptr() as *mut libc::c_void,
            iov_len: bytes.len(),
        };
        let raw: Vec<RawFd> = fds.iter().map(|f| f.as_raw_fd()).collect();
        let payload = std::mem::size_of_val(raw.as_slice()) as u32;
        // SAFETY: CMSG_SPACE is a pure size computation.
        let space = unsafe { libc::CMSG_SPACE(payload) } as usize;
        let mut cbuf = vec![0u64; space.div_ceil(8)];
        // SAFETY: an all-zero msghdr is valid; every pointer set below is
        // live for the sendmsg call.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &raw mut iov;
        msg.msg_iovlen = 1;
        if !raw.is_empty() {
            msg.msg_control = cbuf.as_mut_ptr().cast();
            msg.msg_controllen = space as _;
            // SAFETY: the control buffer is `space` bytes, enough for one
            // header carrying `raw.len()` fds, and is 8-aligned.
            unsafe {
                let cmsg = libc::CMSG_FIRSTHDR(&msg);
                (*cmsg).cmsg_level = libc::SOL_SOCKET;
                (*cmsg).cmsg_type = libc::SCM_RIGHTS;
                (*cmsg).cmsg_len = libc::CMSG_LEN(payload) as _;
                std::ptr::copy_nonoverlapping(
                    raw.as_ptr(),
                    libc::CMSG_DATA(cmsg).cast::<RawFd>(),
                    raw.len(),
                );
            }
        }
        // SAFETY: `msg` is fully initialised above.
        let n = unsafe { libc::sendmsg(self.fd.as_raw_fd(), &msg, libc::MSG_NOSIGNAL) };
        if n < 0 {
            return Err(io::Error::last_os_error());
        }
        if n as usize != bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "short seqpacket send",
            ));
        }
        Ok(())
    }

    /// Receive one packet if one is waiting.
    ///
    /// `Ok(None)`: nothing to read. `Err(UnexpectedEof)`: the peer is gone.
    pub(crate) fn recv(&self) -> io::Result<Option<(Vec<u8>, Vec<OwnedFd>)>> {
        let mut buf = vec![0u8; MAX_PACKET];
        let mut iov = libc::iovec {
            iov_base: buf.as_mut_ptr().cast(),
            iov_len: buf.len(),
        };
        // SAFETY: pure size computation.
        let space =
            unsafe { libc::CMSG_SPACE((MAX_FDS * std::mem::size_of::<RawFd>()) as u32) } as usize;
        let mut cbuf = vec![0u64; space.div_ceil(8)];
        // SAFETY: as in `send`.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &raw mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cbuf.as_mut_ptr().cast();
        msg.msg_controllen = space as _;
        // SAFETY: `msg` points at live buffers of the lengths it states.
        let n = unsafe {
            libc::recvmsg(
                self.fd.as_raw_fd(),
                &mut msg,
                libc::MSG_DONTWAIT | libc::MSG_CMSG_CLOEXEC,
            )
        };
        if n < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::WouldBlock {
                return Ok(None);
            }
            return Err(e);
        }
        // Collect fds before anything can return early, so none leak.
        let mut fds = Vec::new();
        // SAFETY: walking the control buffer the kernel just filled, with the
        // CMSG_* accessors, and adopting each SCM_RIGHTS fd exactly once.
        unsafe {
            let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
            while !cmsg.is_null() {
                if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                    let data = libc::CMSG_DATA(cmsg).cast::<RawFd>();
                    let count = ((*cmsg).cmsg_len as usize - libc::CMSG_LEN(0) as usize)
                        / std::mem::size_of::<RawFd>();
                    for i in 0..count {
                        fds.push(OwnedFd::from_raw_fd(std::ptr::read_unaligned(data.add(i))));
                    }
                }
                cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
            }
        }
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "peer closed"));
        }
        if msg.msg_flags & (libc::MSG_TRUNC | libc::MSG_CTRUNC) != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated packet",
            ));
        }
        buf.truncate(n as usize);
        Ok(Some((buf, fds)))
    }

    /// Wait until readable, or `timeout` passes. `None` waits forever.
    pub(crate) fn wait_readable(&self, timeout: Option<Duration>) -> io::Result<bool> {
        poll_readable(&[self.as_fd()], timeout).map(|r| r[0])
    }
}

/// `poll(2)` on several fds for readability.
pub(crate) fn poll_readable(
    fds: &[BorrowedFd<'_>],
    timeout: Option<Duration>,
) -> io::Result<Vec<bool>> {
    let mut pfds: Vec<libc::pollfd> = fds
        .iter()
        .map(|f| libc::pollfd {
            fd: f.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        })
        .collect();
    let ms = timeout.map_or(-1, |t| t.as_millis().min(i32::MAX as u128) as i32);
    // SAFETY: a live array of initialised pollfds.
    let rc = unsafe { libc::poll(pfds.as_mut_ptr(), pfds.len() as libc::nfds_t, ms) };
    if rc < 0 {
        let e = io::Error::last_os_error();
        if e.kind() == io::ErrorKind::Interrupted {
            return Ok(vec![false; fds.len()]);
        }
        return Err(e);
    }
    Ok(pfds.iter().map(|p| p.revents != 0).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every message round-trips through its encoding unchanged.
    #[test]
    fn messages_round_trip() {
        let shell = [
            ToShell::Hello {
                version: VERSION,
                wayland_display: "wayland-9".into(),
                dmabuf: true,
            },
            ToShell::Mapped {
                id: 3,
                pid: Some(42),
                size: (1280, 800),
                configured: None,
            },
            ToShell::Title {
                id: 3,
                title: "ünïcode".into(),
            },
            ToShell::Commit {
                id: 3,
                generation: 7,
                buffer: u64::MAX - 1,
                width: 256,
                height: 128,
                fourcc: 0x3432_5241,
                modifier: 0x0100_0000_0000_0001,
                offset: 0,
                stride: 1024,
                fence: Fence::Signalled,
            },
            ToShell::BufferGone { buffer: 9 },
            ToShell::Rejected {
                id: None,
                why: "no".into(),
            },
        ];
        for m in shell {
            assert_eq!(ToShell::decode(&m.encode()).unwrap(), m);
        }
        let daemon = [
            ToDaemon::Presented,
            ToDaemon::Configure {
                id: 1,
                width: 2,
                height: 3,
            },
            ToDaemon::Spawn {
                program: "chromium".into(),
                args: vec!["--a".into(), "b".into()],
                env: vec![("X".into(), Some("1".into())), ("DISPLAY".into(), None)],
            },
        ];
        for m in daemon {
            assert_eq!(ToDaemon::decode(&m.encode()).unwrap(), m);
        }
    }

    /// An fd crosses a real socket pair and names the same file.
    #[test]
    fn an_fd_crosses() {
        let mut sv = [0; 2];
        // SAFETY: socketpair into a live array.
        cvt(unsafe {
            libc::socketpair(
                libc::AF_UNIX,
                libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC,
                0,
                sv.as_mut_ptr(),
            )
        })
        .unwrap();
        // SAFETY: both fds were just created and are owned here.
        let (a, b) = unsafe { (OwnedFd::from_raw_fd(sv[0]), OwnedFd::from_raw_fd(sv[1])) };
        let (a, b) = (Conn::new(a).unwrap(), Conn::new(b).unwrap());
        let file = std::fs::File::open("/proc/self/exe").unwrap();
        use std::os::fd::AsFd;
        a.send(&ToDaemon::Presented.encode(), &[file.as_fd()])
            .unwrap();
        let (bytes, fds) = b.recv().unwrap().unwrap();
        assert_eq!(ToDaemon::decode(&bytes).unwrap(), ToDaemon::Presented);
        assert_eq!(fds.len(), 1);
        let ino = |fd: BorrowedFd<'_>| {
            // SAFETY: fstat into a zeroed struct on a live fd.
            let mut st: libc::stat = unsafe { std::mem::zeroed() };
            unsafe { libc::fstat(fd.as_raw_fd(), &mut st) };
            st.st_ino
        };
        assert_eq!(ino(fds[0].as_fd()), ino(file.as_fd()));
        drop(a);
        assert_eq!(b.recv().unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }
}
