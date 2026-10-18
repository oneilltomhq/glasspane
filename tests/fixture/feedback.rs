//! What a client is told about dma-buf, read the two ways clients read it.
//!
//! A v3 client gets `modifier` events straight off the bind. A v4 client
//! asks for the default feedback and gets a format table, a main device and
//! tranches instead; the main device is how Mesa's EGL picks its GPU. Like the
//! painting fixture, this is a real client on a real socket, in a thread.

use std::collections::BTreeSet;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::mpsc::Sender;

use wayland_client::protocol::wl_registry;
use wayland_client::{Connection, Dispatch, QueueHandle, WEnum};
use wayland_protocols::wp::linux_dmabuf::zv1::client::{
    zwp_linux_dmabuf_feedback_v1 as feedback, zwp_linux_dmabuf_v1,
};

/// A `(fourcc, modifier)` pair.
pub type Pair = (u32, u64);

/// Everything one bind was told.
#[derive(Debug, Default)]
pub struct Seen {
    /// The version actually bound.
    pub version: u32,
    /// v3 and below: the `modifier` events.
    pub modifiers: BTreeSet<Pair>,
    /// v4 and up: the default feedback's `main_device`.
    pub main_device: Option<u64>,
    /// v4 and up: the format table, in index order.
    pub table: Vec<Pair>,
    /// v4 and up: the tranches, in the order they were done.
    pub tranches: Vec<Tranche>,
}

/// One feedback tranche, its indices already looked up in the table.
#[derive(Debug, Default, Clone)]
pub struct Tranche {
    pub target_device: Option<u64>,
    pub flags: u32,
    pub formats: BTreeSet<Pair>,
}

struct Reader {
    want: u32,
    seen: Seen,
    dmabuf: Option<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1>,
    pending: Tranche,
    indices: Vec<u16>,
    done: bool,
}

/// Bind `zwp_linux_dmabuf_v1` at `version` (or what is offered, if less),
/// read what it says, and send it down `tx`.
pub fn read(socket_name: std::ffi::OsString, version: u32, tx: Sender<Result<Seen, String>>) {
    std::thread::spawn(move || {
        let _ = tx.send(run(&socket_name, version));
    });
}

fn run(socket_name: &std::ffi::OsStr, version: u32) -> Result<Seen, String> {
    let path = super::socket_path(socket_name);
    let stream =
        UnixStream::connect(&path).map_err(|e| format!("connect {}: {e}", path.display()))?;
    let conn = Connection::from_socket(stream).map_err(|e| format!("wayland connection: {e}"))?;
    let mut queue = conn.new_event_queue();
    let qh = queue.handle();
    conn.display().get_registry(&qh, ());
    let mut reader = Reader {
        want: version,
        seen: Seen::default(),
        dmabuf: None,
        pending: Tranche::default(),
        indices: Vec::new(),
        done: false,
    };
    // The first roundtrip binds; the second carries a v3 bind's modifiers.
    queue
        .roundtrip(&mut reader)
        .map_err(|e| format!("roundtrip: {e}"))?;
    let dmabuf = reader.dmabuf.clone().ok_or("no zwp_linux_dmabuf_v1")?;
    if reader.seen.version >= 4 {
        dmabuf.get_default_feedback(&qh, ());
    } else {
        reader.done = true;
    }
    for _ in 0..10 {
        queue
            .roundtrip(&mut reader)
            .map_err(|e| format!("roundtrip: {e}"))?;
        if reader.done {
            return Ok(reader.seen);
        }
    }
    Err(format!("the feedback never said done: {:?}", reader.seen))
}

/// The format table: `size` bytes of `{ u32 format, u32 pad, u64 modifier }`.
fn read_table(fd: &OwnedFd, size: u32) -> Vec<Pair> {
    let len = size as usize;
    // SAFETY: a read-only private mapping of exactly the size the server
    // gave, unmapped before return.
    unsafe {
        let map = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            fd.as_raw_fd(),
            0,
        );
        assert_ne!(map, libc::MAP_FAILED, "mmap the format table");
        let bytes = std::slice::from_raw_parts(map.cast::<u8>(), len);
        let table = bytes
            .chunks_exact(16)
            .map(|e| {
                (
                    u32::from_ne_bytes(e[0..4].try_into().unwrap()),
                    u64::from_ne_bytes(e[8..16].try_into().unwrap()),
                )
            })
            .collect();
        libc::munmap(map, len);
        table
    }
}

fn dev(bytes: &[u8]) -> u64 {
    u64::from_ne_bytes(bytes.try_into().expect("a dev_t is 8 bytes"))
}

impl Dispatch<wl_registry::WlRegistry, ()> for Reader {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            if interface == "zwp_linux_dmabuf_v1" {
                let v = version.min(state.want);
                state.seen.version = v;
                state.dmabuf = Some(registry.bind(name, v, qh, ()));
            }
        }
    }
}

impl Dispatch<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1, ()> for Reader {
    fn event(
        state: &mut Self,
        _: &zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1,
        event: zwp_linux_dmabuf_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwp_linux_dmabuf_v1::Event::Modifier {
            format,
            modifier_hi,
            modifier_lo,
        } = event
        {
            let modifier = (u64::from(modifier_hi) << 32) | u64::from(modifier_lo);
            state.seen.modifiers.insert((format, modifier));
        }
    }
}

impl Dispatch<feedback::ZwpLinuxDmabufFeedbackV1, ()> for Reader {
    fn event(
        state: &mut Self,
        _: &feedback::ZwpLinuxDmabufFeedbackV1,
        event: feedback::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            feedback::Event::FormatTable { fd, size } => {
                state.seen.table = read_table(&fd, size);
            }
            feedback::Event::MainDevice { device } => {
                state.seen.main_device = Some(dev(&device));
            }
            feedback::Event::TrancheTargetDevice { device } => {
                state.pending.target_device = Some(dev(&device));
            }
            feedback::Event::TrancheFlags { flags } => {
                state.pending.flags = match flags {
                    WEnum::Value(f) => f.bits(),
                    WEnum::Unknown(u) => u,
                };
            }
            feedback::Event::TrancheFormats { indices } => {
                state.indices.extend(
                    indices
                        .chunks_exact(2)
                        .map(|i| u16::from_ne_bytes([i[0], i[1]])),
                );
            }
            feedback::Event::TrancheDone => {
                let mut tranche = std::mem::take(&mut state.pending);
                for i in state.indices.drain(..) {
                    tranche.formats.insert(state.seen.table[i as usize]);
                }
                state.seen.tranches.push(tranche);
            }
            feedback::Event::Done => state.done = true,
            _ => {}
        }
    }
}
