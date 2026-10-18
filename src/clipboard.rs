//! The desktop's clipboard, bridged to the tenants', for a shell that is a
//! window on a Wayland desktop.
//!
//! The tenants share one clipboard ([`crate::state`] relays `wl_data_device`
//! between them), and without this it ends at the edge of the socket: a
//! password copied on the desktop pastes nothing into a pane, and nothing
//! copied in a pane reaches the desktop. glasspane has no window and no
//! connection to a desktop of its own, so the bridge is handed one
//! ([`crate::Glasspane::bridge_clipboard`]): the `wl_display` of the shell's
//! window, and the bridge is a second user of that same client connection.
//!
//! That is the whole choice of mechanism. A desktop offers its clipboard only
//! to the client that has its keyboard, and takes a new selection only from
//! that client, with the serial of one of its input events. The shell's window
//! is that client, and a connection of glasspane's own never would be. What
//! lets a client without focus read and set the clipboard is the data-control
//! protocol (`ext_data_control_v1`, `zwlr_data_control_manager_v1`), which
//! `wl-clipboard-rs`, and `arboard` on Wayland, are built on; wlroots and KDE
//! offer it and GNOME does not, where `arboard` falls back to X11 through
//! Xwayland. Riding the window's connection works with core
//! `wl_data_device_manager` on any desktop, and adds no crate the shell's tree
//! does not already hold: `wayland-client` is the one winit speaks through,
//! and its system backend (libwayland, which winit loads too) is what lets
//! two libraries share one `wl_display` at all, each on its own event queue.
//!
//! Both directions are lazy, and no byte of the clipboard passes through this
//! process:
//!
//! * The desktop's selection, when it is offered to the window (on focus-in,
//!   and whenever it changes while the window has focus), becomes the
//!   tenants' selection with the same mime types. A tenant's `receive` is
//!   forwarded to the desktop's offer with the tenant's own pipe, so the
//!   desktop app that copied writes straight into the tenant.
//! * A tenant's selection is offered on the desktop with the same mime types,
//!   and the desktop's `send` is forwarded to the tenant's source with the
//!   desktop's pipe.
//!
//! A tenant's copy made while the window does not have the keyboard waits
//! for it. The desktop would refuse it now: a serial from before the window's
//! last leave is stale. So it is held, and offered on the window's next enter,
//! unless a later tenant copy or clear replaces it first. Held, it also beats
//! the desktop's own selection, which a desktop sends just before that enter
//! and which predates the copy.
//!
//! A selection of ours comes back: the desktop offers it to the focused
//! client, which is the window. It is known by a marker mime type and
//! ignored; taken, a tenant's copy would replace itself with a desktop offer
//! of itself.
//!
//! Only the clipboard. The primary selection (select, middle-click) stays
//! between tenants.

use std::ffi::c_void;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::Mutex;

use wayland_client::backend::Backend;
use wayland_client::protocol::wl_data_device::{self, WlDataDevice};
use wayland_client::protocol::wl_data_device_manager::WlDataDeviceManager;
use wayland_client::protocol::wl_data_offer::{self, WlDataOffer};
use wayland_client::protocol::wl_data_source::{self, WlDataSource};
use wayland_client::protocol::wl_keyboard::{self, WlKeyboard};
use wayland_client::protocol::wl_registry::{self, WlRegistry};
use wayland_client::protocol::wl_seat::{self, WlSeat};
use wayland_client::{
    event_created_child, Connection, Dispatch, EventQueue, Proxy, QueueHandle, WEnum,
};

/// The desktop's offer of its selection, which a tenant's paste is read from.
///
/// What [`crate::state::State`] attaches to a selection it sets itself, and
/// it sets one only for the desktop.
#[derive(Debug, Clone)]
pub struct Offer(WlDataOffer);

/// What the desktop said that the tenants' side has to act on.
///
/// The seam between the desktop's dispatch, which has no seat, and the
/// tenants' seat, which [`crate::state::State`] holds.
pub(crate) enum Change {
    /// The desktop's clipboard is now this offer, with these mime types.
    Offered(Offer, Vec<String>),
    /// The desktop's clipboard is empty.
    Emptied,
    /// Something on the desktop is pasting the tenants' selection: write it
    /// as this mime type into this pipe.
    Wanted(String, OwnedFd),
}

/// The bridge: a queue of its own on the window's connection.
pub(crate) struct Bridge {
    conn: Connection,
    queue: EventQueue<Desk>,
    desk: Desk,
}

impl std::fmt::Debug for Bridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bridge")
            .field("serial", &self.desk.serial)
            .field("pending", &self.desk.pending)
            .field("offer", &self.desk.offer.is_some())
            .field("source", &self.desk.source.is_some())
            .finish_non_exhaustive()
    }
}

/// The objects the bridge holds on the desktop, and what it has heard.
struct Desk {
    qh: QueueHandle<Desk>,
    /// Bound as the registry names them, and the device made once both are.
    seat: Option<WlSeat>,
    manager: Option<WlDataDeviceManager>,
    device: Option<WlDataDevice>,
    /// Made once the seat says it has a keyboard. It is only ever read for
    /// serials: a desktop takes a selection with the serial of the window's
    /// latest input, and the window's keyboard objects are winit's.
    keyboard: Option<WlKeyboard>,
    /// The serial of the window's latest enter, key or modifiers, and `None`
    /// from its leave: a serial from before a leave is stale, and a desktop
    /// refuses a selection set with it.
    serial: Option<u32>,
    /// A tenant's copy made with no serial to offer it with: its mime types,
    /// offered on the window's next enter. A later tenant copy or clear
    /// replaces it, and while it waits the desktop's selection is not taken.
    pending: Option<Vec<String>>,
    /// The desktop's current selection as last offered, destroyed when the
    /// next one comes, as the protocol asks.
    offer: Option<WlDataOffer>,
    /// The tenants' selection as offered on the desktop.
    source: Option<WlDataSource>,
    /// The mime type every source of ours carries, so the desktop offering
    /// it back is recognised. Per process: a second shell's copy is a copy
    /// like any other.
    marker: String,
    changes: Vec<Change>,
}

impl Bridge {
    /// Ask the desktop for its globals on `display`.
    ///
    /// The request is only queued, and goes out on the first [`Bridge::pump`].
    /// Nothing waits for the answer: the seat and the data device manager are
    /// bound when the registry names them, in a later pump. A roundtrip here
    /// would block the window's thread on the desktop, and whoever is the
    /// desktop may be served from that same thread.
    ///
    /// # Safety
    ///
    /// `display` is a live `wl_display`, and outlives the bridge.
    pub(crate) unsafe fn new(display: NonNull<c_void>) -> Self {
        // SAFETY: the caller's.
        let backend = unsafe { Backend::from_foreign_display(display.as_ptr().cast()) };
        let conn = Connection::from_backend(backend);
        let queue = conn.new_event_queue();
        let qh = queue.handle();
        conn.display().get_registry(&qh, ());
        let desk = Desk {
            qh,
            seat: None,
            manager: None,
            device: None,
            keyboard: None,
            serial: None,
            pending: None,
            offer: None,
            source: None,
            marker: format!("application/x-glasspane-{}", std::process::id()),
            changes: Vec::new(),
        };
        Bridge { conn, queue, desk }
    }

    /// Write what the tenants' side has asked of the desktop, and take in
    /// what the desktop has said since the last call.
    ///
    /// The one place the bridge writes the window's connection (bar its
    /// drop). [`Bridge::offer`], [`Bridge::withdraw`], [`Bridge::receive`]
    /// and the dispatch here only queue requests, and the flushes before the
    /// read and after the dispatch send them, in order.
    ///
    /// The socket is read here too, without blocking, when it has something.
    /// The window's own loop reads it as well, and libwayland sorts each
    /// event onto its own queue whichever of them reads, so the bridge does
    /// not wait on the window's loop, nor the window's loop on the bridge.
    pub(crate) fn pump(&mut self) -> Vec<Change> {
        let _ = self.conn.flush();
        if let Some(guard) = self.queue.prepare_read() {
            let mut pfd = libc::pollfd {
                fd: self.conn.as_fd().as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: one initialised pollfd, and a zero timeout.
            if unsafe { libc::poll(&mut pfd, 1, 0) } > 0 {
                let _ = guard.read();
            }
        }
        let _ = self.queue.dispatch_pending(&mut self.desk);
        let _ = self.conn.flush();
        std::mem::take(&mut self.desk.changes)
    }

    /// A tenant copied these mime types: offer them on the desktop, now if
    /// the window has the keyboard, or on its next enter.
    pub(crate) fn offer(&mut self, mime_types: Vec<String>) {
        self.desk.offer(mime_types);
    }

    /// A tenant cleared its clipboard: withdraw ours from the desktop, and
    /// forget a copy still waiting for the keyboard.
    ///
    /// Only ours: a tenant clearing its clipboard is not a reason to empty
    /// the desktop's. Without a serial ours stays offered, and a paste of it
    /// finds the tenants' clipboard empty.
    pub(crate) fn withdraw(&mut self) {
        let desk = &mut self.desk;
        desk.pending = None;
        let (Some(serial), Some(device)) = (desk.serial, &desk.device) else {
            return;
        };
        if let Some(old) = desk.source.take() {
            device.set_selection(None, serial);
            old.destroy();
        }
    }

    /// A tenant is pasting the desktop's selection: hand its pipe to the
    /// desktop's offer, which hands it to whoever copied.
    pub(crate) fn receive(&self, offer: &Offer, mime_type: String, fd: OwnedFd) {
        // libwayland duplicates the fd as the request is queued, so the
        // tenant's copy closes here and the desktop's when it is done.
        offer.0.receive(mime_type, fd.as_fd());
    }
}

impl Drop for Bridge {
    /// Everything on the desktop is given back before the queue goes: the
    /// connection is the window's and outlives the bridge. The one flush
    /// outside [`Bridge::pump`], since no pump follows.
    fn drop(&mut self) {
        let desk = &mut self.desk;
        if let Some(source) = desk.source.take() {
            source.destroy();
        }
        if let Some(offer) = desk.offer.take() {
            offer.destroy();
        }
        if let Some(device) = desk.device.take() {
            if device.version() >= 2 {
                device.release();
            }
        }
        if let Some(keyboard) = desk.keyboard.take() {
            if keyboard.version() >= 3 {
                keyboard.release();
            }
        }
        if let Some(seat) = desk.seat.take() {
            if seat.version() >= 5 {
                seat.release();
            }
        }
        let _ = self.conn.flush();
    }
}

impl Desk {
    /// Offer a tenant's copy on the desktop as the window's selection, or,
    /// with no serial to set it with, hold it for the next enter.
    fn offer(&mut self, mime_types: Vec<String>) {
        let (Some(serial), Some(device), Some(manager)) =
            (self.serial, &self.device, &self.manager)
        else {
            self.pending = Some(mime_types);
            return;
        };
        self.pending = None;
        let source = manager.create_data_source(&self.qh, ());
        for mime_type in mime_types {
            source.offer(mime_type);
        }
        source.offer(self.marker.clone());
        device.set_selection(Some(&source), serial);
        if let Some(old) = self.source.replace(source) {
            old.destroy();
        }
    }

    /// The desktop's selection changed, or was offered again on focus-in.
    ///
    /// Not taken while a tenant's copy is pending: the desktop sends its
    /// selection just before the enter that offers the copy, and the copy is
    /// newer. A desktop that sends it after the enter instead (smithay does)
    /// is not covered: by then the copy has been offered and is no longer
    /// pending, and a selection the desktop held is taken over it.
    fn selected(&mut self, offer: Option<WlDataOffer>) {
        if let Some(old) = self.offer.take() {
            old.destroy();
        }
        if self.pending.is_some() {
            if let Some(offer) = offer {
                offer.destroy();
            }
            return;
        }
        let Some(offer) = offer else {
            self.changes.push(Change::Emptied);
            return;
        };
        let mime_types = offer
            .data::<Mutex<Vec<String>>>()
            .map(|m| m.lock().unwrap().clone())
            .unwrap_or_default();
        if mime_types.contains(&self.marker) {
            offer.destroy();
            return;
        }
        self.offer = Some(offer.clone());
        self.changes.push(Change::Offered(Offer(offer), mime_types));
    }
}

impl Dispatch<WlRegistry, ()> for Desk {
    /// The first seat and the data device manager are bound, once; a global
    /// coming or going later changes nothing.
    fn event(
        desk: &mut Self,
        registry: &WlRegistry,
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
        if interface == WlSeat::interface().name && desk.seat.is_none() {
            desk.seat = Some(registry.bind(name, version.min(5), qh, ()));
        } else if interface == WlDataDeviceManager::interface().name && desk.manager.is_none() {
            desk.manager = Some(registry.bind(name, version.min(3), qh, ()));
        }
        if let (Some(seat), Some(manager), None) = (&desk.seat, &desk.manager, &desk.device) {
            desk.device = Some(manager.get_data_device(seat, qh, ()));
        }
    }
}

impl Dispatch<WlSeat, ()> for Desk {
    fn event(
        desk: &mut Self,
        seat: &WlSeat,
        event: wl_seat::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities {
            capabilities: WEnum::Value(caps),
        } = event
        {
            if caps.contains(wl_seat::Capability::Keyboard) && desk.keyboard.is_none() {
                desk.keyboard = Some(seat.get_keyboard(qh, ()));
            }
        }
    }
}

impl Dispatch<WlKeyboard, ()> for Desk {
    fn event(
        desk: &mut Self,
        _: &WlKeyboard,
        event: wl_keyboard::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_keyboard::Event::Enter { serial, .. } => {
                desk.serial = Some(serial);
                if let Some(mime_types) = desk.pending.take() {
                    desk.offer(mime_types);
                }
            }
            wl_keyboard::Event::Key { serial, .. }
            | wl_keyboard::Event::Modifiers { serial, .. } => desk.serial = Some(serial),
            wl_keyboard::Event::Leave { .. } => desk.serial = None,
            // The keymap's fd is dropped, and closed, with the event.
            _ => {}
        }
    }
}

impl Dispatch<WlDataDeviceManager, ()> for Desk {
    fn event(
        _: &mut Self,
        _: &WlDataDeviceManager,
        _: <WlDataDeviceManager as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlDataDevice, ()> for Desk {
    fn event(
        desk: &mut Self,
        _: &WlDataDevice,
        event: wl_data_device::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_data_device::Event::Selection { id } => desk.selected(id),
            // A drag over the window: not taken, and not left lying about.
            wl_data_device::Event::Enter {
                id: Some(offer), ..
            } => offer.destroy(),
            _ => {}
        }
    }

    event_created_child!(Desk, WlDataDevice, [
        wl_data_device::EVT_DATA_OFFER_OPCODE => (WlDataOffer, Mutex::new(Vec::<String>::new())),
    ]);
}

impl Dispatch<WlDataOffer, Mutex<Vec<String>>> for Desk {
    /// The mime types arrive one event each, between the offer and the
    /// selection that names it.
    fn event(
        _: &mut Self,
        _: &WlDataOffer,
        event: wl_data_offer::Event,
        mime_types: &Mutex<Vec<String>>,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_data_offer::Event::Offer { mime_type } = event {
            mime_types.lock().unwrap().push(mime_type);
        }
    }
}

impl Dispatch<WlDataSource, ()> for Desk {
    fn event(
        desk: &mut Self,
        source: &WlDataSource,
        event: wl_data_source::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            // The marker has no contents; its pipe closes empty.
            wl_data_source::Event::Send { mime_type, fd } if mime_type != desk.marker => {
                desk.changes.push(Change::Wanted(mime_type, fd));
            }
            // Something else on the desktop copied.
            wl_data_source::Event::Cancelled => {
                if desk.source.as_ref() == Some(source) {
                    desk.source = None;
                }
                source.destroy();
            }
            _ => {}
        }
    }
}
