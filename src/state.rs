//! The server half: every smithay handler this compositor implements, in one
//! place, over one state struct.
//!
//! This module is deliberately the only file that knows the Wayland protocol.
//! Everything above it ([`crate::Glasspane`]) is a facade over
//! [`State`], and everything below it ([`crate::texture`],
//! [`crate::registry`], [`crate::acquire`]) knows nothing about Wayland at
//! all.

use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::input::keyboard::{KeyboardHandle, XkbConfig};
use smithay::input::pointer::PointerHandle;
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::output::Output;
use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode as DecorationMode;
use smithay::reexports::wayland_server::backend::{ClientData, ClientId, DisconnectReason};
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_callback::WlCallback;
use smithay::reexports::wayland_server::protocol::wl_shm;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{Client, DisplayHandle, Resource};
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{
    BufferAssignment, CompositorClientState, CompositorHandler, CompositorState, SurfaceAttributes,
    with_states,
};
use smithay::wayland::dmabuf::{DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier, get_dmabuf};
use smithay::wayland::selection::SelectionHandler;
use smithay::wayland::selection::data_device::{
    DataDeviceHandler, DataDeviceState, ServerDndGrabHandler, set_data_device_focus,
};
use smithay::wayland::shell::xdg::decoration::{XdgDecorationHandler, XdgDecorationState};
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
    XdgToplevelSurfaceData,
};
use smithay::wayland::shm::{ShmHandler, ShmState, with_buffer_contents};

use crate::acquire::AcquireGate;
use crate::registry::{BufferKey, Registry};
use crate::tenant::{Tenant, TenantId};
use crate::{texture, Event};

/// Per-client data. The pid is read once, from the socket's peer credentials,
/// because it is the only thing that ties a tenant back to a
/// [`std::process::Child`] the caller spawned.
#[derive(Default)]
pub struct ClientState {
    pub(crate) compositor_state: CompositorClientState,
    pub(crate) pid: Option<u32>,
}

impl ClientData for ClientState {
    fn initialized(&self, _id: ClientId) {}
    fn disconnected(&self, _id: ClientId, _reason: DisconnectReason) {}
}

/// Stable per-`wl_buffer` cache key, hashed from the protocol object id.
///
/// The id carries wayland-backend's reuse counter, so a destroyed and
/// recreated buffer never aliases the old entry.
pub(crate) fn buffer_key(buffer: &WlBuffer) -> BufferKey {
    let mut h = DefaultHasher::new();
    buffer.id().hash(&mut h);
    BufferKey(h.finish())
}

/// Everything the compositor owns. One struct, because calloop dispatches
/// against exactly one.
pub struct State {
    pub(crate) start: Instant,
    pub(crate) dh: DisplayHandle,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,

    pub(crate) compositor_state: CompositorState,
    pub(crate) xdg_shell_state: XdgShellState,
    /// Held for as long as the display: dropping the state would withdraw
    /// the `zxdg_decoration_manager_v1` global from under live clients.
    #[allow(dead_code)]
    /// Held for as long as the display: dropping the state would withdraw
    /// the `zxdg_decoration_manager_v1` global from under live clients.
    #[allow(dead_code)]
    pub(crate) xdg_decoration_state: XdgDecorationState,
    pub(crate) shm_state: ShmState,
    pub(crate) dmabuf_state: DmabufState,
    pub(crate) dmabuf_global: Option<DmabufGlobal>,
    pub(crate) data_device_state: DataDeviceState,
    pub(crate) seat_state: SeatState<Self>,
    /// The seat itself is driven through `keyboard` and `pointer`; it is held
    /// because dropping it destroys the `wl_seat` global.
    #[allow(dead_code)]
    /// The seat itself is driven through `keyboard` and `pointer`; it is held
    /// because dropping it destroys the `wl_seat` global.
    #[allow(dead_code)]
    pub(crate) seat: Seat<Self>,
    pub(crate) keyboard: KeyboardHandle<Self>,
    #[allow(dead_code)]
    pub(crate) pointer: PointerHandle<Self>,
    #[allow(dead_code)]
    pub(crate) output: Output,

    pub(crate) registry: Registry,
    pub(crate) acquire: Option<AcquireGate>,
    pub(crate) tenants: Vec<Tenant>,
    pub(crate) next_tenant: u64,
    pub(crate) events: VecDeque<Event>,
    pub(crate) pending_callbacks: Vec<WlCallback>,
    pub(crate) default_size: (u32, u32),
    pub(crate) keyboard_focus: Option<TenantId>,
    pub(crate) pointer_focus: Option<TenantId>,
    pub(crate) ticks: u64,
    pub(crate) callbacks_sent: u64,
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("tenants", &self.tenants.len())
            .field("ticks", &self.ticks)
            .field("callbacks_sent", &self.callbacks_sent)
            .finish_non_exhaustive()
    }
}

impl State {
    pub(crate) fn index_of(&self, id: TenantId) -> Option<usize> {
        self.tenants.iter().position(|t| t.id == id)
    }

    pub(crate) fn index_of_surface(&self, surface: &WlSurface) -> Option<usize> {
        self.tenants.iter().position(|t| &t.surface == surface)
    }

    pub(crate) fn emit(&mut self, event: Event) {
        // Bounded so a caller that never drains cannot grow the queue without
        // limit. Dropping the *oldest* is the right end to drop from: the
        // newest events are the ones that still describe the present.
        const CAP: usize = 4096;
        if self.events.len() >= CAP {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    pub(crate) fn reject(&mut self, id: Option<TenantId>, why: String) {
        self.emit(Event::Rejected { id, why });
    }

    pub(crate) fn flush(&self) {
        let _ = self.dh.clone().flush_clients();
    }

    /// Answer every frame callback harvested since the last tick, and say how
    /// many were answered.
    ///
    /// Frame callbacks are the compositor's to *drive* (see the crate docs): a
    /// Wayland client paints only when handed one, so answering them on demand
    /// collapses to "when someone asks" and the tenants freeze.
    pub(crate) fn release_frame_callbacks(&mut self, now_ms: u32) -> usize {
        self.ticks += 1;
        if self.pending_callbacks.is_empty() {
            return 0;
        }
        let mut n = 0;
        for cb in std::mem::take(&mut self.pending_callbacks) {
            cb.done(now_ms);
            self.callbacks_sent += 1;
            n += 1;
        }
        // A `done` is a *volunteered* event: no client request preceded it, so
        // nothing else will drag it out of the buffer. It must be flushed on
        // this line or the tenant never repaints — and then never asks for
        // another callback, which is self-sustaining.
        self.flush();
        n
    }

    fn pid_of(&self, surface: &WlSurface) -> Option<u32> {
        self.dh
            .get_client(surface.id())
            .ok()
            .and_then(|c| c.get_data::<ClientState>().and_then(|d| d.pid))
    }

    /// Re-read the toplevel's title and app id, emitting an event if either
    /// moved. Cheap enough to do per commit, and doing it per commit is what
    /// makes the events arrive without a protocol hook for each one.
    fn refresh_identity(&mut self, index: usize) {
        let surface = self.tenants[index].surface.clone();
        let (title, app_id) = with_states(&surface, |states| {
            states
                .data_map
                .get::<XdgToplevelSurfaceData>()
                .map(|d| {
                    let d = d.lock().unwrap();
                    (d.title.clone(), d.app_id.clone())
                })
                .unwrap_or((None, None))
        });
        let id = self.tenants[index].id;
        if title.is_some() && title != self.tenants[index].title {
            self.tenants[index].title.clone_from(&title);
            self.emit(Event::TitleChanged {
                id,
                title: title.unwrap_or_default(),
            });
        }
        if app_id.is_some() && app_id != self.tenants[index].app_id {
            self.tenants[index].app_id.clone_from(&app_id);
            self.emit(Event::AppIdChanged {
                id,
                app_id: app_id.unwrap_or_default(),
            });
        }
    }

    /// Make sure this tenant's texture is `width` x `height`, recreating it —
    /// and saying so — if it is not.
    fn ensure_texture(&mut self, index: usize, width: u32, height: u32) {
        let id = self.tenants[index].id;
        let fits = self.tenants[index]
            .texture
            .as_ref()
            .is_some_and(|t| t.width() == width && t.height() == height);
        if fits {
            return;
        }
        let tex = texture::create_tenant_texture(&self.device, id, width, height);
        self.tenants[index].texture = Some(Arc::new(tex));
        self.tenants[index].size = (width, height);
        self.emit(Event::Resized {
            id,
            size: (width, height),
        });
    }

    /// One committed buffer becomes texels.
    fn absorb_buffer(&mut self, index: usize, buffer: &WlBuffer) {
        let id = self.tenants[index].id;
        let key = buffer_key(buffer);

        let result = match get_dmabuf(buffer) {
            Ok(dmabuf) => self.absorb_dmabuf(index, key, &dmabuf.clone()),
            Err(_) => self.absorb_shm(index, buffer),
        };

        match result {
            Ok(()) => {
                self.tenants[index].generation += 1;
                let generation = self.tenants[index].generation;
                self.emit(Event::Committed { id, generation });
            }
            Err(why) => self.reject(Some(id), why),
        }
    }

    fn absorb_dmabuf(
        &mut self,
        index: usize,
        key: BufferKey,
        dmabuf: &Dmabuf,
    ) -> Result<(), String> {
        use smithay::backend::allocator::Buffer as _;

        // **The acquire wait goes first**, before anything on this queue can
        // touch the buffer: the next submit then carries it, and the next
        // submit is the blit two dozen lines below.
        if let Some(gate) = self.acquire.as_mut() {
            let plane = dmabuf
                .handles()
                .next()
                .ok_or_else(|| "dmabuf has no plane handle".to_string())?;
            gate.wait_for(key, plane)?;
            self.tenants[index].acquire_staged = true;
        } else {
            self.tenants[index].acquire_staged = false;
        }

        let (width, height) = (dmabuf.width(), dmabuf.height());
        let device = self.device.clone();
        let described = crate::convert::describe(dmabuf)?;
        let (_id, src, _imported) = self
            .registry
            .get_or_import(key, || {
                wgpu_dmabuf::import(&device, &described, texture::IMPORT_USAGE)
            })
            .map_err(|e| format!("dmabuf import failed: {e}"))?;

        self.ensure_texture(index, width, height);
        let dst = self.tenants[index]
            .texture
            .clone()
            .expect("ensure_texture just set it");
        texture::blit(&self.device, &self.queue, &src, &dst, width, height);
        Ok(())
    }

    fn absorb_shm(&mut self, index: usize, buffer: &WlBuffer) -> Result<(), String> {
        let read = with_buffer_contents(buffer, |ptr, len, data| {
            match data.format {
                wl_shm::Format::Argb8888 | wl_shm::Format::Xrgb8888 => {}
                other => return Err(format!("unsupported wl_shm format {other:?}")),
            }
            let (w, h, stride, offset) = (
                data.width.max(0) as u32,
                data.height.max(0) as u32,
                data.stride.max(0) as u32,
                data.offset.max(0) as usize,
            );
            let needed = (h as usize) * (stride as usize);
            if offset.saturating_add(needed) > len {
                return Err(format!(
                    "wl_shm pool is {len} bytes but {h} rows at stride {stride} from offset \
                     {offset} need {}",
                    offset + needed
                ));
            }
            // SAFETY: smithay guarantees ptr/len describe the mapped pool for
            // the duration of this closure, and the bounds above are checked
            // against `len` rather than against the client's word for it.
            let bytes = unsafe { std::slice::from_raw_parts(ptr.add(offset), needed) }.to_vec();
            Ok((w, h, stride, bytes))
        });
        let (w, h, stride, bytes) = match read {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return Err(e),
            Err(e) => return Err(format!("buffer is neither dma-buf nor wl_shm: {e}")),
        };
        self.tenants[index].acquire_staged = false;
        self.ensure_texture(index, w, h);
        let dst = self.tenants[index]
            .texture
            .clone()
            .expect("ensure_texture just set it");
        texture::upload_shm(&self.queue, &dst, &bytes, stride, w, h);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Compositor, shm, dma-buf
// ---------------------------------------------------------------------------

impl CompositorHandler for State {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<ClientState>().unwrap().compositor_state
    }

    fn commit(&mut self, surface: &WlSurface) {
        let (buffer, callbacks) = with_states(surface, |states| {
            let mut guard = states.cached_state.get::<SurfaceAttributes>();
            let attrs = guard.current();
            let buffer = match &attrs.buffer {
                Some(BufferAssignment::NewBuffer(b)) => Some(b.clone()),
                _ => None,
            };
            let callbacks = std::mem::take(&mut attrs.frame_callbacks);
            (buffer, callbacks)
        });

        if let Some(index) = self.index_of_surface(surface) {
            self.refresh_identity(index);
            // The initial configure must be answered before a client is
            // allowed to attach: a toplevel that commits without one is a
            // protocol error on our side, not theirs.
            let toplevel = self.tenants[index].toplevel.clone();
            if !toplevel.is_initial_configure_sent() {
                toplevel.send_configure();
            }
            if let Some(buffer) = buffer {
                self.absorb_buffer(index, &buffer);
            }
        }

        // Harvested, not answered: [`State::release_frame_callbacks`] answers
        // them, on the caller's clock.
        self.pending_callbacks.extend(callbacks);
    }
}

impl BufferHandler for State {
    fn buffer_destroyed(&mut self, buffer: &WlBuffer) {
        let key = buffer_key(buffer);
        self.registry.forget(key);
        if let Some(gate) = self.acquire.as_mut() {
            gate.forget(key);
        }
    }
}

impl ShmHandler for State {
    fn shm_state(&self) -> &ShmState {
        &self.shm_state
    }
}

impl DmabufHandler for State {
    fn dmabuf_state(&mut self) -> &mut DmabufState {
        &mut self.dmabuf_state
    }

    fn dmabuf_imported(
        &mut self,
        _global: &DmabufGlobal,
        dmabuf: Dmabuf,
        notifier: ImportNotifier,
    ) {
        // Validated at `wl_buffer` creation, not at commit: a client that asks
        // for something impossible learns while it can still choose
        // differently. No image is created here — the import itself happens
        // once, later, through the registry.
        match crate::convert::describe(&dmabuf) {
            Ok(_) => {
                let _ = notifier.successful::<Self>();
            }
            Err(why) => {
                self.reject(None, format!("dmabuf create_params rejected: {why}"));
                notifier.failed();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// xdg_shell
// ---------------------------------------------------------------------------

impl XdgShellHandler for State {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell_state
    }

    fn new_toplevel(&mut self, toplevel: ToplevelSurface) {
        self.next_tenant += 1;
        let id = TenantId(self.next_tenant);
        let surface = toplevel.wl_surface().clone();
        let pid = self.pid_of(&surface);
        let size = self.default_size;
        toplevel.with_pending_state(|state| {
            state.size = Some((size.0 as i32, size.1 as i32).into());
        });
        toplevel.send_configure();
        self.tenants.push(Tenant {
            id,
            pid,
            title: None,
            app_id: None,
            size,
            configured: Some(size),
            texture: None,
            generation: 0,
            acquire_staged: false,
            surface,
            toplevel,
        });
        self.emit(Event::Mapped { id, pid });
        self.flush();
    }

    fn toplevel_destroyed(&mut self, toplevel: ToplevelSurface) {
        let Some(index) = self
            .tenants
            .iter()
            .position(|t| t.toplevel.wl_surface() == toplevel.wl_surface())
        else {
            return;
        };
        let gone = self.tenants.remove(index);
        if self.keyboard_focus == Some(gone.id) {
            self.keyboard_focus = None;
            let serial = smithay::utils::SERIAL_COUNTER.next_serial();
            self.keyboard.clone().set_focus(self, None, serial);
        }
        if self.pointer_focus == Some(gone.id) {
            self.pointer_focus = None;
        }
        self.emit(Event::Unmapped { id: gone.id });
    }

    // Popups are out of scope: this crate hands the shell one texture per
    // toplevel, and a popup is a second surface with a position relative to
    // its parent — i.e. placement policy, which belongs to the shell. A popup
    // request is answered with a configure so the client does not hang, and
    // nothing else.
    fn new_popup(&mut self, surface: PopupSurface, _positioner: PositionerState) {
        let _ = surface.send_configure();
    }

    fn grab(
        &mut self,
        _surface: PopupSurface,
        _seat: smithay::reexports::wayland_server::protocol::wl_seat::WlSeat,
        _serial: smithay::utils::Serial,
    ) {
    }

    fn reposition_request(
        &mut self,
        surface: PopupSurface,
        _positioner: PositionerState,
        token: u32,
    ) {
        surface.send_repositioned(token);
    }
}

/// Bare glass: this compositor owns decorations and draws none.
///
/// All three moments of the protocol — the client created a decoration object,
/// stated a preference, withdrew it — are answered identically. A compositor
/// that answers them differently is one whose tenants' buffer sizes depend on
/// what their toolkit happened to ask for.
impl XdgDecorationHandler for State {
    fn new_decoration(&mut self, toplevel: ToplevelSurface) {
        announce_server_side(&toplevel);
    }

    fn request_mode(&mut self, toplevel: ToplevelSurface, _mode: DecorationMode) {
        announce_server_side(&toplevel);
    }

    fn unset_mode(&mut self, toplevel: ToplevelSurface) {
        announce_server_side(&toplevel);
    }
}

fn announce_server_side(toplevel: &ToplevelSurface) {
    toplevel.with_pending_state(|state| {
        state.decoration_mode = Some(DecorationMode::ServerSide);
    });
    if toplevel.is_initial_configure_sent() {
        toplevel.send_pending_configure();
    }
}

// ---------------------------------------------------------------------------
// Seat and selection
// ---------------------------------------------------------------------------

impl SeatHandler for State {
    type KeyboardFocus = WlSurface;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }

    fn focus_changed(&mut self, seat: &Seat<Self>, focused: Option<&WlSurface>) {
        let dh = self.dh.clone();
        let client = focused.and_then(|s| dh.get_client(s.id()).ok());
        set_data_device_focus(&dh, seat, client);
    }

    /// What the focused client wants the pointer to look like.
    ///
    /// glasspane draws nothing, so this is dropped. A shell that wants to
    /// honour a client cursor reads it from the seat itself; wrapping it here
    /// would be scene policy.
    fn cursor_image(&mut self, _seat: &Seat<Self>, _image: smithay::input::pointer::CursorImageStatus) {
    }
}

impl SelectionHandler for State {
    type SelectionUserData = ();
}

impl DataDeviceHandler for State {
    fn data_device_state(&self) -> &DataDeviceState {
        &self.data_device_state
    }
}

impl smithay::wayland::selection::data_device::ClientDndGrabHandler for State {}
impl ServerDndGrabHandler for State {}

impl smithay::wayland::output::OutputHandler for State {}

smithay::delegate_compositor!(State);
smithay::delegate_shm!(State);
smithay::delegate_dmabuf!(State);
smithay::delegate_xdg_shell!(State);
smithay::delegate_xdg_decoration!(State);
smithay::delegate_seat!(State);
smithay::delegate_data_device!(State);
smithay::delegate_output!(State);

/// A seat and the two handles this crate drives it through.
type SeatParts = (Seat<State>, KeyboardHandle<State>, PointerHandle<State>);

impl State {
    /// Build the seat with both capabilities and hand back its handles.
    pub(crate) fn make_seat(
        seat_state: &mut SeatState<Self>,
        dh: &DisplayHandle,
        name: &str,
        xkb: XkbConfig<'_>,
    ) -> Result<SeatParts, crate::Error> {
        let mut seat = seat_state.new_wl_seat(dh, name);
        let keyboard = seat
            .add_keyboard(xkb, 200, 25)
            .map_err(|e| crate::Error::Socket(format!("seat keyboard: {e}")))?;
        let pointer = seat.add_pointer();
        Ok((seat, keyboard, pointer))
    }
}
