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
    PopupSurface, PositionerState, SurfaceCachedState, ToplevelSurface, XdgShellHandler,
    XdgShellState, XdgToplevelSurfaceData,
};
use smithay::wayland::shm::{ShmHandler, ShmState, with_buffer_contents};

use crate::acquire::AcquireGate;
use crate::input::Target;
use crate::popup::{Parent, Popup, PopupId};
use crate::registry::{BufferKey, Registry};
use crate::tenant::{Tenant, TenantId};
use crate::texture::Skin;
use crate::{texture, Event};

/// Which of the two lists a commit, a texture or an input event is about.
///
/// A toplevel and a popup are different things to the shell and the same thing
/// to the buffer path, so the buffer path is written once against this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Slot {
    /// An index into [`State::tenants`].
    Tenant(usize),
    /// An index into [`State::popups`].
    Popup(usize),
}

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
    pub(crate) popups: Vec<Popup>,
    pub(crate) next_popup: u64,
    /// The popup holding an explicit grab, if any: while it is up, a button
    /// pressed outside its chain dismisses it instead of reaching what is
    /// under the cursor.
    pub(crate) grab: Option<PopupId>,
    /// Where the keyboard was before the grab took it, to put it back.
    pub(crate) focus_before_grab: Option<Target>,
    pub(crate) events: VecDeque<Event>,
    pub(crate) pending_callbacks: Vec<WlCallback>,
    pub(crate) default_size: (u32, u32),
    pub(crate) keyboard_focus: Option<Target>,
    pub(crate) pointer_focus: Option<Target>,
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
    /// Say which tenant is the active window, and tell the others they are
    /// not.
    ///
    /// `xdg_toplevel.configure` carries an `activated` state, and a toolkit
    /// that is not told it has it behaves like a window in the background:
    /// Chromium in particular will create a menu's `xdg_popup` and then never
    /// lay the menu out, so the popup maps at 1x1 and the human sees nothing.
    /// Keyboard focus is the one signal this crate has for "active", so the
    /// two are kept together.
    pub(crate) fn set_active(&mut self, active: Option<TenantId>) {
        use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;
        for tenant in &self.tenants {
            let want = Some(tenant.id) == active;
            let toplevel = tenant.toplevel.clone();
            let has = toplevel.with_pending_state(|state| {
                let has = state.states.contains(xdg_toplevel::State::Activated);
                if want && !has {
                    state.states.set(xdg_toplevel::State::Activated);
                } else if !want && has {
                    state.states.unset(xdg_toplevel::State::Activated);
                }
                has
            });
            if want != has && toplevel.is_initial_configure_sent() {
                toplevel.send_pending_configure();
            }
        }
    }

    pub(crate) fn index_of(&self, id: TenantId) -> Option<usize> {
        self.tenants.iter().position(|t| t.id == id)
    }

    pub(crate) fn index_of_surface(&self, surface: &WlSurface) -> Option<usize> {
        self.tenants.iter().position(|t| &t.surface == surface)
    }

    pub(crate) fn popup(&self, id: PopupId) -> Option<&Popup> {
        self.popups.iter().find(|p| p.id == id)
    }

    pub(crate) fn index_of_popup(&self, id: PopupId) -> Option<usize> {
        self.popups.iter().position(|p| p.id == id)
    }

    pub(crate) fn index_of_popup_surface(&self, surface: &WlSurface) -> Option<usize> {
        self.popups.iter().position(|p| &p.surface == surface)
    }

    /// Which list this surface is in, if either.
    fn slot_of_surface(&self, surface: &WlSurface) -> Option<Slot> {
        self.index_of_surface(surface)
            .map(Slot::Tenant)
            .or_else(|| self.index_of_popup_surface(surface).map(Slot::Popup))
    }

    pub(crate) fn slot_of_target(&self, target: Target) -> Option<Slot> {
        match target {
            Target::Tenant(id) => self.index_of(id).map(Slot::Tenant),
            Target::Popup(id) => self.index_of_popup(id).map(Slot::Popup),
        }
    }

    pub(crate) fn surface_of(&self, slot: Slot) -> WlSurface {
        match slot {
            Slot::Tenant(i) => self.tenants[i].surface.clone(),
            Slot::Popup(i) => self.popups[i].surface.clone(),
        }
    }

    fn skin_mut(&mut self, slot: Slot) -> &mut Skin {
        match slot {
            Slot::Tenant(i) => &mut self.tenants[i].skin,
            Slot::Popup(i) => &mut self.popups[i].skin,
        }
    }

    fn skin(&self, slot: Slot) -> &Skin {
        match slot {
            Slot::Tenant(i) => &self.tenants[i].skin,
            Slot::Popup(i) => &self.popups[i].skin,
        }
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
    fn ensure_texture(&mut self, slot: Slot, width: u32, height: u32) {
        let fits = self
            .skin(slot)
            .texture
            .as_ref()
            .is_some_and(|t| t.width() == width && t.height() == height);
        if fits {
            return;
        }
        let label = match slot {
            Slot::Tenant(i) => format!("tenant#{}", self.tenants[i].id.0),
            Slot::Popup(i) => format!("popup#{}", self.popups[i].id.0),
        };
        let tex = texture::create_surface_texture(&self.device, &label, width, height);
        let skin = self.skin_mut(slot);
        skin.texture = Some(Arc::new(tex));
        skin.size = (width, height);
        let event = match slot {
            Slot::Tenant(i) => Event::Resized {
                id: self.tenants[i].id,
                size: (width, height),
            },
            Slot::Popup(i) => Event::PopupResized {
                id: self.popups[i].id,
                size: (width, height),
            },
        };
        self.emit(event);
    }

    /// One committed buffer becomes texels.
    fn absorb_buffer(&mut self, slot: Slot, buffer: &WlBuffer) {
        let key = buffer_key(buffer);

        let result = match get_dmabuf(buffer) {
            Ok(dmabuf) => self.absorb_dmabuf(slot, key, &dmabuf.clone()),
            Err(_) => self.absorb_shm(slot, buffer),
        };

        match result {
            Ok(()) => {
                self.skin_mut(slot).generation += 1;
                let generation = self.skin(slot).generation;
                let event = match slot {
                    Slot::Tenant(i) => Event::Committed {
                        id: self.tenants[i].id,
                        generation,
                    },
                    Slot::Popup(i) => Event::PopupCommitted {
                        id: self.popups[i].id,
                        generation,
                    },
                };
                self.emit(event);
            }
            Err(why) => {
                let id = match slot {
                    Slot::Tenant(i) => Some(self.tenants[i].id),
                    // A popup's pixels belong to its root tenant as far as a
                    // shell watching for trouble is concerned.
                    Slot::Popup(i) => Some(self.popups[i].root),
                };
                self.reject(id, why);
            }
        }
    }

    fn absorb_dmabuf(
        &mut self,
        slot: Slot,
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
            self.skin_mut(slot).acquire_staged = true;
        } else {
            self.skin_mut(slot).acquire_staged = false;
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

        self.ensure_texture(slot, width, height);
        let dst = self
            .skin(slot)
            .texture
            .clone()
            .expect("ensure_texture just set it");
        texture::blit(&self.device, &self.queue, &src, &dst, width, height);
        Ok(())
    }

    fn absorb_shm(&mut self, slot: Slot, buffer: &WlBuffer) -> Result<(), String> {
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
        self.skin_mut(slot).acquire_staged = false;
        self.ensure_texture(slot, w, h);
        let dst = self
            .skin(slot)
            .texture
            .clone()
            .expect("ensure_texture just set it");
        texture::upload_shm(&self.queue, &dst, &bytes, stride, w, h);
        Ok(())
    }
}

/// The positioner's answer, kept inside the root tenant's rectangle.
///
/// Split out from [`State::place`] so the arithmetic — which is the part that
/// is wrong by a window-geometry offset or a sign if it is wrong at all — can
/// be tested without a GPU, a socket or a client.
fn place_in(
    positioner: &PositionerState,
    base: (i32, i32),
    root_size: (u32, u32),
) -> smithay::utils::Rectangle<i32, smithay::utils::Logical> {
    use smithay::utils::{Point, Rectangle, Size};
    if root_size.0 == 0 || root_size.1 == 0 {
        // Nothing to constrain against: a tenant that has not painted yet has
        // no rectangle, and inventing one would place every popup of a
        // slow-starting client against a size it never had.
        return positioner.get_geometry();
    }
    // The visible rectangle is the tenant's, expressed in the same space as
    // the positioner's answer: relative to the parent's window geometry, which
    // sits at `base` in the tenant.
    let target = Rectangle::new(
        Point::from((-base.0, -base.1)),
        Size::from((root_size.0 as i32, root_size.1 as i32)),
    );
    (*positioner).get_unconstrained_geometry(target)
}

/// A surface's `xdg_surface.set_window_geometry` offset, in surface-local
/// pixels. Zero for a client that has not set one, which is most of them.
fn popup_geometry_loc(surface: &WlSurface) -> (i32, i32) {
    with_states(surface, |states| {
        states
            .cached_state
            .get::<SurfaceCachedState>()
            .current()
            .geometry
            .map(|g| (g.loc.x, g.loc.y))
            .unwrap_or((0, 0))
    })
}

// ---------------------------------------------------------------------------
// Popups
// ---------------------------------------------------------------------------

impl State {
    /// Resolve `parent_surface` into the popup's parent, its root tenant, and
    /// the point its positioner is resolved against (the parent's window
    /// geometry origin, in the root tenant's buffer pixels).
    fn parent_of(&self, parent_surface: &WlSurface) -> Option<(Parent, TenantId, (i32, i32))> {
        if let Some(i) = self.index_of_surface(parent_surface) {
            let id = self.tenants[i].id;
            return Some((
                Parent::Tenant(id),
                id,
                popup_geometry_loc(parent_surface),
            ));
        }
        let i = self.index_of_popup_surface(parent_surface)?;
        let p = &self.popups[i];
        Some((Parent::Popup(p.id), p.root, p.anchor))
    }

    /// The same point for a parent already known by identity.
    fn base_of_parent(&self, parent: Parent) -> (i32, i32) {
        match parent {
            Parent::Tenant(id) => match self.index_of(id) {
                Some(i) => popup_geometry_loc(&self.tenants[i].surface),
                None => (0, 0),
            },
            Parent::Popup(id) => match self.index_of_popup(id) {
                Some(i) => self.popups[i].anchor,
                None => (0, 0),
            },
        }
    }

    /// The positioner, resolved.
    ///
    /// The rectangle a popup is kept inside is its **root tenant's own
    /// surface**, not an output: a pane in a scene has no output, and the
    /// tenant's rectangle is the only thing the client and this compositor
    /// both know the size of. `get_unconstrained_geometry` then applies the
    /// client's own `flip`, `slide` and `resize` permissions, which is how a
    /// `<select>` near the bottom of a page opens upwards instead of off the
    /// end of the pane.
    fn place(
        &self,
        positioner: &PositionerState,
        root: TenantId,
        base: (i32, i32),
    ) -> smithay::utils::Rectangle<i32, smithay::utils::Logical> {
        let size = self
            .index_of(root)
            .map(|i| self.tenants[i].skin.size)
            .unwrap_or((0, 0));
        place_in(positioner, base, size)
    }

    /// Re-read a popup's own window geometry after a commit and move it if the
    /// offset between its buffer and its window changed.
    fn refresh_popup_position(&mut self, index: usize) {
        let own = popup_geometry_loc(&self.popups[index].surface);
        let anchor = self.popups[index].anchor;
        let position = (anchor.0 - own.0, anchor.1 - own.1);
        if position == self.popups[index].position {
            return;
        }
        self.popups[index].position = position;
        let id = self.popups[index].id;
        let size = self.popups[index].skin.size;
        self.emit(Event::PopupMoved { id, position, size });
    }

    /// Every popup hanging off `id`, deepest first.
    fn descendants_of(&self, id: PopupId) -> Vec<PopupId> {
        let mut out = Vec::new();
        let mut frontier = vec![id];
        while let Some(parent) = frontier.pop() {
            for p in self.popups.iter().filter(|p| p.parent == Parent::Popup(parent)) {
                out.push(p.id);
                frontier.push(p.id);
            }
        }
        out.reverse();
        out
    }

    /// Is `id` inside the chain rooted at `root` — the chain a grab covers?
    pub(crate) fn in_chain(&self, id: PopupId, root: PopupId) -> bool {
        if id == root {
            return true;
        }
        let mut at = id;
        loop {
            let Some(i) = self.index_of_popup(at) else {
                return false;
            };
            match self.popups[i].parent {
                Parent::Popup(next) if next == root => return true,
                Parent::Popup(next) => at = next,
                Parent::Tenant(_) => return false,
            }
        }
    }

    /// Drop `id` and everything under it, optionally telling the client first.
    ///
    /// The popup is forgotten here rather than when the client gets round to
    /// destroying the object, because the shell must stop drawing a dismissed
    /// menu *now*. A client that ignores `popup_done` and destroys the popup
    /// later finds nothing left to remove, which is the same outcome.
    pub(crate) fn forget_popup_chain(&mut self, id: PopupId, tell: bool) {
        let mut doomed = self.descendants_of(id);
        doomed.push(id);
        for gone in doomed {
            let Some(i) = self.index_of_popup(gone) else {
                continue;
            };
            let popup = self.popups.remove(i);
            if tell && popup.popup.alive() {
                popup.popup.send_popup_done();
            }
            if self.pointer_focus == Some(Target::Popup(gone)) {
                self.pointer_focus = None;
            }
            if self.focus_before_grab == Some(Target::Popup(gone)) {
                self.focus_before_grab = None;
            }
            self.emit(Event::PopupUnmapped { id: gone });
        }
        if self.grab.is_some_and(|g| self.index_of_popup(g).is_none()) {
            self.release_grab();
        }
        self.flush();
    }

    /// Give the keyboard back to whoever had it before the grab.
    fn release_grab(&mut self) {
        self.grab = None;
        let back = self.focus_before_grab.take();
        let surface = back.and_then(|t| self.slot_of_target(t)).map(|slot| self.surface_of(slot));
        let serial = smithay::utils::SERIAL_COUNTER.next_serial();
        self.keyboard.clone().set_focus(self, surface, serial);
        self.keyboard_focus = back.filter(|t| self.slot_of_target(*t).is_some());
    }

    /// Dismiss the grabbing popup and its chain: `popup_done` to the client,
    /// gone from the shell's list.
    pub(crate) fn dismiss_grab(&mut self) {
        if let Some(id) = self.grab {
            self.forget_popup_chain(id, true);
        }
    }

    /// Take down every popup belonging to a tenant.
    pub(crate) fn close_popups_of(&mut self, root: TenantId, tell: bool) {
        let roots: Vec<PopupId> = self
            .popups
            .iter()
            .filter(|p| p.root == root && matches!(p.parent, Parent::Tenant(_)))
            .map(|p| p.id)
            .collect();
        for id in roots {
            self.forget_popup_chain(id, tell);
        }
        // A popup whose chain was rooted elsewhere cannot survive its tenant.
        let orphans: Vec<PopupId> = self
            .popups
            .iter()
            .filter(|p| p.root == root)
            .map(|p| p.id)
            .collect();
        for id in orphans {
            self.forget_popup_chain(id, tell);
        }
    }

    /// Forget popups whose protocol object has gone away without a
    /// `popup_destroyed` reaching us — a client that disconnected mid-menu.
    pub(crate) fn prune_popups(&mut self) {
        let dead: Vec<PopupId> = self
            .popups
            .iter()
            .filter(|p| !p.popup.alive())
            .map(|p| p.id)
            .collect();
        for id in dead {
            self.forget_popup_chain(id, false);
        }
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

        match self.slot_of_surface(surface) {
            Some(Slot::Tenant(index)) => {
                self.refresh_identity(index);
                // The initial configure must be answered before a client is
                // allowed to attach: a toplevel that commits without one is a
                // protocol error on our side, not theirs.
                let toplevel = self.tenants[index].toplevel.clone();
                if !toplevel.is_initial_configure_sent() {
                    toplevel.send_configure();
                }
                if let Some(buffer) = buffer {
                    self.absorb_buffer(Slot::Tenant(index), &buffer);
                }
            }
            Some(Slot::Popup(index)) => {
                // Same rule, same reason: a popup that attaches before it has
                // been configured is a protocol error.
                let popup = self.popups[index].popup.clone();
                if !popup.is_initial_configure_sent() {
                    let _ = popup.send_configure();
                }
                // The client's own window geometry is only known once it has
                // committed one, and it is the difference between where the
                // popup's *buffer* starts and where its *window* does — the
                // shadow margin, for a client that draws one. Re-read it here
                // and move the popup if it moved.
                self.refresh_popup_position(index);
                if let Some(buffer) = buffer {
                    self.absorb_buffer(Slot::Popup(index), &buffer);
                }
            }
            None => {}
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
        // A client that never gets `wl_surface.enter` has to guess which
        // output it is on, and some of them guess a scale of zero.
        self.output.enter(&surface);
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
            configured: Some(size),
            skin: Skin {
                size,
                ..Default::default()
            },
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
        // A popup belongs to its tenant and cannot outlive it. `popup_done`
        // is sent anyway: the client may still be alive and holding the
        // object, and a popup whose parent vanished is exactly what the event
        // is for.
        self.close_popups_of(gone.id, true);
        if self.keyboard_focus == Some(Target::Tenant(gone.id)) {
            self.keyboard_focus = None;
            let serial = smithay::utils::SERIAL_COUNTER.next_serial();
            self.keyboard.clone().set_focus(self, None, serial);
        }
        if self.pointer_focus == Some(Target::Tenant(gone.id)) {
            self.pointer_focus = None;
        }
        if self.focus_before_grab == Some(Target::Tenant(gone.id)) {
            self.focus_before_grab = None;
        }
        self.emit(Event::Unmapped { id: gone.id });
    }

    /// A popup is a second surface over a tenant: it gets its own id, its own
    /// texture and a position, and the shell draws it wherever it draws that
    /// tenant.
    ///
    /// The positioner is resolved here and not in the shell, because resolving
    /// it is protocol — anchor, gravity, offset and the constraint rules — and
    /// not placement. What *is* placement, and stays the shell's, is where the
    /// pane the popup hangs off is, how the position is scaled, and how far
    /// off its plane the popup is lifted.
    fn new_popup(&mut self, surface: PopupSurface, positioner: PositionerState) {
        let Some(parent_surface) = surface.get_parent_surface() else {
            // No parent yet: the client will set one before it commits, and
            // the protocol makes committing without one its error, not ours.
            let _ = surface.send_configure();
            return;
        };
        let Some((parent, root, base)) = self.parent_of(&parent_surface) else {
            self.reject(None, "xdg_popup on a surface this compositor does not host".into());
            let _ = surface.send_configure();
            return;
        };

        let geometry = self.place(&positioner, root, base);
        surface.with_pending_state(|state| {
            state.positioner = positioner;
            state.geometry = geometry;
        });

        self.next_popup += 1;
        let id = PopupId(self.next_popup);
        let anchor = (base.0 + geometry.loc.x, base.1 + geometry.loc.y);
        let size = (geometry.size.w.max(0) as u32, geometry.size.h.max(0) as u32);
        let wl_surface = surface.wl_surface().clone();
        // Same as a toplevel: the client is told which output it is on before
        // it is asked to paint.
        self.output.enter(&wl_surface);
        self.popups.push(Popup {
            id,
            parent,
            root,
            anchor,
            position: anchor,
            grabbed: false,
            skin: Skin {
                size,
                ..Default::default()
            },
            surface: wl_surface,
            popup: surface,
        });
        self.emit(Event::PopupMapped {
            id,
            parent,
            root,
            position: anchor,
            size,
        });
        self.flush();
    }

    /// The popup asks to be modal.
    ///
    /// This crate has no input devices of its own — the shell injects every
    /// motion and every button — so there is no seat grab to install. What a
    /// grab means here is the one behaviour the client is actually waiting on:
    /// a button pressed outside this popup's chain sends `popup_done` instead
    /// of reaching what is under it. Without that a `<select>` menu never
    /// closes and the page behind it never gets its click, which is the
    /// hanging-Chromium shape.
    fn grab(
        &mut self,
        surface: PopupSurface,
        _seat: smithay::reexports::wayland_server::protocol::wl_seat::WlSeat,
        _serial: smithay::utils::Serial,
    ) {
        let Some(index) = self.index_of_popup_surface(surface.wl_surface()) else {
            return;
        };
        let id = self.popups[index].id;
        self.popups[index].grabbed = true;
        if self.grab.is_none() {
            self.focus_before_grab = self.keyboard_focus;
        }
        self.grab = Some(id);
        // The keyboard follows the grab: a menu you cannot arrow through is
        // half a menu.
        let wl_surface = self.popups[index].surface.clone();
        let serial = smithay::utils::SERIAL_COUNTER.next_serial();
        self.keyboard
            .clone()
            .set_focus(self, Some(wl_surface), serial);
        self.keyboard_focus = Some(Target::Popup(id));
        self.emit(Event::PopupGrabbed { id });
        self.flush();
    }

    /// The client wants the popup somewhere else — an autocomplete list that
    /// grew, a menu whose anchor moved. Same resolution as `new_popup`, and
    /// the token comes back so the client knows which request it answers.
    fn reposition_request(
        &mut self,
        surface: PopupSurface,
        positioner: PositionerState,
        token: u32,
    ) {
        let Some(index) = self.index_of_popup_surface(surface.wl_surface()) else {
            surface.send_repositioned(token);
            let _ = surface.send_configure();
            return;
        };
        let root = self.popups[index].root;
        let base = self.base_of_parent(self.popups[index].parent);
        let geometry = self.place(&positioner, root, base);
        surface.with_pending_state(|state| {
            state.positioner = positioner;
            state.geometry = geometry;
        });
        surface.send_repositioned(token);
        let _ = surface.send_configure();

        let anchor = (base.0 + geometry.loc.x, base.1 + geometry.loc.y);
        let size = (geometry.size.w.max(0) as u32, geometry.size.h.max(0) as u32);
        self.popups[index].anchor = anchor;
        let id = self.popups[index].id;
        // The buffer's own offset is re-read on the commit that follows; the
        // move is announced now so a shell has the new place before the new
        // pixels.
        let own = popup_geometry_loc(&self.popups[index].surface);
        let position = (anchor.0 - own.0, anchor.1 - own.1);
        self.popups[index].position = position;
        self.emit(Event::PopupMoved { id, position, size });
        self.flush();
    }

    /// The client destroyed the popup. It is gone whether or not it was ever
    /// dismissed, and so is everything hanging off it.
    fn popup_destroyed(&mut self, surface: PopupSurface) {
        let Some(index) = self.index_of_popup_surface(surface.wl_surface()) else {
            return;
        };
        let id = self.popups[index].id;
        self.forget_popup_chain(id, false);
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

#[cfg(test)]
mod tests {
    use super::place_in;
    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_positioner::{
        Anchor, ConstraintAdjustment, Gravity,
    };
    use smithay::utils::{Point, Rectangle, Size};
    use smithay::wayland::shell::xdg::PositionerState;

    /// A `<select>` at `(x, y)` on the page, `w` x `h`, opening downwards —
    /// which is what Chromium's menus ask for.
    fn menu(x: i32, y: i32, w: i32, h: i32) -> PositionerState {
        PositionerState {
            rect_size: Size::from((w, h)),
            anchor_rect: Rectangle::new(Point::from((x, y)), Size::from((1, 1))),
            anchor_edges: Anchor::BottomLeft,
            gravity: Gravity::BottomRight,
            constraint_adjustment: ConstraintAdjustment::FlipY
                | ConstraintAdjustment::SlideX
                | ConstraintAdjustment::SlideY,
            ..Default::default()
        }
    }

    /// A menu with room below it lands under its anchor, and nowhere else. The
    /// numbers are the anchor's, not a guess: bottom-left anchor with
    /// bottom-right gravity is "hang from this point, growing right and down".
    #[test]
    fn a_menu_hangs_from_its_anchor() {
        let geo = place_in(&menu(100, 200, 160, 120), (0, 0), (1280, 800));
        assert_eq!((geo.loc.x, geo.loc.y), (100, 201));
        assert_eq!((geo.size.w, geo.size.h), (160, 120));
    }

    /// A menu near the foot of the page flips above its anchor rather than
    /// running off the end of the pane. This is the whole reason the tenant's
    /// rectangle is used as the constraint target: with no target it would be
    /// placed at y = 761 on an 800-pixel pane and half of it would be nowhere.
    #[test]
    fn a_menu_at_the_foot_flips_up() {
        let geo = place_in(&menu(100, 760, 160, 120), (0, 0), (1280, 800));
        assert!(
            geo.loc.y + geo.size.h <= 800,
            "flipped menu still runs off the pane: {geo:?}"
        );
        assert!(geo.loc.y < 760, "it did not flip: {geo:?}");
    }

    /// The constraint rectangle follows the parent's window geometry. A parent
    /// whose window starts 40 pixels into its buffer has 40 fewer pixels below
    /// the anchor, so the target rectangle starts at -40 in the positioner's
    /// space — and a menu that fitted at `base = 0` is pushed or flipped here.
    #[test]
    fn the_constraint_is_relative_to_the_parents_window_geometry() {
        // It fits below the anchor on the pane (671 + 120 = 791 < 800), but
        // not once the parent's window starts 40 pixels down the buffer.
        let p = menu(100, 670, 160, 120);
        let flat = place_in(&p, (0, 0), (1280, 800));
        let inset = place_in(&p, (0, 40), (1280, 800));
        assert_ne!(flat, inset, "the base offset made no difference");
        assert_eq!((flat.loc.x, flat.loc.y), (100, 671), "it should fit as it is");
        assert!(
            inset.loc.y + inset.size.h <= 800 - 40,
            "constrained against the wrong rectangle: {inset:?}"
        );
    }

    /// A tenant that has not painted has no rectangle to constrain against,
    /// and the positioner's own answer is used unchanged rather than being
    /// squeezed into a size the tenant never had.
    #[test]
    fn an_unpainted_tenant_constrains_nothing() {
        let p = menu(100, 760, 160, 120);
        assert_eq!(place_in(&p, (0, 0), (0, 0)), p.get_geometry());
    }
}
