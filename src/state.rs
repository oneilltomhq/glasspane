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
use std::sync::{Arc, Mutex};
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
use smithay::utils::Transform;
use smithay::wayland::compositor::{
    BufferAssignment, CompositorClientState, CompositorHandler, CompositorState,
    SubsurfaceCachedState, SurfaceAttributes, SurfaceData, TraversalAction, get_children,
    get_parent, send_surface_state, with_states, with_surface_tree_upward,
};
use smithay::wayland::fractional_scale::{
    FractionalScaleHandler, FractionalScaleManagerState, with_fractional_scale,
};
use smithay::wayland::viewporter::{ViewportCachedState, ViewporterState};
use smithay::wayland::dmabuf::{DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier, get_dmabuf};
use smithay::wayland::selection::SelectionHandler;
use smithay::wayland::selection::primary_selection::{
    PrimarySelectionHandler, PrimarySelectionState, set_primary_focus,
};
use smithay::wayland::selection::data_device::{
    DataDeviceHandler, DataDeviceState, ServerDndGrabHandler, set_data_device_focus,
};
use smithay::wayland::shell::xdg::decoration::{XdgDecorationHandler, XdgDecorationState};
use smithay::wayland::shell::kde::decoration::{KdeDecorationHandler, KdeDecorationState};
use smithay::reexports::wayland_protocols_misc::server_decoration::server::org_kde_kwin_server_decoration::{
    Mode as KdeMode, OrgKdeKwinServerDecoration,
};
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, SurfaceCachedState, ToplevelSurface, XdgShellHandler,
    XdgShellState, XdgToplevelSurfaceData,
};
use smithay::wayland::shm::{ShmHandler, ShmState, with_buffer_contents};

use crate::acquire::AcquireGate;
use crate::compose::{Compositor, Quad};
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

/// What a commit says about how its buffer maps onto the surface: the
/// `wl_surface.set_buffer_scale`, the `wp_viewport` size (its destination,
/// else its source's), and the viewport's source rectangle, if any.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Shape {
    pub(crate) buffer_scale: i32,
    pub(crate) viewport: Option<(u32, u32)>,
    /// The `wp_viewport` source: `x, y, width, height` in surface-local
    /// logical pixels, which is buffer pixels over `buffer_scale`.
    pub(crate) source: Option<(f64, f64, f64, f64)>,
}

impl Shape {
    /// The surface's logical size for a buffer of `buffer` pixels.
    ///
    /// A viewport destination wins, since it is exactly the client saying
    /// "this buffer is this big": a fractionally scaled client draws 2.1x and
    /// sets one. Otherwise the buffer is divided by its integer scale. A
    /// viewport *source* is not this function's: the texture is already the
    /// cropped rectangle ([`Shape::crop`]) when it is asked.
    pub(crate) fn logical(self, buffer: (u32, u32)) -> (u32, u32) {
        if let Some(size) = self.viewport {
            return size;
        }
        let s = self.buffer_scale.max(1) as u32;
        (buffer.0 / s, buffer.1 / s)
    }

    /// The rectangle of a `buffer`-pixel buffer that is the surface's picture:
    /// `x, y, width, height` in buffer pixels. The viewport's source, scaled
    /// to buffer pixels and kept inside the buffer, else the whole buffer.
    ///
    /// A client that sets one is drawing into part of a bigger buffer —
    /// Chromium keeps its buffer when a window shrinks and crops it — and the
    /// texture is only that part, so a shell drawing the texture whole draws
    /// what the client meant. A source that misses the buffer is a protocol
    /// error glasspane does not raise; it gets the whole buffer.
    pub(crate) fn crop(self, buffer: (u32, u32)) -> (u32, u32, u32, u32) {
        let whole = (0, 0, buffer.0, buffer.1);
        let Some((x, y, w, h)) = self.source else {
            return whole;
        };
        let s = f64::from(self.buffer_scale.max(1));
        let (bw, bh) = (f64::from(buffer.0), f64::from(buffer.1));
        let (x0, y0) = (
            (x * s).round().clamp(0.0, bw),
            (y * s).round().clamp(0.0, bh),
        );
        let (x1, y1) = (
            ((x + w) * s).round().clamp(0.0, bw),
            ((y + h) * s).round().clamp(0.0, bh),
        );
        if x1 <= x0 || y1 <= y0 {
            return whole;
        }
        (x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32)
    }

    /// The surface's picture as the texture holds it: the rectangle of a
    /// `buffer`-pixel buffer that is its window, the logical size that is, and
    /// where that window starts in the surface. [`Shape::crop`] and
    /// [`Shape::logical`], then `window` inside them, if it is set and leaves
    /// anything inside the surface.
    ///
    /// `window` is a toplevel's `xdg_surface.set_window_geometry` (`State::window`):
    /// `x, y, width, height` in surface-local logical pixels. A client that
    /// draws its own decorations — GTK always does, and takes no
    /// `zxdg_toplevel_decoration_v1` to say otherwise — sizes the *window* to
    /// the configure and its buffer bigger by a shadow margin on every side,
    /// saying where the window is in it with this. The texture is the window
    /// alone, so a pane shows the window it asked for, not a shadow round it.
    pub(crate) fn frame(self, buffer: (u32, u32), window: Option<(i32, i32, i32, i32)>) -> Frame {
        let crop = self.crop(buffer);
        let logical = self.logical((crop.2, crop.3));
        let whole = Frame {
            crop,
            logical,
            origin: (0, 0),
        };
        let Some((x, y, w, h)) = window.and_then(|g| window_in(g, logical)) else {
            return whole;
        };
        // Window edges in logical pixels, scaled to the crop's texels.
        let (kx, ky) = (
            f64::from(crop.2) / f64::from(logical.0),
            f64::from(crop.3) / f64::from(logical.1),
        );
        let at = |v: u32, k: f64| (f64::from(v) * k).round() as u32;
        let (x0, y0) = (at(x, kx), at(y, ky));
        let (x1, y1) = (at(x + w, kx), at(y + h, ky));
        if x1 <= x0 || y1 <= y0 {
            return whole;
        }
        Frame {
            crop: (crop.0 + x0, crop.1 + y0, x1 - x0, y1 - y0),
            logical: (w, h),
            origin: (x as i32, y as i32),
        }
    }

    fn of(states: &SurfaceData) -> Shape {
        let buffer_scale = states
            .cached_state
            .get::<SurfaceAttributes>()
            .current()
            .buffer_scale;
        let mut cached = states.cached_state.get::<ViewportCachedState>();
        let port = cached.current();
        let viewport = port
            .size()
            .filter(|s| s.w > 0 && s.h > 0)
            .map(|s| (s.w as u32, s.h as u32));
        let source = port.src.map(|r| (r.loc.x, r.loc.y, r.size.w, r.size.h));
        Shape {
            buffer_scale,
            viewport,
            source,
        }
    }
}

/// What [`Shape::frame`] makes of a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Frame {
    /// `x, y, width, height` in buffer pixels: the texels taken.
    pub(crate) crop: (u32, u32, u32, u32),
    /// The logical size they are.
    pub(crate) logical: (u32, u32),
    /// Where they start in the surface, in its logical pixels.
    pub(crate) origin: (i32, i32),
}

/// A window geometry clamped to a surface of `logical` size, as
/// `xdg_surface.set_window_geometry` says the effective geometry is: `x, y,
/// width, height` inside it, or `None` if nothing is left or it is the whole
/// surface anyway.
fn window_in(window: (i32, i32, i32, i32), logical: (u32, u32)) -> Option<(u32, u32, u32, u32)> {
    let (lw, lh) = (logical.0 as i32, logical.1 as i32);
    let (x0, y0) = (window.0.clamp(0, lw), window.1.clamp(0, lh));
    let (x1, y1) = (
        window.0.saturating_add(window.2).clamp(0, lw),
        window.1.saturating_add(window.3).clamp(0, lh),
    );
    if x1 <= x0 || y1 <= y0 || (x0, y0, x1, y1) == (0, 0, lw, lh) {
        return None;
    }
    Some((x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32))
}

/// A toplevel's window geometry as it was last committed, if it set one.
fn window_of(surface: &WlSurface) -> Option<(i32, i32, i32, i32)> {
    with_states(surface, |states| {
        states
            .cached_state
            .get::<SurfaceCachedState>()
            .current()
            .geometry
            .map(|g| (g.loc.x, g.loc.y, g.size.w, g.size.h))
    })
}

/// Ask one surface to draw at `scale`, through both protocols that can say
/// it. Each only sends when the value moved, so asking twice is free.
pub(crate) fn ask_scale(surface: &WlSurface, scale: f64) {
    with_states(surface, |states| {
        with_fractional_scale(states, |fs| fs.set_preferred_scale(scale));
        // Up, not to nearest: an integer-only client asked for 1.5 that
        // draws 2 is downsampled a little; one that draws 1 is blurred.
        send_surface_state(
            surface,
            states,
            scale.ceil().max(1.0) as i32,
            Transform::Normal,
        );
    });
}

/// Everything the compositor owns. One struct, because calloop dispatches
/// against exactly one.
pub struct State {
    pub(crate) start: Instant,
    pub(crate) dh: DisplayHandle,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,

    pub(crate) compositor_state: CompositorState,
    /// Held for as long as the display, like the decoration state: dropping
    /// it withdraws `wp_fractional_scale_manager_v1`.
    #[allow(dead_code)]
    pub(crate) fractional_scale_state: FractionalScaleManagerState,
    /// Held for as long as the display: dropping it withdraws `wp_viewporter`,
    /// which is how a fractionally scaled client says its logical size.
    #[allow(dead_code)]
    pub(crate) viewporter_state: ViewporterState,
    pub(crate) xdg_shell_state: XdgShellState,
    /// Held for as long as the display: dropping the state would withdraw
    /// the `zxdg_decoration_manager_v1` global from under live clients.
    #[allow(dead_code)]
    /// Held for as long as the display: dropping the state would withdraw
    /// the `zxdg_decoration_manager_v1` global from under live clients.
    #[allow(dead_code)]
    pub(crate) xdg_decoration_state: XdgDecorationState,
    /// Held for as long as the display: dropping it withdraws
    /// `org_kde_kwin_server_decoration_manager`, the other way a client asks
    /// who draws its frame, and the only one GTK4 apps that ask at all use.
    #[allow(dead_code)]
    pub(crate) kde_decoration_state: KdeDecorationState,
    pub(crate) shm_state: ShmState,
    pub(crate) dmabuf_state: DmabufState,
    pub(crate) dmabuf_global: Option<DmabufGlobal>,
    /// What the global's default feedback names; see
    /// [`crate::Glasspane::dmabuf_main_device`].
    pub(crate) dmabuf_main_device: Option<libc::dev_t>,
    pub(crate) data_device_state: DataDeviceState,
    pub(crate) primary_selection_state: PrimarySelectionState,
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
    /// The draw for dma-bufs in the other byte order, built on the first.
    pub(crate) reorder: Option<texture::Reorder>,
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
    /// Frame callbacks harvested at commit and not yet answered, each with
    /// the tenant it belongs to: a toplevel's own, and a popup's or a
    /// subsurface's its root tenant's, so a shell that paces a pane paces its
    /// menus with it.
    /// `None` for a surface that is neither (nothing to pace it by), which
    /// every release answers.
    pub(crate) pending_callbacks: Vec<(Option<TenantId>, WlCallback)>,
    pub(crate) default_size: (u32, u32),
    pub(crate) keyboard_focus: Option<Target>,
    pub(crate) pointer_focus: Option<Target>,
    pub(crate) ticks: u64,
    pub(crate) callbacks_sent: u64,
    /// Built the first time a tenant or popup has subsurfaces to draw, and
    /// never for a run without one.
    pub(crate) compositor: Option<Compositor>,
    /// The root surfaces — a tenant's or a popup's — of trees that a commit
    /// changed since the last [`State::compose_trees`]. Drawn once per
    /// dispatch however many of their surfaces committed: a synchronized
    /// tree applies all its children's commits along with its parent's, and
    /// drawing after each would draw the same picture several times.
    pub(crate) dirty_trees: Vec<WlSurface>,
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

    /// A slot's window geometry as last committed: a toplevel's, if it set
    /// one. Read from smithay whenever it is wanted, by a buffer drawn alone
    /// and by a tree recomposed with no commit (a restack) alike.
    fn window(&self, slot: Slot) -> Option<(i32, i32, i32, i32)> {
        match slot {
            Slot::Tenant(i) => window_of(&self.tenants[i].surface),
            Slot::Popup(_) => None,
        }
    }

    pub(crate) fn skin(&self, slot: Slot) -> &Skin {
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

    /// Answer the frame callbacks harvested since the last tick of every
    /// tenant `due` picks, leave the rest pending, and say how many were
    /// answered.
    ///
    /// Frame callbacks are the compositor's to *drive* (see the crate docs): a
    /// Wayland client paints only when handed one, so answering them on demand
    /// collapses to "when someone asks" and the tenants freeze. Withholding
    /// one from a tenant is how a shell slows it down: it is held, not
    /// dropped, so the tenant paints again the tick it is picked.
    ///
    /// `due` is asked once per tenant holding a pending callback, and only
    /// then: a tenant it says yes to is one that got callbacks this tick.
    /// Callbacks of a surface that belongs to no live tenant are always
    /// answered; those of a client that has gone are dropped.
    pub(crate) fn release_frame_callbacks(
        &mut self,
        now_ms: u32,
        mut due: impl FnMut(TenantId) -> bool,
    ) -> usize {
        self.ticks += 1;
        if self.pending_callbacks.is_empty() {
            return 0;
        }
        let mut asked: Vec<(TenantId, bool)> = Vec::new();
        let mut kept = Vec::new();
        let mut n = 0;
        for (owner, cb) in std::mem::take(&mut self.pending_callbacks) {
            if !cb.is_alive() {
                continue;
            }
            let index = owner.and_then(|id| self.index_of(id));
            let go = match (owner, index) {
                (Some(id), Some(_)) => match asked.iter().find(|(a, _)| *a == id) {
                    Some((_, go)) => *go,
                    None => {
                        let go = due(id);
                        asked.push((id, go));
                        go
                    }
                },
                _ => true,
            };
            if !go {
                kept.push((owner, cb));
                continue;
            }
            cb.done(now_ms);
            self.callbacks_sent += 1;
            if let Some(i) = index {
                self.tenants[i].callbacks_sent += 1;
            }
            n += 1;
        }
        self.pending_callbacks = kept;
        if n == 0 {
            return 0;
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

    /// Make sure this surface's texture holds `frame`: its texels (the
    /// crop's size), its logical size and its origin in the surface,
    /// recreating the texture if it has to and saying so if texels or size
    /// moved.
    ///
    /// The one place a skin's picture is described, for a buffer drawn alone
    /// and a tree's composite alike.
    fn ensure_texture(&mut self, slot: Slot, frame: Frame) {
        let (width, height) = (frame.crop.2, frame.crop.3);
        let size = frame.logical;
        self.skin_mut(slot).origin = frame.origin;
        let fits = self
            .skin(slot)
            .texture
            .as_ref()
            .is_some_and(|t| t.width() == width && t.height() == height);
        if fits && self.skin(slot).size == size {
            return;
        }
        if !fits {
            let label = match slot {
                Slot::Tenant(i) => format!("tenant#{}", self.tenants[i].id.0),
                Slot::Popup(i) => format!("popup#{}", self.popups[i].id.0),
            };
            let tex = texture::create_surface_texture(&self.device, &label, width, height);
            self.skin_mut(slot).texture = Some(Arc::new(tex));
        }
        let skin = self.skin_mut(slot);
        skin.size = size;
        skin.buffer = (width, height);
        let event = match slot {
            Slot::Tenant(i) => Event::Resized {
                id: self.tenants[i].id,
                size,
            },
            Slot::Popup(i) => Event::PopupResized {
                id: self.popups[i].id,
                size,
            },
        };
        self.emit(event);
    }

    /// The tenant a slot's pixels belong to, as far as a shell watching for
    /// trouble is concerned: a popup's are its root tenant's.
    fn owner_of(&self, slot: Slot) -> TenantId {
        match slot {
            Slot::Tenant(i) => self.tenants[i].id,
            Slot::Popup(i) => self.popups[i].root,
        }
    }

    /// Say that the texture behind `slot` has new pixels.
    fn committed(&mut self, slot: Slot) {
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

    /// One committed buffer becomes texels: the fast path, for a surface with
    /// no subsurfaces, straight into its texture.
    fn absorb_buffer(&mut self, slot: Slot, buffer: &WlBuffer, shape: Shape) {
        match self.absorb_into(Dest::Skin(slot, self.window(slot)), buffer, shape) {
            Ok(()) => self.committed(slot),
            Err(why) => {
                let id = self.owner_of(slot);
                self.reject(Some(id), why);
            }
        }
    }

    /// Put a buffer's pixels at `dest`, by whichever path its memory needs.
    fn absorb_into(
        &mut self,
        dest: Dest<'_>,
        buffer: &WlBuffer,
        shape: Shape,
    ) -> Result<(), String> {
        match get_dmabuf(buffer) {
            Ok(dmabuf) => self.absorb_dmabuf(dest, buffer_key(buffer), &dmabuf.clone(), shape),
            Err(_) => self.absorb_shm(dest, buffer, shape),
        }
    }

    /// The texture `frame`'s texels are written into at `dest`, made to fit
    /// first.
    fn target(&mut self, dest: Dest<'_>, frame: Frame, opaque: bool) -> Arc<wgpu::Texture> {
        let crop = (frame.crop.2, frame.crop.3);
        match dest {
            Dest::Skin(slot, _) => {
                self.ensure_texture(slot, frame);
                self.skin(slot)
                    .texture
                    .clone()
                    .expect("ensure_texture just set it")
            }
            Dest::Layer(surface, _) => {
                let device = &self.device;
                with_states(surface, |states| {
                    let mut layer = layer_of(states).lock().unwrap();
                    let fits = layer
                        .texture
                        .as_ref()
                        .is_some_and(|t| (t.width(), t.height()) == crop);
                    if !fits {
                        let label = format!("layer@{}", surface.id().protocol_id());
                        let tex = texture::create_surface_texture(device, &label, crop.0, crop.1);
                        layer.texture = Some(Arc::new(tex));
                    }
                    layer.size = frame.logical;
                    layer.opaque = opaque;
                    layer.texture.clone().expect("just set")
                })
            }
        }
    }

    /// Record whether the last dma-buf commit staged an acquire wait, on the
    /// skin the pixels end up in.
    fn set_staged(&mut self, dest: Dest<'_>, staged: bool) {
        let slot = match dest {
            Dest::Skin(slot, _) | Dest::Layer(_, Some(slot)) => slot,
            Dest::Layer(_, None) => return,
        };
        self.skin_mut(slot).acquire_staged = staged;
    }

    fn absorb_dmabuf(
        &mut self,
        dest: Dest<'_>,
        key: BufferKey,
        dmabuf: &Dmabuf,
        shape: Shape,
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
            self.set_staged(dest, true);
        } else {
            self.set_staged(dest, false);
        }

        let (width, height) = (dmabuf.width(), dmabuf.height());
        let opaque = dmabuf.format().code == drm_fourcc::DrmFourcc::Xrgb8888;
        let device = self.device.clone();
        let described = crate::convert::describe(dmabuf)?;
        let (_id, src, _imported) = self
            .registry
            .get_or_import(key, || {
                wgpu_dmabuf::import(&device, &described, texture::IMPORT_USAGE)
            })
            .map_err(|e| format!("dmabuf import failed: {e}"))?;

        let frame = shape.frame((width, height), dest.window());
        let crop = frame.crop;
        let dst = self.target(dest, frame, opaque);
        if src.format() == dst.format() {
            texture::blit(&self.device, &self.queue, &src, &dst, crop);
        } else {
            let device = &self.device;
            self.reorder
                .get_or_insert_with(|| texture::Reorder::new(device))
                .draw(device, &self.queue, &src, &dst, crop);
        }
        Ok(())
    }

    fn absorb_shm(
        &mut self,
        dest: Dest<'_>,
        buffer: &WlBuffer,
        shape: Shape,
    ) -> Result<(), String> {
        let read = with_buffer_contents(buffer, |ptr, len, data| {
            let opaque = match data.format {
                wl_shm::Format::Argb8888 => false,
                wl_shm::Format::Xrgb8888 => true,
                other => return Err(format!("unsupported wl_shm format {other:?}")),
            };
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
            Ok((w, h, stride, bytes, opaque))
        });
        let (w, h, stride, bytes, opaque) = match read {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return Err(e),
            Err(e) => return Err(format!("buffer is neither dma-buf nor wl_shm: {e}")),
        };
        self.set_staged(dest, false);
        let frame = shape.frame((w, h), dest.window());
        let crop = frame.crop;
        let dst = self.target(dest, frame, opaque);
        texture::upload_shm(&self.queue, &dst, &bytes, stride, crop);
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
    /// geometry origin, in the root tenant's logical pixels).
    fn parent_of(&self, parent_surface: &WlSurface) -> Option<(Parent, TenantId, (i32, i32))> {
        if let Some(i) = self.index_of_surface(parent_surface) {
            let id = self.tenants[i].id;
            return Some((Parent::Tenant(id), id, self.tenant_base(i)));
        }
        let i = self.index_of_popup_surface(parent_surface)?;
        let p = &self.popups[i];
        Some((Parent::Popup(p.id), p.root, p.anchor))
    }

    /// The same point for a parent already known by identity.
    fn base_of_parent(&self, parent: Parent) -> (i32, i32) {
        match parent {
            Parent::Tenant(id) => match self.index_of(id) {
                Some(i) => self.tenant_base(i),
                None => (0, 0),
            },
            Parent::Popup(id) => match self.index_of_popup(id) {
                Some(i) => self.popups[i].anchor,
                None => (0, 0),
            },
        }
    }

    /// A tenant's window geometry origin in the tenant's own space: zero once
    /// its texture is cut to the window (`Shape::frame`), which is the case
    /// that matters, and the geometry's offset still for one the surface's
    /// extent clamps.
    fn tenant_base(&self, index: usize) -> (i32, i32) {
        let (x, y) = popup_geometry_loc(&self.tenants[index].surface);
        let origin = self.tenants[index].skin.origin;
        (x - origin.0, y - origin.1)
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
            for p in self
                .popups
                .iter()
                .filter(|p| p.parent == Parent::Popup(parent))
            {
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
        let surface = back
            .and_then(|t| self.slot_of_target(t))
            .map(|slot| self.surface_of(slot));
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
// Subsurfaces
// ---------------------------------------------------------------------------

/// Where a committed buffer's pixels are put.
#[derive(Debug, Clone, Copy)]
enum Dest<'a> {
    /// Straight into a tenant's or a popup's texture: a surface with no
    /// subsurfaces, which is most of them. With the window cut from the
    /// buffer, for a toplevel that set one ([`Shape::frame`]).
    Skin(Slot, Option<(i32, i32, i32, i32)>),
    /// Into a surface's own [`Layer`], to be composited into its tree's
    /// texture: a subsurface, or a root that has some. The slot is the root's,
    /// if this crate hosts it yet.
    Layer(&'a WlSurface, Option<Slot>),
}

impl Dest<'_> {
    /// The window cut from a buffer put here: a layer is always its whole
    /// surface, and a tree's window is cut from its composite (`compose`).
    fn window(self) -> Option<(i32, i32, i32, i32)> {
        match self {
            Dest::Skin(_, window) => window,
            Dest::Layer(..) => None,
        }
    }
}

/// One surface of a tree's own pixels, kept on the surface (in its
/// `data_map`, so it goes when the surface does) until the tree is drawn.
///
/// A copy per surface on top of the composite is the cost of drawing a tree:
/// a desync subsurface commits alone, and its parent's pixels have to be
/// somewhere to be drawn under it again. A client without subsurfaces never
/// gets one.
#[derive(Debug, Default)]
struct Layer {
    /// The buffer, cropped to its viewport's source; `None` once the client
    /// attaches a null buffer, which unmaps the surface.
    texture: Option<Arc<wgpu::Texture>>,
    /// The logical size it is drawn at: [`Shape::logical`] of the texture.
    size: (u32, u32),
    /// `XRGB8888`: drawn as if its alpha were 1. See [`Quad::opaque`].
    opaque: bool,
}

fn layer_of(states: &SurfaceData) -> &Mutex<Layer> {
    states
        .data_map
        .get_or_insert_threadsafe(Mutex::<Layer>::default)
}

/// Where a surface of a tree is, given where its parent is: `None` for a
/// surface that is not drawn, because it has no buffer or because its parent
/// is not drawn.
fn place_of(states: &SurfaceData, parent: Option<(i32, i32)>) -> Option<(i32, i32)> {
    let (x, y) = parent?;
    let mapped = matches!(
        states
            .cached_state
            .get::<SurfaceAttributes>()
            .current()
            .buffer,
        Some(BufferAssignment::NewBuffer(_))
    );
    if !mapped {
        return None;
    }
    if states.role != Some(SUBSURFACE) {
        return Some((x, y));
    }
    let at = states
        .cached_state
        .get::<SubsurfaceCachedState>()
        .current()
        .location;
    Some((x + at.x, y + at.y))
}

/// smithay's name for the role `wl_subcompositor.get_subsurface` gives.
const SUBSURFACE: &str = "subsurface";

/// One drawn surface of a tree, as [`walk`] finds it.
struct Piece {
    surface: WlSurface,
    /// Its top-left in the root's logical pixels.
    at: (i32, i32),
    /// The buffer it holds, for a layer that was never filled.
    buffer: Option<WlBuffer>,
    shape: Shape,
    layer: Option<(Arc<wgpu::Texture>, (u32, u32), bool)>,
}

fn layer_snapshot(states: &SurfaceData) -> Option<(Arc<wgpu::Texture>, (u32, u32), bool)> {
    let layer = layer_of(states).lock().unwrap();
    layer.texture.clone().map(|t| (t, layer.size, layer.opaque))
}

/// Every drawn surface of the tree under `root`, back to front, and a hash of
/// the tree's shape: which surfaces there are, in what order, and where.
///
/// Back to front is smithay's upward walk, which honours `place_above` and
/// `place_below` — including a subsurface placed below its own parent, which
/// is drawn before it.
fn walk(root: &WlSurface) -> (Vec<Piece>, u64) {
    let mut pieces = Vec::new();
    let mut hash = DefaultHasher::new();
    with_surface_tree_upward(
        root,
        Some((0, 0)),
        |_, states, parent| TraversalAction::DoChildren(place_of(states, *parent)),
        |surface, states, parent| {
            let at = place_of(states, *parent);
            surface.id().hash(&mut hash);
            at.hash(&mut hash);
            let Some(at) = at else {
                return;
            };
            let buffer = match &states
                .cached_state
                .get::<SurfaceAttributes>()
                .current()
                .buffer
            {
                Some(BufferAssignment::NewBuffer(b)) => Some(b.clone()),
                _ => None,
            };
            pieces.push(Piece {
                surface: surface.clone(),
                at,
                buffer,
                shape: Shape::of(states),
                layer: layer_snapshot(states),
            });
        },
        |_, _, _| true,
    );
    (pieces, hash.finish())
}

/// The size of a tree's composite texture: the root's logical size, at the
/// denser of the root's own buffer and the scale the tenant was asked for.
///
/// Not simply the root's buffer, because the root's buffer can be a token:
/// Waydroid's is 1x1, stretched over the window by a viewport, with all of
/// Android in a subsurface on top. A composite the size of that buffer would
/// be one pixel.
fn composite_size(root: (u32, u32), logical: (u32, u32), scale: f64) -> (u32, u32) {
    let asked = |l: u32| (f64::from(l) * scale).round().max(1.0) as u32;
    (root.0.max(asked(logical.0)), root.1.max(asked(logical.1)))
}

/// The root of the subsurface tree `surface` is in: the surface at the top of
/// its chain of `wl_subsurface` parents, itself if it has none.
fn root_of(surface: &WlSurface) -> WlSurface {
    let mut at = surface.clone();
    while let Some(parent) = get_parent(&at) {
        at = parent;
    }
    at
}

/// Every subsurface under `surface`, at any depth.
pub(crate) fn subsurfaces_of(surface: &WlSurface) -> Vec<WlSurface> {
    let mut out = Vec::new();
    let mut frontier = get_children(surface);
    while let Some(next) = frontier.pop() {
        frontier.extend(get_children(&next));
        out.push(next);
    }
    out
}

impl State {
    /// A root's commit: its own buffer, and — when its tree is synchronized —
    /// its children's, which smithay has just applied and committed first.
    ///
    /// **A root with no subsurfaces takes the path it always has**: its
    /// buffer straight into its texture, no layer, no composite, no pipeline.
    /// What a subsurface costs is paid only by a client that makes one.
    fn absorb_root(
        &mut self,
        slot: Slot,
        surface: &WlSurface,
        buffer: Option<Option<WlBuffer>>,
        shape: Shape,
    ) {
        if get_children(surface).is_empty() {
            if self.skin_mut(slot).tree.take().is_some() {
                // It had a tree, and has drawn alone since: its layer is a
                // copy nothing will read.
                with_states(surface, |states| {
                    layer_of(states).lock().unwrap().texture = None
                });
            }
            if let Some(Some(buffer)) = buffer {
                self.absorb_buffer(slot, &buffer, shape);
            }
            return;
        }
        // The root's layer is its whole surface, as every layer is; the
        // window is cut from the composite (`compose`).
        self.absorb_layer(surface, Some(slot), buffer, shape);
    }

    /// A tree surface's commit: its buffer into its layer, a null buffer out
    /// of it, and the tree marked for drawing either way, since a commit may
    /// also have moved it.
    fn absorb_layer(
        &mut self,
        surface: &WlSurface,
        root: Option<Slot>,
        buffer: Option<Option<WlBuffer>>,
        shape: Shape,
    ) {
        match buffer {
            Some(Some(buffer)) => {
                if let Err(why) = self.absorb_into(Dest::Layer(surface, root), &buffer, shape) {
                    let id = root.map(|s| self.owner_of(s));
                    self.reject(id, why);
                }
            }
            Some(None) => with_states(surface, |states| {
                layer_of(states).lock().unwrap().texture = None
            }),
            None => {}
        }
        if let Some(slot) = root {
            let root = self.surface_of(slot);
            if !self.dirty_trees.contains(&root) {
                self.dirty_trees.push(root);
            }
        }
    }

    /// Draw every tree that changed since the last call. Once per dispatch,
    /// after the client requests in it, so a synchronized tree whose parent
    /// and children all applied in one go is drawn once.
    ///
    /// A tree can also change with no commit to say so — a subsurface
    /// destroyed, or restacked (smithay applies `place_above` at once) — so
    /// every composited tree's shape is re-read here, which is a walk of a
    /// few surfaces, and the tree is redrawn if it moved.
    pub(crate) fn compose_trees(&mut self) {
        let mut roots = std::mem::take(&mut self.dirty_trees);
        let slots = (0..self.tenants.len())
            .map(Slot::Tenant)
            .chain((0..self.popups.len()).map(Slot::Popup));
        for slot in slots {
            let Some(drawn) = self.skin(slot).tree else {
                continue;
            };
            let root = self.surface_of(slot);
            if walk(&root).1 != drawn && !roots.contains(&root) {
                roots.push(root);
            }
        }
        for root in roots {
            if let Some(slot) = self.slot_of_surface(&root) {
                self.compose(slot);
            }
        }
    }

    /// Draw one tree into its root's texture.
    ///
    /// Each surface is drawn at its position and its logical size, in the
    /// root's logical pixels scaled to the texture; a viewport destination or
    /// a buffer scale is in that logical size already, and a viewport source
    /// in the layer's crop. **Whatever falls outside the root's own rectangle
    /// is clipped**: the texture is the root's size, the size the shell draws
    /// the pane at and hit-tests in, and a menu hanging past the window's edge
    /// is cut there. That is a limit, not a design: a client that does it
    /// expects the overhang to be seen.
    fn compose(&mut self, slot: Slot) {
        let root = self.surface_of(slot);
        let (mut pieces, shape) = walk(&root);
        // A surface whose pixels were never taken into a layer is read now,
        // from the buffer it still holds: the root, which drew straight into
        // its texture until it had subsurfaces, and a subsurface that
        // committed before its tree belonged to a tenant.
        for piece in &mut pieces {
            if piece.layer.is_some() {
                continue;
            }
            let Some(buffer) = piece.buffer.clone() else {
                continue;
            };
            match self.absorb_into(
                Dest::Layer(&piece.surface, Some(slot)),
                &buffer,
                piece.shape,
            ) {
                Ok(()) => piece.layer = with_states(&piece.surface, layer_snapshot),
                Err(why) => {
                    let id = self.owner_of(slot);
                    self.reject(Some(id), why);
                }
            }
        }
        self.skin_mut(slot).tree = Some(shape);
        // Nothing to draw over: an unmapped root hides its whole tree, and
        // the texture keeps its last picture, as a surface drawn alone does.
        let Some((base, logical, _)) = pieces
            .iter()
            .find(|p| p.surface == root)
            .and_then(|p| p.layer.clone())
        else {
            return;
        };
        if logical.0 == 0 || logical.1 == 0 {
            return;
        }
        let scale = match slot {
            Slot::Tenant(i) => self.tenants[i].scale,
            Slot::Popup(i) => self
                .index_of(self.popups[i].root)
                .map_or(1.0, |t| self.tenants[t].scale),
        };
        // A toplevel's window is cut from the tree as from a buffer drawn
        // alone (`Shape::frame`): the texture starts at its origin and is its
        // size, at the root layer's own density.
        let window = self.window(slot).and_then(|g| window_in(g, logical));
        let (origin, shown) = window.map_or(((0, 0), logical), |(x, y, w, h)| {
            ((x as i32, y as i32), (w, h))
        });
        let dense = |texels: u32, of: u32, part: u32| {
            (u64::from(texels) * u64::from(part) / u64::from(of)).max(1) as u32
        };
        let base_texels = (
            dense(base.width(), logical.0, shown.0),
            dense(base.height(), logical.1, shown.1),
        );
        let size = composite_size(base_texels, shown, scale);
        // The composite is all texels of its own.
        self.ensure_texture(
            slot,
            Frame {
                crop: (0, 0, size.0, size.1),
                logical: shown,
                origin,
            },
        );
        let target = self
            .skin(slot)
            .texture
            .clone()
            .expect("ensure_texture just set it");
        let (sx, sy) = (
            size.0 as f32 / shown.0 as f32,
            size.1 as f32 / shown.1 as f32,
        );
        let quads: Vec<Quad<'_>> = pieces
            .iter()
            .filter_map(|p| {
                let (texture, (w, h), opaque) = p.layer.as_ref()?;
                Some(Quad {
                    texture,
                    rect: [
                        (p.at.0 - origin.0) as f32 * sx,
                        (p.at.1 - origin.1) as f32 * sy,
                        *w as f32 * sx,
                        *h as f32 * sy,
                    ],
                    opaque: *opaque,
                })
            })
            .collect();
        let compositor = self
            .compositor
            .get_or_insert_with(|| Compositor::new(&self.device));
        compositor.draw(&self.device, &self.queue, &target, &quads);
        self.committed(slot);
    }

    /// The tenant or popup at the root of `surface`'s subsurface tree, if
    /// this crate hosts one.
    fn root_slot_of(&self, surface: &WlSurface) -> Option<Slot> {
        self.slot_of_surface(&root_of(surface))
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

    fn new_subsurface(&mut self, surface: &WlSurface, parent: &WlSurface) {
        // What a toplevel is told, a subsurface is told too: it is drawn at
        // its tree's scale, on the one output there is.
        self.output.enter(surface);
        let scale = match self.slot_of_surface(&root_of(parent)) {
            Some(Slot::Tenant(i)) => self.tenants[i].scale,
            Some(Slot::Popup(i)) => self
                .index_of(self.popups[i].root)
                .map_or(1.0, |t| self.tenants[t].scale),
            None => 1.0,
        };
        ask_scale(surface, scale);
    }

    fn commit(&mut self, surface: &WlSurface) {
        let (buffer, callbacks, shape, role) = with_states(surface, |states| {
            let mut guard = states.cached_state.get::<SurfaceAttributes>();
            let attrs = guard.current();
            // `Some(None)` is a null buffer attached, which unmaps the
            // surface; `None` is no buffer ever.
            let buffer = match &attrs.buffer {
                Some(BufferAssignment::NewBuffer(b)) => Some(Some(b.clone())),
                Some(BufferAssignment::Removed) => Some(None),
                None => None,
            };
            let callbacks = std::mem::take(&mut attrs.frame_callbacks);
            drop(guard);
            (buffer, callbacks, Shape::of(states), states.role)
        });

        let slot = self.slot_of_surface(surface);
        let subsurface = slot.is_none() && role == Some(SUBSURFACE);
        // A subsurface's callbacks pace the tenant whose tree it is in.
        let paced_by = if subsurface {
            self.root_slot_of(surface)
        } else {
            slot
        };
        let owner = paced_by.map(|slot| self.owner_of(slot));
        match slot {
            Some(Slot::Tenant(index)) => {
                self.refresh_identity(index);
                // The initial configure must be answered before a client is
                // allowed to attach: a toplevel that commits without one is a
                // protocol error on our side, not theirs.
                let toplevel = self.tenants[index].toplevel.clone();
                if !toplevel.is_initial_configure_sent() {
                    toplevel.send_configure();
                }
                self.absorb_root(Slot::Tenant(index), surface, buffer, shape);
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
                self.absorb_root(Slot::Popup(index), surface, buffer, shape);
            }
            // A desync subsurface's own commit, or a sync one's applied with
            // its parent's: either way its tree is drawn again.
            None if subsurface => {
                let root = self.root_slot_of(surface);
                self.absorb_layer(surface, root, buffer, shape);
            }
            None => {}
        }

        // Harvested, not answered: [`State::release_frame_callbacks`] answers
        // them, on the caller's clock, filed under the tenant they pace.
        self.pending_callbacks
            .extend(callbacks.into_iter().map(|cb| (owner, cb)));
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
            tile(state);
        });
        toplevel.send_configure();
        self.tenants.push(Tenant {
            id,
            pid,
            title: None,
            app_id: None,
            configured: Some(size),
            scale: 1.0,
            skin: Skin {
                size,
                ..Default::default()
            },
            surface,
            toplevel,
            callbacks_sent: 0,
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
            self.reject(
                None,
                "xdg_popup on a surface this compositor does not host".into(),
            );
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
        // And at the scale its tenant was asked for, before it paints its
        // first buffer: a menu drawn at 1 over a pane drawn at 2 is the one
        // blurry thing on screen.
        let scale = self.index_of(root).map_or(1.0, |i| self.tenants[i].scale);
        if scale != 1.0 {
            ask_scale(&wl_surface, scale);
        }
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

/// Mark a toplevel tiled on all four edges.
///
/// A configured size is otherwise only an upper bound: an `xdg_toplevel` that
/// believes it is floating may pick something smaller, and foot does — it
/// rounds down to whole cells, and on a scale change it keeps its grid of
/// cells and resizes itself around the new font. Every tenant here is a pane
/// whose size the shell decides, which is what "tiled" says, and a client
/// told so takes the size it is given.
pub(crate) fn tile(state: &mut smithay::wayland::shell::xdg::ToplevelState) {
    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel::State as S;
    for edge in [S::TiledLeft, S::TiledRight, S::TiledTop, S::TiledBottom] {
        state.states.set(edge);
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

/// The same bare glass for a client that asks the KDE way: GTK4 has no
/// xdg-decoration, so a GTK app that would rather not draw its own frame —
/// Ghostty, by `window-decoration` — asks by this protocol or not at all. The
/// manager's default says server-side, and every request is answered with
/// it, whatever mode it asked for.
impl KdeDecorationHandler for State {
    fn kde_decoration_state(&self) -> &KdeDecorationState {
        &self.kde_decoration_state
    }

    fn request_mode(
        &mut self,
        _surface: &WlSurface,
        decoration: &OrgKdeKwinServerDecoration,
        _mode: smithay::reexports::wayland_server::WEnum<KdeMode>,
    ) {
        decoration.mode(KdeMode::Server);
    }
}

smithay::delegate_kde_decoration!(State);

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
        set_data_device_focus(&dh, seat, client.clone());
        set_primary_focus(&dh, seat, client);
    }

    /// What the focused client wants the pointer to look like.
    ///
    /// glasspane draws nothing, so this is dropped. A shell that wants to
    /// honour a client cursor reads it from the seat itself; wrapping it here
    /// would be scene policy.
    fn cursor_image(
        &mut self,
        _seat: &Seat<Self>,
        _image: smithay::input::pointer::CursorImageStatus,
    ) {
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

impl PrimarySelectionHandler for State {
    fn primary_selection_state(&self) -> &PrimarySelectionState {
        &self.primary_selection_state
    }
}

impl smithay::wayland::selection::data_device::ClientDndGrabHandler for State {}
impl ServerDndGrabHandler for State {}

impl smithay::wayland::output::OutputHandler for State {}

impl FractionalScaleHandler for State {}

smithay::delegate_compositor!(State);
smithay::delegate_fractional_scale!(State);
smithay::delegate_viewporter!(State);
smithay::delegate_shm!(State);
smithay::delegate_dmabuf!(State);
smithay::delegate_xdg_shell!(State);
smithay::delegate_xdg_decoration!(State);
smithay::delegate_seat!(State);
smithay::delegate_data_device!(State);
smithay::delegate_primary_selection!(State);
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
    /// Chromium shrunk to 1434x551 keeps its 1920x1200 buffer and crops it:
    /// the texture is the cropped corner, not the whole buffer squeezed.
    #[test]
    fn a_viewport_source_is_the_part_of_the_buffer_drawn() {
        let cropped = super::Shape {
            buffer_scale: 1,
            viewport: Some((1434, 551)),
            source: Some((0.0, 0.0, 1434.0, 551.0)),
        };
        assert_eq!(cropped.crop((1920, 1200)), (0, 0, 1434, 551));
        assert_eq!(cropped.logical((1434, 551)), (1434, 551));
        // In logical pixels: at buffer scale 2 it is twice as many texels.
        let scaled = super::Shape {
            buffer_scale: 2,
            source: Some((10.0, 5.0, 100.0, 50.0)),
            ..cropped
        };
        assert_eq!(scaled.crop((1920, 1200)), (20, 10, 200, 100));
        // Kept inside the buffer; a source wholly off it is no crop.
        let over = super::Shape {
            source: Some((1800.0, 0.0, 400.0, 100.0)),
            ..cropped
        };
        assert_eq!(over.crop((1920, 1200)), (1800, 0, 120, 100));
        let off = super::Shape {
            source: Some((2000.0, 0.0, 10.0, 10.0)),
            ..cropped
        };
        assert_eq!(off.crop((1920, 1200)), (0, 0, 1920, 1200));
        let none = super::Shape {
            source: None,
            ..cropped
        };
        assert_eq!(none.crop((1920, 1200)), (0, 0, 1920, 1200));
    }

    #[test]
    fn a_viewport_destination_is_the_logical_size() {
        let shape = super::Shape {
            buffer_scale: 1,
            viewport: Some((390, 520)),
            source: None,
        };
        assert_eq!(shape.logical((819, 1092)), (390, 520));
    }

    #[test]
    fn without_a_viewport_the_buffer_is_divided_by_its_scale() {
        let shape = super::Shape {
            buffer_scale: 2,
            viewport: None,
            source: None,
        };
        assert_eq!(shape.logical((780, 1040)), (390, 520));
        let shape = super::Shape {
            buffer_scale: 1,
            viewport: None,
            source: None,
        };
        assert_eq!(shape.logical((390, 520)), (390, 520));
    }

    #[test]
    fn a_nonsense_buffer_scale_is_read_as_one() {
        let shape = super::Shape {
            buffer_scale: 0,
            viewport: None,
            source: None,
        };
        assert_eq!(shape.logical((390, 520)), (390, 520));
    }

    /// The window is cut from the picture in its logical pixels, whatever the
    /// buffer's density, and a geometry past the surface is clamped to it.
    #[test]
    fn a_window_geometry_is_the_part_of_the_surface_drawn() {
        use super::Frame;
        let plain = super::Shape {
            buffer_scale: 1,
            viewport: None,
            source: None,
        };
        let window = Some((20, 20, 1280, 800));
        assert_eq!(
            plain.frame((1320, 840), window),
            Frame {
                crop: (20, 20, 1280, 800),
                logical: (1280, 800),
                origin: (20, 20)
            }
        );
        // At buffer scale 2 the margin is twice as many texels.
        let scaled = super::Shape {
            buffer_scale: 2,
            ..plain
        };
        assert_eq!(
            scaled.frame((2640, 1680), window),
            Frame {
                crop: (40, 40, 2560, 1600),
                logical: (1280, 800),
                origin: (20, 20)
            }
        );
        // A fractional client: 1.5x buffer, the viewport saying its logical size.
        let fractional = super::Shape {
            viewport: Some((1320, 840)),
            ..plain
        };
        assert_eq!(
            fractional.frame((1980, 1260), window),
            Frame {
                crop: (30, 30, 1920, 1200),
                logical: (1280, 800),
                origin: (20, 20)
            }
        );
        // Hanging off the top-left: the part inside the surface.
        assert_eq!(
            plain.frame((1320, 840), Some((-10, -10, 100, 100))),
            Frame {
                crop: (0, 0, 90, 90),
                logical: (90, 90),
                origin: (0, 0)
            }
        );
        // The whole surface, or none of it, is no cut.
        let whole = Frame {
            crop: (0, 0, 1320, 840),
            logical: (1320, 840),
            origin: (0, 0),
        };
        assert_eq!(plain.frame((1320, 840), Some((0, 0, 1320, 840))), whole);
        assert_eq!(plain.frame((1320, 840), Some((2000, 0, 10, 10))), whole);
        assert_eq!(plain.frame((1320, 840), None), whole);
    }

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
        assert_eq!(
            (flat.loc.x, flat.loc.y),
            (100, 671),
            "it should fit as it is"
        );
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

    /// Waydroid's root is a 1x1 buffer stretched over the window: the
    /// composite is the window's size, not the buffer's.
    #[test]
    fn a_token_root_composites_at_its_logical_size() {
        assert_eq!(super::composite_size((1, 1), (1280, 800), 1.0), (1280, 800));
        assert_eq!(
            super::composite_size((1, 1), (1280, 800), 1.5),
            (1920, 1200)
        );
    }

    /// A root that draws denser than it was asked keeps every pixel it drew;
    /// one asked for more than it drew is drawn up to what was asked, so its
    /// subsurfaces are not held to its resolution.
    #[test]
    fn a_composite_is_the_denser_of_buffer_and_scale() {
        assert_eq!(
            super::composite_size((512, 512), (256, 256), 1.0),
            (512, 512)
        );
        assert_eq!(
            super::composite_size((256, 256), (256, 256), 2.0),
            (512, 512)
        );
        assert_eq!(
            super::composite_size((300, 100), (200, 200), 1.0),
            (300, 200)
        );
    }
}
