//! What a popup *is*, from the shell's side of the seam.
//!
//! A popup is an `xdg_popup`: a second surface belonging to a tenant, with a
//! size and a position its client asked for through an `xdg_positioner`. A
//! `<select>` menu, an autocomplete list, a context menu and a permission
//! bubble are all one of these.
//!
//! It is handed over exactly as a tenant is — its own id, its own texture, its
//! own generation — plus the two facts a toplevel does not have: **whose** it
//! is, and **where** relative to that parent. Everything after that is the
//! shell's: how far off the pane's plane to lift it, what to do when the pane
//! is edge-on, whether it is in the seam. This crate resolves the positioner
//! and stops.
//!
//! # The position
//!
//! [`Popup::position`] is the popup's top-left corner **in its root tenant's
//! logical pixels**, not in its immediate parent's. A submenu's parent is
//! another popup, and a shell that drew it relative to that popup would have
//! to walk the chain itself to find a quad to hang it off; the walk is done
//! here instead, once, when the popup is created or repositioned. So
//! `position` is always in the same space as [`crate::Tenant::size`] and as
//! the input contract — which is also the space the shell's own hit test
//! already produces.

use std::sync::Arc;

use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::wayland::shell::xdg::PopupSurface;

use crate::tenant::TenantId;
use crate::texture::Skin;

/// A popup's identity, stable for its whole life and never reused within a
/// run.
///
/// A separate space from [`TenantId`]: a popup is not a tenant, does not
/// appear in [`crate::Glasspane::tenants`], and a shell that keeps a row of
/// panes must not find a `<select>` menu in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PopupId(pub u64);

impl std::fmt::Display for PopupId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "popup#{}", self.0)
    }
}

/// Whose child a popup is: a tenant, or another popup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parent {
    /// A popup on a toplevel: a `<select>` menu on a page.
    Tenant(TenantId),
    /// A popup on a popup: a submenu.
    Popup(PopupId),
}

/// One `xdg_popup`, as the shell sees it.
#[derive(Debug)]
pub struct Popup {
    pub(crate) id: PopupId,
    pub(crate) parent: Parent,
    pub(crate) root: TenantId,
    /// The popup's *window geometry* origin in root logical pixels: what a
    /// child popup's positioner is resolved against.
    pub(crate) anchor: (i32, i32),
    /// The popup's *buffer* top-left in root logical pixels, which is
    /// [`Popup::anchor`] less whatever offset the client's own window
    /// geometry has. What the shell draws at.
    pub(crate) position: (i32, i32),
    pub(crate) grabbed: bool,
    pub(crate) skin: Skin,
    pub(crate) surface: WlSurface,
    pub(crate) popup: PopupSurface,
}

impl Popup {
    /// This popup's identity.
    pub fn id(&self) -> PopupId {
        self.id
    }

    /// Whose child it is — which for a submenu is another popup.
    pub fn parent(&self) -> Parent {
        self.parent
    }

    /// The tenant at the top of the chain: the pane this popup belongs over.
    pub fn root(&self) -> TenantId {
        self.root
    }

    /// The popup's top-left corner in [`Popup::root`]'s logical pixels. See the
    /// module docs: this is resolved against the whole parent chain, so it is
    /// always relative to the tenant and never to an intermediate popup.
    ///
    /// It may be negative, and it may run past the tenant's size: this crate
    /// constrains a popup to its root tenant's rectangle where the client's
    /// positioner allows it to (`flip`, `slide`, `resize`), and does not
    /// clamp where it does not. A pane in a scene is not an output, and a
    /// popup that hangs over the edge of one is the shell's to draw or to cut.
    pub fn position(&self) -> (i32, i32) {
        self.position
    }

    /// The popup's logical size: the space input coordinates for this popup
    /// are in, and the size to draw it at beside its root tenant's
    /// [`crate::Tenant::size`]. See that for how it differs from the buffer.
    pub fn size(&self) -> (u32, u32) {
        self.skin.size
    }

    /// The size of the last committed buffer in pixels, cropped to its
    /// viewport's source if it sets one: the texture's size. `(0, 0)` before
    /// the first commit.
    pub fn buffer_size(&self) -> (u32, u32) {
        self.skin.buffer
    }

    /// Whether this popup asked for — and was given — an explicit grab.
    ///
    /// A grabbing popup is modal over its root tenant: while it is up, a
    /// pointer button pressed anywhere that is not inside the grab's own chain
    /// dismisses it (`xdg_popup.popup_done`) instead of reaching what is under
    /// the cursor. That is what stops a `<select>` menu hanging when the human
    /// clicks the page behind it.
    pub fn has_grab(&self) -> bool {
        self.grabbed
    }

    /// The texture holding this popup's pixels, or `None` before its first
    /// commit. Same contract as [`crate::Tenant::texture`]: the handle is
    /// stable while [`Popup::size`] is.
    pub fn texture(&self) -> Option<&wgpu::Texture> {
        self.skin.texture.as_deref()
    }

    /// The texture as a shared handle.
    pub fn texture_arc(&self) -> Option<&Arc<wgpu::Texture>> {
        self.skin.texture.as_ref()
    }

    /// How many commits have landed in this texture.
    pub fn generation(&self) -> u64 {
        self.skin.generation
    }

    /// Whether the last dma-buf commit's acquire fence was staged as a wait on
    /// the caller's queue. See [`crate::Tenant::acquire_staged`].
    pub fn acquire_staged(&self) -> bool {
        self.skin.acquire_staged
    }

    /// The underlying `wl_surface`.
    pub fn surface(&self) -> &WlSurface {
        &self.surface
    }

    /// The underlying `xdg_popup`.
    pub fn popup_surface(&self) -> &PopupSurface {
        &self.popup
    }
}
