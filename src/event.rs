//! What the compositor tells the shell, and nothing more.
//!
//! Events are *facts about tenants*, never requests for the shell to do
//! something. There is no `NeedsRedraw` and no `Damage`: this crate does not
//! know what a frame is, and a shell that renders at its own rate reads
//! [`Event::Committed`]'s generation to decide whether a texture changed since
//! it last looked.

use crate::popup::{Parent, PopupId};
use crate::tenant::TenantId;

/// One thing that happened to one tenant.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Event {
    /// A client's toplevel became a tenant. It has an id and a size, but not
    /// necessarily a texture yet — that arrives with the first
    /// [`Event::Committed`].
    Mapped {
        /// The new tenant.
        id: TenantId,
        /// The process id of the client that owns it, read from the socket's
        /// peer credentials. This is how a caller correlates a tenant with the
        /// [`std::process::Child`] it got back from [`crate::Glasspane::spawn`]
        /// — a client may map several toplevels, and a toplevel may arrive
        /// seconds after the process did.
        pid: Option<u32>,
    },
    /// The tenant is gone: its toplevel was destroyed or its client
    /// disconnected. The [`TenantId`] is never reused.
    Unmapped {
        /// The tenant that left.
        id: TenantId,
    },
    /// `xdg_toplevel.set_title`.
    TitleChanged {
        /// The tenant.
        id: TenantId,
        /// The new title.
        title: String,
    },
    /// `xdg_toplevel.set_app_id`.
    AppIdChanged {
        /// The tenant.
        id: TenantId,
        /// The new app id.
        app_id: String,
    },
    /// New pixels landed in this tenant's texture.
    ///
    /// `generation` is the tenant's commit counter *after* this commit. The
    /// texture handle is unchanged unless a [`Event::Resized`] was emitted for
    /// the same tenant first.
    Committed {
        /// The tenant.
        id: TenantId,
        /// [`crate::Tenant::generation`] as of this commit.
        generation: u64,
    },
    /// The tenant's size changed: its logical size, its buffer's
    /// ([`crate::Tenant::buffer_size`]), or both.
    ///
    /// When the buffer's changed the texture was recreated, and any bind
    /// group, view or copy descriptor the caller derived from the old handle
    /// is stale as of this event and *only* as of this event. A tenant
    /// repainting at a new scale fires this with the same `size`.
    Resized {
        /// The tenant.
        id: TenantId,
        /// The new logical size: the space input and popups are in. See
        /// [`crate::Tenant::size`].
        size: (u32, u32),
    },
    /// An `xdg_popup` became a popup over one of this compositor's tenants.
    ///
    /// It has an id, a parent and a place; like a tenant it has no texture
    /// until its first [`Event::PopupCommitted`]. A shell that draws panes
    /// draws this as a second quad hung off the parent pane at `position`.
    PopupMapped {
        /// The new popup.
        id: PopupId,
        /// Its immediate parent: a tenant, or another popup for a submenu.
        parent: Parent,
        /// The tenant at the top of the chain — the pane it belongs over.
        root: TenantId,
        /// Its top-left corner in `root`'s logical pixels. See
        /// [`crate::Popup::position`].
        position: (i32, i32),
        /// The size its positioner asked for, in logical pixels. The size that
        /// arrives with the first commit is the one to believe.
        size: (u32, u32),
    },
    /// The popup asked for an explicit grab and was given one: it is modal
    /// over its root tenant until it is dismissed. See
    /// [`crate::Popup::has_grab`].
    PopupGrabbed {
        /// The popup.
        id: PopupId,
    },
    /// The popup was repositioned — `xdg_popup.reposition`, or its parent
    /// moved under it.
    PopupMoved {
        /// The popup.
        id: PopupId,
        /// Its new top-left in its root tenant's logical pixels.
        position: (i32, i32),
        /// The size its new positioner asks for.
        size: (u32, u32),
    },
    /// New pixels landed in this popup's texture. The popup analogue of
    /// [`Event::Committed`].
    PopupCommitted {
        /// The popup.
        id: PopupId,
        /// [`crate::Popup::generation`] as of this commit.
        generation: u64,
    },
    /// The popup's size changed. The popup analogue of [`Event::Resized`],
    /// with the same staleness rule.
    PopupResized {
        /// The popup.
        id: PopupId,
        /// The new logical size. The buffer's is
        /// [`crate::Popup::buffer_size`].
        size: (u32, u32),
    },
    /// The popup is gone: dismissed, destroyed by its client, or taken down
    /// with the tenant it belonged to. The [`PopupId`] is never reused.
    PopupUnmapped {
        /// The popup that left.
        id: PopupId,
    },
    /// A tenant copied or cleared a selection, over its data device or over
    /// data control.
    ///
    /// Only which selection: the mime types and the source stay with the
    /// seat. A shell can show a clipboard indicator from it. It fires once
    /// per set, never per frame. A selection that goes because the tenant
    /// that owned it quit fires nothing; the next paste is told it is empty.
    /// Nor does the desktop's clipboard arriving through
    /// [`crate::Glasspane::bridge_clipboard`]: that is no tenant's copy.
    SelectionChanged {
        /// The clipboard or the primary selection.
        selection: Selection,
    },
    /// A commit was refused and dropped. The client is still connected.
    ///
    /// Loud on purpose: the alternative is a tenant that keeps compositing at
    /// full rate while showing the past.
    Rejected {
        /// The tenant whose commit was dropped, if it was identifiable.
        id: Option<TenantId>,
        /// Why.
        why: String,
    },
}

/// Which of the seat's two selections an [`Event::SelectionChanged`] is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Selection {
    /// The clipboard: copy and paste, `wl_data_device`.
    Clipboard,
    /// The primary selection: select and middle-click,
    /// `zwp_primary_selection_device_v1`.
    Primary,
}
