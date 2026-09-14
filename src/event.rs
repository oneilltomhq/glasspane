//! What the compositor tells the shell, and nothing more.
//!
//! Events are *facts about tenants*, never requests for the shell to do
//! something. There is no `NeedsRedraw` and no `Damage`: this crate does not
//! know what a frame is, and a shell that renders at its own rate reads
//! [`Event::Committed`]'s generation to decide whether a texture changed since
//! it last looked.

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
    /// The tenant's buffer size changed, so its texture was recreated. Any
    /// bind group, view or copy descriptor the caller derived from the old
    /// handle is stale as of this event and *only* as of this event.
    Resized {
        /// The tenant.
        id: TenantId,
        /// The new size in buffer pixels.
        size: (u32, u32),
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
