//! What a tenant *is*, from the shell's side of the seam.
//!
//! A tenant is one `xdg_toplevel`. It has an identity that outlives any
//! particular buffer, a texture the shell can sample, and a generation counter
//! that says whether the pixels moved. It has no position, no z-order, no
//! parent and no camera: placing it is the shell's job and this crate has no
//! opinion about it.

pub mod browser;

use std::sync::Arc;


use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::wayland::shell::xdg::ToplevelSurface;

use crate::texture::Skin;

/// A tenant's identity, stable for its whole life and never reused within a
/// run.
///
/// Index by this, never by position in [`crate::Glasspane::tenants`]: that
/// order is not insertion order and a "the last one is the new one" reading
/// produces real pixels under the wrong label, which is worse than an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TenantId(pub u64);

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tenant#{}", self.0)
    }
}

/// One hosted Wayland client surface, as the shell sees it.
#[derive(Debug)]
pub struct Tenant {
    pub(crate) id: TenantId,
    pub(crate) pid: Option<u32>,
    pub(crate) title: Option<String>,
    pub(crate) app_id: Option<String>,
    pub(crate) configured: Option<(u32, u32)>,
    /// The scale last asked of this tenant through
    /// [`crate::Glasspane::set_scale`]; its popups are asked the same.
    pub(crate) scale: f64,
    pub(crate) skin: Skin,
    pub(crate) surface: WlSurface,
    pub(crate) toplevel: ToplevelSurface,
    /// Frame callbacks answered for this tenant and its popups.
    pub(crate) callbacks_sent: u64,
}

impl Tenant {
    /// This tenant's identity.
    pub fn id(&self) -> TenantId {
        self.id
    }

    /// The client process, from the connection's peer credentials.
    pub fn pid(&self) -> Option<u32> {
        self.pid
    }

    /// `xdg_toplevel.set_title`, if the client has sent one.
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// `xdg_toplevel.set_app_id`, if the client has sent one.
    pub fn app_id(&self) -> Option<&str> {
        self.app_id.as_deref()
    }

    /// The tenant's logical size: the size it was configured in, the space
    /// input coordinates are expressed in, and the space its popups are
    /// positioned in.
    ///
    /// For a client drawing at scale 1 this is also its buffer's size. For one
    /// drawing at a higher scale (see [`crate::Glasspane::set_scale`]) the
    /// buffer, and so the texture, is larger: [`Tenant::buffer_size`]. A shell
    /// places and hit-tests by this one and samples the texture by UV, and
    /// never needs the other.
    ///
    /// Before the first commit this is the size this tenant was configured
    /// with, which the client may not have honoured yet.
    pub fn size(&self) -> (u32, u32) {
        self.skin.size
    }

    /// The texture's size in pixels: the last committed buffer, cropped to
    /// its viewport's source if it sets one, and then to its window if it
    /// sets `xdg_surface.set_window_geometry` inside the surface (a shadow
    /// round the window is not drawn). `(0, 0)` before the first commit.
    ///
    /// For a surface with subsurfaces the texture is the whole tree drawn
    /// together and cut to the window the same way, and this is that
    /// picture's size: the window's logical size at the denser of the
    /// surface's own buffer and the scale it was asked for.
    pub fn buffer_size(&self) -> (u32, u32) {
        self.skin.buffer
    }

    /// The scale this tenant was last asked to draw at, `1.0` until
    /// [`crate::Glasspane::set_scale`] says otherwise. What the client
    /// actually drew at is [`Tenant::buffer_size`] over [`Tenant::size`].
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The size this tenant was last told to be, via
    /// [`crate::Glasspane::configure`]. A resize is a negotiation — configure,
    /// ack, commit — so this and [`Tenant::size`] disagree until the client
    /// has painted, and a starved client keeps its old rect forever.
    pub fn configured(&self) -> Option<(u32, u32)> {
        self.configured
    }

    /// The texture holding this tenant's pixels, or `None` before its first
    /// commit.
    ///
    /// **The handle is stable while [`Tenant::buffer_size`] is unchanged.** Every
    /// commit lands in the same texture; the handle changes only across an
    /// [`crate::Event::Resized`]. Index it, do not rebind per frame.
    pub fn texture(&self) -> Option<&wgpu::Texture> {
        self.skin.texture.as_deref()
    }

    /// The texture as a shared handle, for a caller that wants to keep it
    /// alive past the next `&mut Glasspane` call.
    pub fn texture_arc(&self) -> Option<&Arc<wgpu::Texture>> {
        self.skin.texture.as_ref()
    }

    /// How many commits have landed in this texture. Bumps on every commit
    /// that carried a buffer, including one that re-committed the same
    /// `wl_buffer`.
    pub fn generation(&self) -> u64 {
        self.skin.generation
    }

    /// Frame callbacks answered for this tenant, its popups' included, over
    /// its life: how often it has been let paint. See
    /// [`crate::Glasspane::tick_where`] for a shell that paces tenants apart.
    pub fn callbacks_sent(&self) -> u64 {
        self.callbacks_sent
    }

    /// Whether the last dma-buf commit's acquire fence was staged as a wait on
    /// the caller's queue.
    ///
    /// glasspane does the waiting itself — the fence has to be on the queue
    /// before the shell's next submit, and the shell does not know when a
    /// commit happened. This reports that it was done, so a harness can fail
    /// on a run that sampled client memory with nothing ordering it. Always
    /// `false` for an `wl_shm` tenant, which has no GPU work to order.
    pub fn acquire_staged(&self) -> bool {
        self.skin.acquire_staged
    }

    /// The underlying `wl_surface`, for a caller that needs to reach protocol
    /// this crate does not wrap.
    pub fn surface(&self) -> &WlSurface {
        &self.surface
    }

    /// The underlying `xdg_toplevel`.
    pub fn toplevel(&self) -> &ToplevelSurface {
        &self.toplevel
    }
}
