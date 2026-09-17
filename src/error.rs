//! One error type for everything this crate can refuse to do.

use std::fmt;

/// The result of a glasspane operation.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything that can go wrong between a client's socket and a
/// [`wgpu::Texture`].
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The listening socket could not be created, or the event loop could not
    /// be built around it.
    Socket(String),
    /// A `wgpu::Device` this crate cannot use. dma-buf interop is a Vulkan
    /// concept; see [`wgpu_dmabuf`].
    Device(String),
    /// dma-buf import, export, or fencing failed.
    Dmabuf(wgpu_dmabuf::Error),
    /// The caller named a [`crate::TenantId`] that is not (or is no longer) a
    /// live tenant.
    NoSuchTenant(crate::TenantId),
    /// The caller named a [`crate::PopupId`] that is not (or is no longer) a
    /// live popup. Routine rather than exceptional: a popup can be dismissed
    /// between the frame a shell hit-tested and the input it sends.
    NoSuchPopup(crate::PopupId),
    /// A client committed something this compositor refused. The client is
    /// still connected; the commit was dropped and recorded.
    Rejected(String),
    /// A child process could not be spawned on this compositor's socket.
    Spawn(std::io::Error),
    /// The calloop dispatch itself failed.
    Dispatch(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Socket(s) => write!(f, "wayland socket: {s}"),
            Error::Device(s) => write!(f, "wgpu device: {s}"),
            Error::Dmabuf(e) => write!(f, "dma-buf: {e}"),
            Error::NoSuchTenant(id) => write!(f, "no such tenant: {id}"),
            Error::NoSuchPopup(id) => write!(f, "no such popup: {id}"),
            Error::Rejected(s) => write!(f, "commit rejected: {s}"),
            Error::Spawn(e) => write!(f, "spawn: {e}"),
            Error::Dispatch(e) => write!(f, "dispatch: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Dmabuf(e) => Some(e),
            Error::Spawn(e) => Some(e),
            _ => None,
        }
    }
}

impl From<wgpu_dmabuf::Error> for Error {
    fn from(e: wgpu_dmabuf::Error) -> Self {
        Error::Dmabuf(e)
    }
}
