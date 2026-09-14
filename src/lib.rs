//! **A tenant Wayland compositor as a library.** It speaks Wayland to real
//! clients — terminals, browsers — and its entire output is one
//! [`wgpu::Texture`] per client plus an input-injection API.
//!
//! ```text
//!   socket ─ wl_compositor + xdg_shell + zwp_linux_dmabuf_v1 + wl_shm
//!          └─ commit ─ import-once ─ blit ─▶ Tenant::texture   (your wgpu::Device)
//!   tick() ───────────────────────────────▶ wl_callback.done ─ client paints again
//!   pointer_motion / keyboard_key ────────▶ wl_pointer / wl_keyboard
//! ```
//!
//! # What it is not
//!
//! No placement policy: glasspane never decides where a tenant is, how large
//! it is drawn, or what is in front of what. No scene graph, no renderer, no
//! window. No scanout: nothing here talks to KMS. The shell owns the
//! `wgpu::Device`, the camera, the frame rate and the screen; glasspane
//! borrows the first and is told about the third.
//!
//! There is one mode, and it is headless. A caller that wants a window makes
//! one.
//!
//! # Three rules that are load-bearing
//!
//! **The frame clock is yours to drive.** A Wayland client paints *only* when
//! handed a frame callback. A compositor that answers callbacks "on demand"
//! collapses to "when someone asks", and its tenants stall — measured
//! elsewhere at 0.3–11 s per frame, and stopping entirely when nothing was
//! watching. So glasspane never answers a callback by itself: [`Glasspane::tick`]
//! does, and you call it once per frame of *your* clock, whether or not
//! anything looks like it needs it.
//!
//! **Import once per `wl_buffer`.** A client that re-commits the same buffer —
//! every double-buffered client, forever — must produce exactly one dma-buf
//! import and then cache hits. See [`registry`].
//!
//! **The texture handle is stable while the size is.** Each tenant owns one
//! texture; commits are copied into it. It is replaced only when the client's
//! buffer size changes, which is announced as [`Event::Resized`] before any
//! [`Event::Committed`] against the new handle. Index tenants by [`TenantId`],
//! sample their textures, and never rebind per frame.
//!
//! # Coordinates
//!
//! Input is in tenant-local buffer pixels, origin top-left. See [`input`].
//!
//! # A whole session
//!
//! ```no_run
//! use std::time::{Duration, Instant};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
//!     backends: wgpu::Backends::VULKAN,
//!     ..Default::default()
//! });
//! let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
//! // Stock `request_device` cannot import an acquire fence; this can.
//! let (device, queue, _caps) =
//!     wgpu_dmabuf::device::open_device(&adapter, &Default::default())?;
//!
//! let mut pane = glasspane::Glasspane::new(
//!     device,
//!     queue,
//!     glasspane::Caps::new().with_adapter(&adapter).with_size(1280, 800),
//! )?;
//! let _child = pane.spawn(std::process::Command::new("foot"))?;
//!
//! let start = Instant::now();
//! while start.elapsed() < Duration::from_secs(2) {
//!     pane.dispatch(Some(Duration::from_millis(4)))?;
//!     pane.tick(Instant::now());
//!     for event in pane.events() {
//!         println!("{event:?}");
//!     }
//!     for tenant in pane.tenants() {
//!         if let Some(tex) = tenant.texture() {
//!             let _ = tex; // sample it in your scene
//!         }
//!     }
//! }
//! # Ok(()) }
//! ```

#![cfg(target_os = "linux")]
#![deny(missing_debug_implementations)]
#![warn(missing_docs)]
#![doc(html_root_url = "https://docs.rs/glasspane/0.1.0")]

pub mod error;
pub mod tenant;
pub mod event;
pub mod registry;

pub use error::{Error, Result};
pub use event::Event;
pub use tenant::{Tenant, TenantId};

/// The smithay this crate is built against, re-exported so a caller naming
/// `ToplevelSurface` or `calloop::EventLoop` names *the* smithay rather than a
/// semver-compatible second copy.
pub use smithay;
