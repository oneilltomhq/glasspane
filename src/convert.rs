//! smithay's description of a client dma-buf, restated as [`wgpu_dmabuf`]'s.
//!
//! Two crates, one `drm-fourcc`: both depend on `drm-fourcc = "2"`, so the
//! `Fourcc` and `Modifier` values cross this boundary as the same types and
//! nothing is reinterpreted. What this function actually does is dup the plane
//! fds (the client keeps its own) and refuse the descriptions this crate
//! cannot honour.

use std::os::fd::AsFd;

use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::Buffer as _;

use crate::texture::ACCEPTED_FOURCC;

/// Describe a client dma-buf in the terms [`wgpu_dmabuf::import`] wants.
///
/// Refuses, with a reason a client author could act on:
///
/// * **multi-plane buffers.** `texture_from_dmabuf_fd` is single-plane, and
///   the multi-plane route needs a `VkSamplerYcbcrConversion` that has no
///   representation in wgpu's texture model. Refusing beats importing the
///   luma plane and calling it a picture.
/// * **any fourcc but [`ACCEPTED_FOURCC`]**, so that every tenant texture has
///   the same `wgpu::TextureFormat`.
pub(crate) fn describe(dmabuf: &Dmabuf) -> Result<wgpu_dmabuf::Dmabuf, String> {
    let format = dmabuf.format();
    if !ACCEPTED_FOURCC.contains(&format.code) {
        return Err(format!(
            "fourcc {:?} is not one of {:?}",
            format.code, ACCEPTED_FOURCC
        ));
    }
    if dmabuf.num_planes() != 1 {
        return Err(format!(
            "{} planes; glasspane imports single-plane buffers only",
            dmabuf.num_planes()
        ));
    }
    let fd = dmabuf
        .handles()
        .next()
        .ok_or_else(|| "dmabuf has no plane handle".to_string())?
        .as_fd()
        .try_clone_to_owned()
        .map_err(|e| format!("dup the client's plane fd: {e}"))?;
    let offset = dmabuf.offsets().next().unwrap_or(0);
    let stride = dmabuf
        .strides()
        .next()
        .ok_or_else(|| "dmabuf has no stride".to_string())?;
    Ok(wgpu_dmabuf::Dmabuf::single_plane(
        dmabuf.width(),
        dmabuf.height(),
        format.code,
        u64::from(format.modifier),
        fd,
        offset,
        stride,
    ))
}
