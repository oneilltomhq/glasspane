//! Client buffers become texels.
//!
//! Two paths, one destination. A dma-buf is imported once per `wl_buffer`
//! (see [`crate::registry`]) and then **copied** into the tenant's own
//! texture; an `wl_shm` buffer is written straight into it. The copy is what
//! makes both of this crate's laws hold at the same time: import-once per
//! buffer, *and* one stable texture handle per tenant even though a
//! double-buffered client alternates between two `wl_buffer`s forever.
//!
//! The cost is one GPU blit per dma-buf commit. The alternative — handing the
//! shell whichever imported texture the client last attached — makes the
//! shell rebind a bind group every time the client flips buffers, and makes
//! "the texture for tenant N" a thing that is only true until the next
//! dispatch.

use wgpu_dmabuf::format::Fourcc;

/// The format every tenant texture has.
///
/// `Bgra8Unorm` is the wgpu spelling of `ARGB8888`/`XRGB8888`, which are the
/// only two formats this crate advertises. One format for every tenant means
/// the shell can write one shader and one bind group layout.
///
/// Note that `XRGB8888` maps here too: there is no ignore-the-alpha `UNORM`
/// format in Vulkan, so an `XRGB8888` client's alpha byte *is* sampled. A
/// shell that wants those tenants opaque must ignore alpha itself.
pub const TENANT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// The usages a tenant texture is created with.
///
/// `COPY_DST` for both upload paths, `COPY_SRC` so a harness (or a screenshot)
/// can read it back, `TEXTURE_BINDING` because sampling it is the point, and
/// `RENDER_ATTACHMENT` so a shell can render *into* a tenant texture if it
/// wants to (a placeholder, a dimmed copy) without a second allocation.
pub const TENANT_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::COPY_DST
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::TEXTURE_BINDING)
    .union(wgpu::TextureUsages::RENDER_ATTACHMENT);

/// The usages an imported client dma-buf is given.
///
/// Read-only: glasspane copies out of it and never into it. Writing into a
/// client's buffer would be writing into memory the client still owns.
pub const IMPORT_USAGE: wgpu::TextureUsages =
    wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::COPY_SRC);

/// The `wl_shm`/DRM formats this crate accepts, in advertisement order.
///
/// Deliberately two. Every other fourcc either maps to a different
/// `wgpu::TextureFormat` — which would make `Tenant::texture` a different type
/// per tenant — or is multi-planar, which [`wgpu_dmabuf`] does not import. A
/// client asking for anything else is refused at `wl_buffer` creation, where
/// it can still choose differently, rather than at commit.
pub const ACCEPTED_FOURCC: [Fourcc; 2] = [Fourcc::Argb8888, Fourcc::Xrgb8888];

/// A tenant's stable texture.
pub(crate) fn create_tenant_texture(
    device: &wgpu::Device,
    id: crate::TenantId,
    width: u32,
    height: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("glasspane tenant#{}", id.0)),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TENANT_FORMAT,
        usage: TENANT_USAGE,
        view_formats: &[],
    })
}

/// Upload an `wl_shm` pool's bytes into a tenant texture.
///
/// The whole buffer, every commit. glasspane does not yet track damage: a
/// partial upload that is wrong by one rectangle produces a tenant that looks
/// right and is stale in a corner, and that failure is not worth having until
/// there is a measurement saying the full upload costs something.
pub(crate) fn upload_shm(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    bytes: &[u8],
    stride: u32,
    width: u32,
    height: u32,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(stride),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Copy an imported client image into the tenant texture.
///
/// The submit this issues is the one that carries the acquire wait staged by
/// [`crate::acquire`], which is why the wait is staged at the top of the
/// commit and not later.
pub(crate) fn blit(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Texture,
    dst: &wgpu::Texture,
    width: u32,
    height: u32,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("glasspane tenant blit"),
    });
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: dst,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
}
