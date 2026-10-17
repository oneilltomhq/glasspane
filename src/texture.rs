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
/// `Bgra8Unorm` is the wgpu spelling of `ARGB8888`/`XRGB8888`. A client
/// committing in the other byte order (`ABGR8888`/`XBGR8888`) is drawn into a
/// texture of this format all the same (see [`Reorder`]). One format for every
/// tenant means the shell can write one shader and one bind group layout.
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
/// The 8888 formats in both byte orders, and no others. `ARGB8888` and
/// `XRGB8888` import as [`TENANT_FORMAT`] and are copied. `ABGR8888` and
/// `XBGR8888` import as `Rgba8Unorm` and are drawn across by [`Reorder`]:
/// Android's gralloc allocates nothing else for RGBA, and Waydroid's
/// hwcomposer passes gralloc's format through rather than picking from those
/// advertised, so a compositor without them is a fatal `invalid_format` on
/// Android's first frame. Every other fourcc is multi-planar, which
/// [`wgpu_dmabuf`] does not import, or another bit depth. A client asking for
/// anything else is refused at `wl_buffer` creation, where it can still choose
/// differently, rather than at commit.
///
/// `wl_shm` stays on the first two: smithay offers only those, and every shm
/// client this crate has met takes them.
pub const ACCEPTED_FOURCC: [Fourcc; 4] = [
    Fourcc::Argb8888,
    Fourcc::Xrgb8888,
    Fourcc::Abgr8888,
    Fourcc::Xbgr8888,
];

/// A tenant's or a popup's stable texture. `label` is what it is called in a
/// GPU capture — `tenant#3`, `popup#7`.
pub(crate) fn create_surface_texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("glasspane {label}")),
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

/// Upload an `wl_shm` pool's bytes into a tenant texture: the `crop`
/// rectangle of them (`x, y, width, height`, see `Shape::crop`), which is
/// the whole buffer unless a viewport says otherwise.
///
/// The whole picture, every commit. glasspane does not yet track damage: a
/// partial upload that is wrong by one rectangle produces a tenant that looks
/// right and is stale in a corner, and that failure is not worth having until
/// there is a measurement saying the full upload costs something.
pub(crate) fn upload_shm(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    bytes: &[u8],
    stride: u32,
    crop: (u32, u32, u32, u32),
) {
    let (x, y, width, height) = crop;
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            // Four bytes a pixel: both formats `absorb_shm` takes.
            offset: u64::from(y) * u64::from(stride) + u64::from(x) * 4,
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

/// Copy the `crop` rectangle of an imported client image (`x, y, width,
/// height`, see `Shape::crop`) into the tenant texture, at its origin.
///
/// The submit this issues is the one that carries the acquire wait staged by
/// [`crate::acquire`], which is why the wait is staged at the top of the
/// commit and not later.
pub(crate) fn blit(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Texture,
    dst: &wgpu::Texture,
    crop: (u32, u32, u32, u32),
) {
    let (x, y, width, height) = crop;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("glasspane tenant blit"),
    });
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: src,
            mip_level: 0,
            origin: wgpu::Origin3d { x, y, z: 0 },
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

/// Draws an imported image whose format is not [`TENANT_FORMAT`] into a
/// tenant texture, for the clients [`blit`] cannot serve.
///
/// `copy_texture_to_texture` needs both formats to match, and `Rgba8Unorm`
/// and `Bgra8Unorm` do not. A draw does not care: the fragment shader loads
/// texels as `(r, g, b, a)` whatever their order in memory, and the target
/// stores them in its own. One full-screen triangle, one `textureLoad` per
/// pixel, no sampler and so no filtering: byte for byte, as the copy is.
///
/// Built on the first commit that needs it, so a shell with no such client
/// never compiles the shader.
pub(crate) struct Reorder {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

const REORDER_WGSL: &str = r"
struct Crop { origin: vec2<u32> }

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> crop: Crop;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(src, vec2<u32>(at.xy) + crop.origin, 0);
}
";

impl Reorder {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("glasspane reorder"),
            source: wgpu::ShaderSource::Wgsl(REORDER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("glasspane reorder"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("glasspane reorder"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("glasspane reorder"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TENANT_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// [`blit`]'s contract, drawn rather than copied: the `crop` rectangle
    /// of `src` into `dst` at its origin, in one submit. `dst` is the crop's
    /// size (`ensure_texture` made it so), so the triangle covers all of it.
    pub(crate) fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        crop: (u32, u32, u32, u32),
    ) {
        use wgpu::util::DeviceExt as _;
        let (x, y, _, _) = crop;
        // A vec2<u32>, padded to the 16 bytes a uniform binding is sized in.
        let mut origin = [0u8; 16];
        origin[..4].copy_from_slice(&x.to_ne_bytes());
        origin[4..8].copy_from_slice(&y.to_ne_bytes());
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("glasspane reorder crop"),
            contents: &origin,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glasspane reorder"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        let dst_view = dst.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("glasspane tenant reorder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("glasspane tenant reorder"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }
}

/// The fields every surface this crate hands to the shell has: its size, its
/// buffer's size, its texture, how many commits have landed in it, and
/// whether the last one was fenced.
///
/// A toplevel and a popup differ in what they *are* — one is a window, the
/// other a rectangle floating over one — and not at all in how their pixels
/// get here, so the commit path is written once against this and both wear it.
#[derive(Debug, Default)]
pub(crate) struct Skin {
    /// The surface's logical size: what it was configured to, what input is
    /// in, and what a popup is positioned in. Equal to [`Skin::buffer`] for a
    /// client drawing at scale 1.
    pub(crate) size: (u32, u32),
    /// The last committed buffer's size in pixels, cropped to its viewport's
    /// source if it sets one: the texture's. For a surface with subsurfaces,
    /// the composite's (see [`crate::compose`]).
    /// Larger than [`Skin::size`] when the client draws at a scale above 1;
    /// `(0, 0)` before the first commit.
    pub(crate) buffer: (u32, u32),
    /// Where the texture's top-left corner is in the surface, in its logical
    /// pixels: a toplevel's window geometry origin, the shadow margin a client
    /// that draws its own frame leaves around the window (see
    /// [`crate::state::Shape::window`]). `(0, 0)` for every other surface. The
    /// pointer is handed over shifted by it, so the shell's coordinates and
    /// the texture's stay one space.
    pub(crate) origin: (i32, i32),
    pub(crate) texture: Option<std::sync::Arc<wgpu::Texture>>,
    pub(crate) generation: u64,
    pub(crate) acquire_staged: bool,
    /// `None` for a surface drawn alone, its buffer straight into
    /// [`Skin::texture`]. For one with subsurfaces, whose texture is a
    /// composite (see [`crate::compose`]), a hash of the tree's shape — which
    /// surfaces, in what order, where — when it was last drawn, so a tree that
    /// changes shape without a commit to say so is noticed and redrawn.
    pub(crate) tree: Option<u64>,
}
