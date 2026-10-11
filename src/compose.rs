//! Several textures become one: a surface and its subsurfaces, flattened
//! into the single texture the shell samples.
//!
//! A tenant is one texture (see [`crate::texture`]), and a client that draws
//! into `wl_subsurface`s — Wine's menus and GL client areas, Waydroid's whole
//! Android picture — has handed over several. The shell must not have to know:
//! this crate's contract is one texture per tenant, and a pane that grew a
//! second texture whenever its client opened a menu would make every shell
//! write a scene graph for something that is the client's own business.
//!
//! So the tree is drawn here, on the GPU, into the tenant's texture: one
//! textured quad per surface, back to front, premultiplied-alpha over a
//! transparent clear. A render pass and not a chain of
//! `copy_texture_to_texture`, because a copy cannot scale, and a
//! `wp_viewport` destination is exactly a surface asking to be scaled
//! (Waydroid stretches a 1x1 buffer over the whole window); and because a copy
//! cannot blend, and a subsurface with alpha is a surface asking to be blended.
//! The pixels are already on the GPU — both upload paths land them in a
//! texture — so a CPU composite would mean reading them back first.
//!
//! Like [`crate::texture`] this module knows nothing about Wayland. What a
//! quad is, where, and how big, is decided in [`crate::state`]; this draws it.

/// A surface's pixels, and where in the target they go.
#[derive(Debug)]
pub(crate) struct Quad<'a> {
    /// The surface's texture: its buffer, already cropped to its viewport's
    /// source, so it is drawn whole.
    pub(crate) texture: &'a wgpu::Texture,
    /// `x, y, width, height` in the target's pixels. May run past the target,
    /// or start before it; the target's edge clips it.
    pub(crate) rect: [f32; 4],
    /// The buffer is `XRGB8888`: its alpha byte means nothing, and is drawn
    /// as `1`. Blending by it would make a menu whose client left the byte at
    /// zero invisible.
    pub(crate) opaque: bool,
}

/// The pipeline the quads are drawn with. Built the first time a tree needs
/// compositing, never for a client without subsurfaces.
#[derive(Debug)]
pub(crate) struct Compositor {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

const WGSL: &str = r#"
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) opaque: f32,
};

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;

// One instance per quad: `rect` is `x0, y0, dx, dy` in clip space, already
// flipped so that y grows down the target as it does in the buffer.
@vertex
fn vs(@builtin(vertex_index) i: u32, @location(0) rect: vec4<f32>, @location(1) opaque: f32) -> Out {
    let corner = vec2<f32>(f32(i & 1u), f32(i >> 1u));
    var out: Out;
    out.pos = vec4<f32>(rect.xy + corner * rect.zw, 0.0, 1.0);
    out.uv = corner;
    out.opaque = opaque;
    return out;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    var c = textureSample(tex, smp, in.uv);
    if (in.opaque > 0.5) {
        c.a = 1.0;
    }
    return c;
}
"#;

/// Bytes per instance: a `vec4<f32>` rect and an `f32` flag.
const INSTANCE: u64 = 20;

impl Compositor {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("glasspane compose"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("glasspane compose"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("glasspane compose"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("glasspane compose"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: INSTANCE,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: 16,
                            shader_location: 1,
                        },
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::texture::TENANT_FORMAT,
                    // Wayland buffers are premultiplied.
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        // Linear, because a viewport destination is a request to be scaled,
        // and at one texel per pixel at a whole-pixel offset — every surface
        // drawn at its own size — linear samples texel centres and is exact.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("glasspane compose"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Compositor {
            pipeline,
            layout,
            sampler,
        }
    }

    /// Clear `target` to transparent and draw `quads` into it, first to last.
    pub(crate) fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::Texture,
        quads: &[Quad<'_>],
    ) {
        let (w, h) = (target.width() as f32, target.height() as f32);
        let mut bytes = Vec::with_capacity(quads.len() * INSTANCE as usize);
        for q in quads {
            let [x, y, qw, qh] = q.rect;
            let clip = [
                x / w * 2.0 - 1.0,
                1.0 - y / h * 2.0,
                qw / w * 2.0,
                -(qh / h * 2.0),
                if q.opaque { 1.0 } else { 0.0 },
            ];
            for f in clip {
                bytes.extend_from_slice(&f.to_ne_bytes());
            }
        }
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glasspane compose quads"),
            size: (bytes.len() as u64).max(INSTANCE),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instances, 0, &bytes);
        let groups: Vec<wgpu::BindGroup> = quads
            .iter()
            .map(|q| {
                let view = q.texture.create_view(&Default::default());
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("glasspane compose layer"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                })
            })
            .collect();
        let view = target.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("glasspane compose"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("glasspane compose"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
            pass.set_vertex_buffer(0, instances.slice(..));
            for (i, group) in groups.iter().enumerate() {
                pass.set_bind_group(0, group, &[]);
                pass.draw(0..4, i as u32..i as u32 + 1);
            }
        }
        queue.submit([encoder.finish()]);
    }
}
