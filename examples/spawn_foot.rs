//! Run a terminal with no display attached, and save what it drew.
//!
//! ```console
//! cargo run --example spawn_foot
//! ```
//!
//! There is no window, no compositor session and no X server involved: this
//! program *is* the compositor, and `foot` connects to a socket it owns. After
//! two seconds of frame clock the tenant's texture is copied back and written
//! to `foot.png`.
//!
//! Needs `foot` on `PATH` and a DRM render node. A shell would sample the same
//! texture instead of saving it.

use std::process::Command;
use std::time::{Duration, Instant};

use glasspane::{Caps, Event, Glasspane};

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .expect("a Vulkan adapter");

    // open_device, not request_device: the fence extension a dma-buf tenant
    // needs is not one stock wgpu enables.
    let (device, queue, caps) =
        wgpu_dmabuf::device::open_device(&adapter, &Default::default()).expect("device");
    eprintln!("adapter: {} ({caps:?})", adapter.get_info().name);

    let mut pane = Glasspane::new(
        device,
        queue,
        Caps::new().with_adapter(&adapter).with_size(800, 480),
    )
    .expect("glasspane");
    eprintln!(
        "socket: {} (dma-buf advertised: {})",
        pane.socket_name().to_string_lossy(),
        pane.advertises_dmabuf()
    );

    let mut child = pane
        .spawn(Command::new("foot"))
        .expect("foot is not on PATH");

    // Two seconds of *our* clock. A client paints when it is handed a frame
    // callback and at no other time, so this loop is what makes foot draw.
    let mut id = None;
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        pane.dispatch(Some(Duration::from_millis(8))).unwrap();
        pane.tick(Instant::now());
        for event in pane.events() {
            match event {
                Event::Mapped { id: new, pid } => {
                    eprintln!("mapped {new} (pid {pid:?})");
                    id = Some(new);
                }
                Event::TitleChanged { id, title } => eprintln!("{id}: {title:?}"),
                Event::Resized { id, size } => eprintln!("{id}: resized to {size:?}"),
                Event::Rejected { id, why } => eprintln!("{id:?}: REJECTED {why}"),
                _ => {}
            }
        }
    }

    let Some(id) = id else {
        eprintln!("foot never mapped a toplevel");
        let _ = child.kill();
        let _ = child.wait();
        return;
    };
    let tenant = pane.tenant(id).expect("tenant");
    eprintln!(
        "{id}: {:?} {:?} {:?}, {} commits",
        tenant.title(),
        tenant.app_id(),
        tenant.size(),
        tenant.generation()
    );
    if let Some(report) = pane.acquire_report() {
        eprintln!("acquire: {report}");
    }
    let (imports, hits) = pane.import_stats();
    eprintln!("imports: {imports}, cache hits: {hits}");

    match tenant.texture() {
        Some(texture) => {
            let (w, h) = (texture.width(), texture.height());
            write_png("foot.png", &read_back(&pane, texture), w, h);
            eprintln!("wrote foot.png ({w}x{h})");
        }
        None => eprintln!("the tenant never committed a buffer"),
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Copy a texture back to the CPU as tightly packed BGRA.
pub fn read_back(pane: &Glasspane, texture: &wgpu::Texture) -> Vec<u8> {
    let (w, h) = (texture.width(), texture.height());
    // copy_texture_to_buffer wants 256-byte aligned rows; unpad afterwards.
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = (w * 4).div_ceil(align) * align;
    let device = pane.device();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    pane.queue().submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let view = slice.get_mapped_range().expect("mapped range");
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for row in view.chunks_exact(padded as usize) {
        out.extend_from_slice(&row[..(w * 4) as usize]);
    }
    drop(view);
    staging.unmap();
    out
}

/// Write BGRA bytes out as an RGBA PNG.
pub fn write_png(path: &str, bgra: &[u8], w: u32, h: u32) {
    let mut rgba = bgra.to_vec();
    for px in rgba.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
    let file = std::fs::File::create(path).expect("create png");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .expect("png header")
        .write_image_data(&rgba)
        .expect("png data");
}
