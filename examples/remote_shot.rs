//! Connect to a running `glasspane-daemon` as a shell, and save one tenant's
//! texture as a PNG.
//!
//! ```console
//! # one terminal (headless: the daemon opens no window, and Chromium is a
//! # client of the daemon's socket, not of your desktop)
//! cargo run --features daemon --bin glasspane-daemon -- --url https://example.com
//! # another
//! cargo run --features remote --example remote_shot -- [OUT.png] [CONTROL_SOCKET]
//! ```
//!
//! Defaults: `remote.png`, and `$XDG_RUNTIME_DIR/plurality/glasspane.sock`.
//! Run it twice and the second shot is the same window — the daemon replays
//! every tenant to each shell that connects, and the tenant never notices the
//! shell changed.
//!
//! `GLASSPANE_SECONDS` (default 3) is how long to tick before the shot, so a
//! page that is still loading has time to paint.

use std::time::{Duration, Instant};

use glasspane::Event;

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "remote.png".into());
    let control = args.next().map(std::path::PathBuf::from);

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
    let (device, queue, _caps) =
        wgpu_dmabuf::device::open_device(&adapter, &Default::default()).expect("device");

    let mut pane = match &control {
        Some(path) => glasspane::remote::Glasspane::connect(device, queue, path),
        None => glasspane::remote::Glasspane::new(device, queue),
    }
    .expect("connect to glasspane-daemon");
    eprintln!(
        "connected; daemon's WAYLAND_DISPLAY is {}",
        pane.socket_name().to_string_lossy()
    );

    let seconds: u64 = std::env::var("GLASSPANE_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let deadline = Instant::now() + Duration::from_secs(seconds);
    while Instant::now() < deadline {
        pane.dispatch(Some(Duration::from_millis(8)))
            .expect("dispatch");
        pane.tick(Instant::now());
        for event in pane.events() {
            match event {
                Event::Mapped { id, pid } => eprintln!("mapped {id} (pid {pid:?})"),
                Event::TitleChanged { id, title } => eprintln!("{id}: title {title:?}"),
                Event::Resized { id, size } => eprintln!("{id}: resized {size:?}"),
                Event::Rejected { id, why } => eprintln!("{id:?}: REJECTED {why}"),
                Event::Unmapped { id } => eprintln!("{id}: unmapped"),
                _ => {}
            }
        }
    }

    let (imports, hits) = pane.import_stats();
    eprintln!("imports {imports}, cache hits {hits}");
    if let Some(report) = pane.acquire_report() {
        eprintln!("{report}");
    }
    // The lowest id is the oldest tenant: a stable choice, not a list position.
    let Some(tenant) = pane
        .tenants()
        .filter(|t| t.texture().is_some())
        .min_by_key(|t| t.id())
    else {
        eprintln!("no tenant has committed a buffer");
        std::process::exit(1);
    };
    let texture = tenant.texture().unwrap();
    let (w, h) = (texture.width(), texture.height());
    eprintln!(
        "{}: {:?} {w}x{h} generation {} buffer {:?}",
        tenant.id(),
        tenant.title(),
        tenant.generation(),
        tenant.buffer()
    );
    write_png(&out, &read_back(pane.device(), pane.queue(), texture), w, h);
    eprintln!("wrote {out}");
}

/// Copy a texture back as tightly packed BGRA.
fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let (w, h) = (texture.width(), texture.height());
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = (w * 4).div_ceil(align) * align;
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
    queue.submit([encoder.finish()]);
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
fn write_png(path: &str, bgra: &[u8], w: u32, h: u32) {
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
