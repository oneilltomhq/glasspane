//! A Chromium as a tenant: spawn it on our private socket, tick it, and say
//! what it committed.
//!
//! ```console
//! cargo run --example chromium_tenant -- https://example.com
//! ```
//!
//! Headless in the strict sense — there is no display of any kind involved,
//! and `DISPLAY` is removed from the child's environment so it cannot quietly
//! attach to the user's session and report a success that happened elsewhere.
//! The tenant's texture is written to `chromium.png`.

use std::time::{Duration, Instant};

use glasspane::{Caps, Event, Glasspane};

// The readback and PNG plumbing is the other example's, borrowed rather than
// copied; its `main` is dead here and that is all this allow is for.
#[allow(dead_code)]
#[path = "spawn_foot.rs"]
mod foot;

fn main() {
    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data:text/html,<body style='background:#ff0000'>".into());
    let profile = std::env::temp_dir().join(format!("glasspane-chromium-{}", std::process::id()));

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

    let mut pane = Glasspane::new(
        device,
        queue,
        Caps::new().with_adapter(&adapter).with_size(1280, 800),
    )
    .expect("glasspane");
    eprintln!(
        "socket {} (dma-buf advertised: {})",
        pane.socket_name().to_string_lossy(),
        pane.advertises_dmabuf()
    );

    let (command, port) = glasspane::tenant::browser::chromium(&url, &profile).expect("chromium");
    eprintln!("cdp port {port}, profile {}", profile.display());
    let mut child = pane.spawn(command).expect("spawn");

    let seconds: u64 = std::env::var("GLASSPANE_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut id = None;
    while Instant::now() < deadline {
        pane.dispatch(Some(Duration::from_millis(8))).unwrap();
        pane.tick(Instant::now());
        for event in pane.events() {
            match event {
                Event::Mapped { id: new, pid } => {
                    eprintln!("mapped {new} (pid {pid:?})");
                    id = Some(new);
                }
                Event::TitleChanged { id, title } => eprintln!("{id}: title {title:?}"),
                Event::AppIdChanged { id, app_id } => eprintln!("{id}: app_id {app_id:?}"),
                Event::Resized { id, size } => eprintln!("{id}: resized {size:?}"),
                Event::Rejected { id, why } => eprintln!("{id:?}: REJECTED {why}"),
                Event::Unmapped { id } => eprintln!("{id}: unmapped"),
                Event::Committed { .. } => {}
                _ => {}
            }
        }
    }

    for tenant in pane.tenants() {
        eprintln!(
            "tenant {} {:?} {:?} size {:?} commits {} fenced {}",
            tenant.id(),
            tenant.title(),
            tenant.app_id(),
            tenant.size(),
            tenant.generation(),
            tenant.acquire_staged()
        );
    }
    let (imports, hits) = pane.import_stats();
    eprintln!("imports {imports}, cache hits {hits}");
    if let Some(report) = pane.acquire_report() {
        eprintln!("acquire: {report}");
    }

    if let Some(id) = id {
        if let Some(texture) = pane.tenant(id).and_then(|t| t.texture()) {
            let (w, h) = (texture.width(), texture.height());
            foot::write_png("chromium.png", &foot::read_back(&pane, texture), w, h);
            eprintln!("wrote chromium.png ({w}x{h})");
        }
    } else {
        eprintln!("no toplevel ever mapped");
    }
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&profile);
}
