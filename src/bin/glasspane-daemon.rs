//! `glasspane-daemon`: the tenant compositor as its own process.
//!
//! ```console
//! glasspane-daemon [--wayland NAME] [--control PATH] [--size WxH]
//!                  [--url URL [--profile DIR]] [--seconds N]
//! ```
//!
//! Opens a Wayland socket in `$XDG_RUNTIME_DIR` and a control socket (default
//! `$XDG_RUNTIME_DIR/plurality/glasspane.sock`) for a shell to connect to with
//! `glasspane::remote::Glasspane`. With `--url`, spawns one Chromium on that
//! URL as a tenant; the daemon owns and kills it. Prints `ready <wayland>
//! <control>` on stdout once both sockets are listening. Stops on SIGINT or
//! SIGTERM, or after `--seconds`.
//!
//! No wgpu device is opened: an adapter is requested only to enumerate the
//! dma-buf formats to advertise.

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use glasspane::daemon::{Config, Daemon};

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

fn usage() -> ! {
    eprintln!(
        "usage: glasspane-daemon [--wayland NAME] [--control PATH] [--size WxH] \
         [--url URL [--profile DIR]] [--seconds N]"
    );
    std::process::exit(2);
}

fn main() {
    let mut config = Config::default();
    let mut url = None;
    let mut profile = None;
    let mut seconds = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--wayland" => config.socket_name = Some(val()),
            "--control" => config.control_path = PathBuf::from(val()),
            "--size" => {
                let v = val();
                let (w, h) = v.split_once('x').unwrap_or_else(|| usage());
                config.default_size = (
                    w.parse().unwrap_or_else(|_| usage()),
                    h.parse().unwrap_or_else(|_| usage()),
                );
            }
            "--url" => url = Some(val()),
            "--profile" => profile = Some(PathBuf::from(val())),
            "--seconds" => seconds = Some(val().parse::<u64>().unwrap_or_else(|_| usage())),
            _ => usage(),
        }
    }

    // SAFETY: installing an async-signal-safe handler that stores an atomic.
    unsafe {
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
    }

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
    .ok();
    if adapter.is_none() {
        eprintln!("glasspane-daemon: no Vulkan adapter; no dma-buf will be advertised");
    }

    let mut daemon = match Daemon::new(adapter.as_ref(), config) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("glasspane-daemon: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "glasspane-daemon: wayland {} control {} dma-buf {}",
        daemon.socket_name().to_string_lossy(),
        daemon.control_path().display(),
        daemon.advertises_dmabuf()
    );

    let mut made_profile = None;
    if let Some(url) = url {
        let profile = profile.unwrap_or_else(|| {
            let base = std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
                .unwrap_or_else(|| PathBuf::from("."));
            let p = base
                .join("glasspane")
                .join(format!("chromium-{}", std::process::id()));
            made_profile = Some(p.clone());
            p
        });
        match glasspane::tenant::browser::chromium(&url, &profile) {
            Ok((cmd, port)) => match daemon.spawn(cmd) {
                Ok(pid) => eprintln!(
                    "glasspane-daemon: chromium pid {pid}, cdp port {port}, profile {}",
                    profile.display()
                ),
                Err(e) => eprintln!("glasspane-daemon: spawn chromium: {e}"),
            },
            Err(e) => eprintln!("glasspane-daemon: {e}"),
        }
    }

    println!(
        "ready {} {}",
        daemon.socket_name().to_string_lossy(),
        daemon.control_path().display()
    );
    let _ = std::io::stdout().flush();

    let deadline = seconds.map(|s| Instant::now() + Duration::from_secs(s));
    while !STOP.load(Ordering::Relaxed) && deadline.is_none_or(|d| Instant::now() < d) {
        if let Err(e) = daemon.turn(Duration::from_millis(50)) {
            eprintln!("glasspane-daemon: {e}");
            break;
        }
    }
    eprintln!("glasspane-daemon: stopping; {:?}", daemon.stats());
    drop(daemon);
    if let Some(p) = made_profile {
        let _ = std::fs::remove_dir_all(p);
    }
}
