//! **A window survives the shell.**
//!
//! The daemon runs as its own process (the real `glasspane-daemon` binary),
//! the fixture client paints a dma-buf into it, and two shells — each a
//! separate process, this test binary re-run as [`shell_process`] — connect
//! one after the other. The first imports the tenant's buffer and checks every
//! pixel, then exits. The second connects to a daemon that has been running
//! blind, receives the replayed tenant, imports the *same* buffer and checks
//! the same pixels. The client must have seen nothing: no disconnect, and no
//! configure after its first.
//!
//! ```console
//! cargo test --features daemon,remote --test daemon
//! ```

mod fixture;

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fixture::{Fixture, Report, Transport};
use glasspane::{Event, TenantId};

/// Set in a shell child: the control socket to connect to.
const CONTROL_ENV: &str = "GLASSPANE_TEST_SHELL_CONTROL";
/// Set in a shell child: its name in the log.
const NAME_ENV: &str = "GLASSPANE_TEST_SHELL_NAME";

fn gpu() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    if wgpu_dmabuf::testkit::render_node().is_none() {
        eprintln!("SKIP: no /dev/dri/renderD* on this machine");
        return None;
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
    .ok()?;
    match wgpu_dmabuf::device::open_device(&adapter, &Default::default()) {
        Ok((device, queue, caps)) if caps.dmabuf => Some((adapter, device, queue)),
        Ok(_) => {
            eprintln!("SKIP: adapter has no VULKAN_EXTERNAL_MEMORY_DMA_BUF");
            None
        }
        Err(e) => {
            eprintln!("SKIP: cannot open a dma-buf capable device: {e}");
            None
        }
    }
}

fn runtime_dir() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| format!("/run/user/{}", unsafe { libc::getuid() }).into())
}

/// Kills the daemon however the test ends.
struct Reap(Child);

impl Drop for Reap {
    fn drop(&mut self) {
        // SIGTERM first, so the daemon's own Drop runs and removes its socket.
        unsafe { libc::kill(self.0.id() as i32, libc::SIGTERM) };
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.0.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// What one shell process reported, parsed from its `SHELL` line.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ShellReport {
    tenant: u64,
    size_at_mapped: (u32, u32),
    title: String,
    buffer: u64,
    generation: u64,
    live_commits: u64,
    pixels: String,
}

fn parse(line: &str) -> ShellReport {
    let mut r = ShellReport {
        tenant: 0,
        size_at_mapped: (0, 0),
        title: String::new(),
        buffer: 0,
        generation: 0,
        live_commits: 0,
        pixels: String::new(),
    };
    for kv in line.split_whitespace().skip(1) {
        let (k, v) = kv.split_once('=').expect("key=value");
        match k {
            "tenant" => r.tenant = v.parse().unwrap(),
            "size_at_mapped" => {
                let (w, h) = v.split_once('x').unwrap();
                r.size_at_mapped = (w.parse().unwrap(), h.parse().unwrap());
            }
            "title" => r.title = v.to_string(),
            "buffer" => r.buffer = v.parse().unwrap(),
            "generation" => r.generation = v.parse().unwrap(),
            "live_commits" => r.live_commits = v.parse().unwrap(),
            "pixels" => r.pixels = v.to_string(),
            _ => {}
        }
    }
    r
}

/// Run one shell as a separate process and wait for it to exit.
fn run_shell(control: &Path, name: &str) -> ShellReport {
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "shell_process",
            "--ignored",
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .env(CONTROL_ENV, control)
        .env(NAME_ENV, name)
        .stderr(Stdio::inherit())
        .output()
        .expect("run shell process");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "shell {name} failed: {}\n{stdout}",
        out.status
    );
    let line = stdout
        .lines()
        // libtest prints `test shell_process ... ` on the same line first.
        .find_map(|l| l.find("SHELL ").map(|i| &l[i..]))
        .unwrap_or_else(|| panic!("shell {name} printed no SHELL line:\n{stdout}"));
    eprintln!("{line}");
    parse(line)
}

/// Wait for a report matching `want`, keeping *every* report in `seen` —
/// unlike `Fixture::expect`, which drops the ones it skips.
fn collect_until(
    client: &Fixture,
    seen: &mut Vec<Report>,
    timeout: Duration,
    want: impl Fn(&Report) -> bool,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match client
            .reports
            .recv_timeout(left.max(Duration::from_millis(1)))
        {
            Ok(r) => {
                let hit = want(&r);
                seen.push(r);
                if hit {
                    return;
                }
            }
            Err(_) => panic!("the client never reported it; it did report {seen:?}"),
        }
    }
}

fn count_commits(reports: &[Report]) -> usize {
    reports
        .iter()
        .filter(|r| matches!(r, Report::Committed { .. }))
        .count()
}

/// **The spike's proof.**
#[test]
fn a_window_survives_a_shell_restart() {
    if gpu().is_none() {
        return;
    }
    let tag = format!("gp-daemon-test-{}", std::process::id());
    let control = runtime_dir().join(format!("{tag}.sock"));
    let wayland = format!("{tag}-wl");

    let mut child = Command::new(env!("CARGO_BIN_EXE_glasspane-daemon"))
        .args(["--wayland", &wayland, "--size", "256x256", "--control"])
        .arg(&control)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn glasspane-daemon");
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let daemon = Reap(child);

    // The daemon's log, kept so the test can say the daemon *noticed* the
    // shell leave, and echoed so a failure shows it.
    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    {
        let log = Arc::clone(&log);
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                eprintln!("  [daemon] {line}");
                log.lock().unwrap().push(line);
            }
        });
    }
    let mut ready = String::new();
    BufReader::new(stdout).read_line(&mut ready).unwrap();
    assert!(
        ready.starts_with("ready "),
        "daemon did not come up: {ready:?}"
    );

    // The tenant. It paints once on configure, and again on every frame
    // callback — which, with no shell connected, the daemon's blind clock
    // answers.
    let client = Fixture::start(std::ffi::OsStr::new(&wayland), Transport::Dmabuf);
    let mut seen = Vec::new();
    collect_until(&client, &mut seen, Duration::from_secs(10), |r| {
        matches!(r, Report::Committed { .. })
    });
    std::thread::sleep(Duration::from_millis(300));
    seen.extend(client.drain());
    let before_any_shell = count_commits(&seen);
    assert!(
        before_any_shell > 3,
        "with no shell the daemon must answer frame callbacks on its own clock; \
         the client committed only {before_any_shell} times in 300 ms"
    );

    // Shell A: connect, import, check, exit.
    let a = run_shell(&control, "A");

    // Nobody is connected now. The daemon must have noticed, and the client
    // must keep running.
    std::thread::sleep(Duration::from_millis(400));
    assert!(
        log.lock().unwrap().iter().any(|l| l.contains("shell gone")),
        "the daemon never noticed shell A leave"
    );
    let blind = client.drain();
    let blind_commits = count_commits(&blind);
    seen.extend(blind);
    assert!(
        blind_commits > 3,
        "the client stalled once shell A left: {blind_commits} commits in 400 ms"
    );

    // Shell B: a new process, a daemon that has run blind in between.
    let b = run_shell(&control, "B");

    // And the client is still alive and painting after both.
    let after = seen.len();
    collect_until(&client, &mut seen, Duration::from_secs(5), |r| {
        matches!(r, Report::Committed { .. })
    });
    assert!(count_commits(&seen[after..]) > 0);
    seen.extend(client.drain());

    for (name, s) in [("A", &a), ("B", &b)] {
        assert_eq!(s.pixels, "ok", "shell {name} read back the wrong pixels");
        assert_eq!(s.size_at_mapped, (256, 256), "shell {name}: size on Mapped");
        assert_eq!(s.title, "glasspane-fixture", "shell {name}: title");
        assert!(
            s.live_commits > 0,
            "shell {name}: its ticks drove no commits"
        );
    }
    assert_eq!(
        a.tenant, b.tenant,
        "the tenant's identity changed across the restart"
    );
    assert_eq!(a.buffer, b.buffer, "B was not handed the same wl_buffer");
    assert!(
        b.generation > a.generation,
        "B's replay must carry commits made while blind: A saw {}, B {}",
        a.generation,
        b.generation
    );

    let failed: Vec<_> = seen
        .iter()
        .filter(|r| matches!(r, Report::Failed(_)))
        .collect();
    assert!(failed.is_empty(), "the client failed: {failed:?}");
    let configures: Vec<_> = seen
        .iter()
        .filter(|r| matches!(r, Report::Configured(..)))
        .collect();
    assert_eq!(
        configures,
        vec![&Report::Configured(256, 256)],
        "the client was reconfigured across the restart"
    );
    let log = log.lock().unwrap();
    assert_eq!(
        log.iter().filter(|l| l.contains("shell connected")).count(),
        2,
        "two shells, two connections"
    );
    eprintln!(
        "client commits: {} total, {} before any shell, {} while blind between A and B",
        count_commits(&seen),
        before_any_shell,
        blind_commits
    );
    drop(client);
    drop(daemon);
}

/// Read a texture back, tightly packed, in memory order.
fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let (w, h) = (texture.width(), texture.height());
    let row = w * 4;
    assert_eq!(row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (row * h) as u64,
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
                bytes_per_row: Some(row),
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
    let out = slice.get_mapped_range().expect("mapped").to_vec();
    staging.unmap();
    out
}

/// **One shell**, run as its own process by the test above. Skipped when run
/// directly (it is `#[ignore]`d, and does nothing without its environment).
///
/// Phase one does *not* tick: the daemon answers nothing while a shell is
/// connected and silent, so the tenant cannot commit, and the first
/// `Committed` to arrive is necessarily the replay. Phase two ticks, and the
/// commits that follow are the `Presented` path driving the client.
#[test]
#[ignore = "run as a child process by a_window_survives_a_shell_restart"]
fn shell_process() {
    let Some(control) = std::env::var_os(CONTROL_ENV) else {
        return;
    };
    let name = std::env::var(NAME_ENV).unwrap_or_default();
    let (_adapter, device, queue) = gpu().expect("the parent found a GPU");
    let mut pane =
        glasspane::remote::Glasspane::connect(device, queue, Path::new(&control)).expect("connect");

    let mut events = Vec::new();
    let mut size_at_mapped = None;
    let mut id: Option<TenantId> = None;
    let deadline = Instant::now() + Duration::from_secs(10);
    let replay = loop {
        assert!(
            Instant::now() < deadline,
            "shell {name}: no replay; events {events:?}"
        );
        pane.dispatch(Some(Duration::from_millis(20)))
            .expect("dispatch");
        for e in pane.events().collect::<Vec<_>>() {
            if let Event::Mapped { id: new, .. } = e {
                id = Some(new);
                size_at_mapped = pane.tenant(new).map(|t| t.size());
            }
            events.push(e);
        }
        if let Some(id) = id {
            if events
                .iter()
                .any(|e| matches!(e, Event::Committed { id: c, .. } if *c == id))
            {
                break id;
            }
        }
    };
    // Mapped, then Resized (a fresh texture), then Committed — in that order.
    // One more Committed may follow without a tick: a frame the blind clock
    // released just before this shell connected, still in flight.
    let order: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            Event::Mapped { .. } => Some("mapped"),
            Event::Resized { .. } => Some("resized"),
            Event::Committed { .. } => Some("committed"),
            _ => None,
        })
        .collect();
    assert_eq!(
        order[..3],
        ["mapped", "resized", "committed"],
        "shell {name}: {events:?}"
    );
    assert!(
        order[3..].iter().all(|e| *e == "committed") && order.len() <= 4,
        "shell {name}: {events:?}"
    );
    let generation = events
        .iter()
        .find_map(|e| match e {
            Event::Committed { generation, .. } => Some(*generation),
            _ => None,
        })
        .unwrap();

    let tenant = pane.tenant(replay).unwrap();
    let buffer = tenant.buffer().unwrap();
    let title = tenant.title().unwrap_or("").to_string();
    assert!(
        tenant.acquire_staged(),
        "shell {name}: no acquire wait staged"
    );
    let texture = tenant.texture().expect("texture");
    let (w, h) = (texture.width(), texture.height());
    let got = read_back(pane.device(), pane.queue(), texture);
    let want = fixture::expected(w, h);
    let pixels = match got
        .chunks_exact(4)
        .zip(want.chunks_exact(4))
        .position(|(g, w)| g != w)
    {
        None => "ok".to_string(),
        Some(i) => format!("first-diff-at-({},{})", i as u32 % w, i as u32 / w),
    };

    // Phase two: drive the clock and watch commits come back.
    let mut live_commits = 0u64;
    let until = Instant::now() + Duration::from_millis(300);
    while Instant::now() < until {
        pane.dispatch(Some(Duration::from_millis(8)))
            .expect("dispatch");
        pane.tick(Instant::now());
        live_commits += pane
            .events()
            .filter(|e| matches!(e, Event::Committed { .. }))
            .count() as u64;
    }
    let (imports, hits) = pane.import_stats();
    eprintln!(
        "shell {name}: imports {imports}, cache hits {hits}, {}",
        pane.acquire_report().unwrap()
    );
    println!(
        "SHELL name={name} tenant={} size_at_mapped={}x{} title={title} buffer={buffer} \
         generation={generation} live_commits={live_commits} pixels={pixels}",
        replay.0,
        size_at_mapped.unwrap().0,
        size_at_mapped.unwrap().1,
    );
    // Dropped here: the process exits, and the connection with it.
}
