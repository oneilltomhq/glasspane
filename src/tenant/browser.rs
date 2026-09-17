//! Browsers as tenants: the argv and environment that make a Chromium-family
//! browser a well-behaved client of *this* compositor rather than of the
//! user's desktop.
//!
//! Everything here is about the child process, not about pixels. It builds a
//! [`Command`]; [`crate::Glasspane::spawn`] is what points it at the socket.
//!
//! **Two pieces of hygiene are load-bearing rather than tidy.** `DISPLAY` and
//! `WAYLAND_SOCKET` are both removed. Chromium prefers X11 when it can see it,
//! and a run that quietly attached to the user's live session would report a
//! browser that painted nothing here. `WAYLAND_SOCKET` is the same bug one
//! layer up: it names an already-connected fd and takes precedence over
//! `WAYLAND_DISPLAY`, so a process inheriting one would hand the browser
//! somebody else's compositor while looking correctly configured.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Chromium-family binaries, in preference order.
///
/// Chromium first, and not as a taste: `google-chrome` is a proprietary binary
/// from a vendor repo that is not in any distribution's package set, while
/// Chromium is distro-packaged and reproducible from a release tag. Chrome
/// stays as a fallback because it is what many machines actually have.
pub const CANDIDATES: &[&str] = &["chromium", "chromium-browser", "google-chrome"];

/// Find a Chromium-family browser on `PATH`.
///
/// The error names what was searched, because the only useful thing an error
/// here can say is "install one of these, or point me at yours".
pub fn find() -> Result<PathBuf, String> {
    let dirs: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect())
        .unwrap_or_default();
    for c in CANDIDATES {
        if let Some(hit) = which_in(&dirs, c) {
            return Ok(hit);
        }
    }
    Err(format!(
        "no browser found: searched PATH for {}",
        CANDIDATES.join(", ")
    ))
}

/// A [`Command`] that opens `url` as a chromeless tenant, plus the CDP port it
/// will listen on.
///
/// The port is picked here, by binding `127.0.0.1:0` and reading back the
/// number, so the caller can drive the *same* browser instance it is hosting.
/// A second instance would be a second layout — different fonts loaded,
/// different lazy images resolved — and comparing against it would assert that
/// two browsers agree rather than that this one painted.
///
/// `profile_dir` is created if it does not exist and is **not** cleaned up:
/// this crate does not own the caller's disk. Use a throwaway directory; a
/// Chromium profile is 60–170 MB and a `$XDG_RUNTIME_DIR` full of them is a
/// tmpfs full of them.
///
/// ```no_run
/// let (cmd, port) = glasspane::tenant::browser::chromium(
///     "https://example.com",
///     std::path::Path::new("/tmp/gp-profile"),
/// )?;
/// # Ok::<_, String>(())
/// ```
pub fn chromium(url: &str, profile_dir: &Path) -> Result<(Command, u16), String> {
    let bin = find()?;
    let port = free_port()?;
    Ok((command_at(&bin, url, profile_dir, port)?, port))
}

/// [`chromium`] with the binary and port chosen by the caller.
pub fn command_at(
    binary: &Path,
    url: &str,
    profile_dir: &Path,
    cdp_port: u16,
) -> Result<Command, String> {
    std::fs::create_dir_all(profile_dir)
        .map_err(|e| format!("create {}: {e}", profile_dir.display()))?;
    let mut cmd = Command::new(binary);
    cmd.args(argv(url, profile_dir, cdp_port))
        .env_remove("DISPLAY")
        .env_remove("WAYLAND_SOCKET");
    Ok(cmd)
}

/// The argv, with every flag's reason attached to it.
pub fn argv(url: &str, profile_dir: &Path, cdp_port: u16) -> Vec<String> {
    vec![
        // The whole point: a Wayland client, not an X11 one.
        "--ozone-platform=wayland".into(),
        // Exact bytes out. Chromium will otherwise convert through the
        // display's ICC profile and #ff0000 stops being 0xff,0x00,0x00 —
        // which turns every pixel assertion from an identity into a tolerance.
        "--force-color-profile=srgb".into(),
        // A window with no browser UI at all: no tab strip, no omnibox. The
        // page is the whole surface. (The window *frame* is a different
        // problem, and xdg-decoration solves it — glasspane announces
        // server-side decorations and draws none.)
        format!("--app={url}"),
        // A private profile per tenant: no first-run bubble, no session
        // restore prompt, no interference with the user's real profile.
        format!("--user-data-dir={}", profile_dir.display()),
        format!("--remote-debugging-port={cdp_port}"),
        "--remote-allow-origins=*".into(),
        "--no-first-run".into(),
        "--no-default-browser-check".into(),
        // Startup noise that costs seconds and can fail. None of it is needed
        // to paint a page, and each one is a way for a headless run to become
        // flaky for reasons unrelated to the compositor.
        "--disable-default-apps".into(),
        "--disable-component-update".into(),
        "--disable-sync".into(),
        "--disable-background-networking".into(),
        "--no-service-autorun".into(),
        "--metrics-recording-only".into(),
        "--disable-breakpad".into(),
    ]
}

/// The same, as a pure `wl_shm` client.
///
/// `--disable-gpu` makes Chromium software-rasterise, which removes an entire
/// class of "did the browser or the compositor get the modifier wrong" from a
/// measurement that is about neither. Use it to isolate a dma-buf question;
/// do not use it to make a dma-buf problem go away.
pub fn argv_software(url: &str, profile_dir: &Path, cdp_port: u16) -> Vec<String> {
    let mut v = argv(url, profile_dir, cdp_port);
    v.push("--disable-gpu".into());
    v
}

/// A TCP port nothing is listening on, as of a moment ago.
///
/// Inherently racy and unavoidably so: the browser binds it, not us, and there
/// is no way to hand a listening socket to `--remote-debugging-port`.
pub fn free_port() -> Result<u16, String> {
    TcpListener::bind(("127.0.0.1", 0))
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .map_err(|e| format!("no free port: {e}"))
}

/// `which(1)`, minus the process.
///
/// Empty `PATH` entries are skipped. A `PATH` of `"/usr/bin:"` splits into an
/// empty component, and joining a name onto that yields a *relative* path —
/// which historic `which` resolves against the working directory, so a file
/// dropped in a checkout could shadow the installed browser. Nothing here
/// wants a cwd-dependent answer.
fn which_in(dirs: &[PathBuf], name: &str) -> Option<PathBuf> {
    dirs.iter()
        .filter(|d| !d.as_os_str().is_empty())
        .map(|d| d.join(name))
        .find(|c| is_executable(c))
}

fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty `PATH` component must never become a relative lookup.
    #[test]
    fn empty_path_components_are_skipped() {
        let dirs = vec![PathBuf::new(), PathBuf::from("/nonexistent-glasspane")];
        assert!(which_in(&dirs, "sh").is_none());
    }

    /// Every flag the fixtures depend on is present, and the URL is carried by
    /// `--app` rather than as a positional argument (which would open a tab in
    /// a framed window instead).
    #[test]
    fn argv_carries_the_load_bearing_flags() {
        let v = argv("file:///x.html", Path::new("/tmp/p"), 9333);
        assert!(v.contains(&"--ozone-platform=wayland".to_string()));
        assert!(v.contains(&"--app=file:///x.html".to_string()));
        assert!(v.contains(&"--remote-debugging-port=9333".to_string()));
        assert!(v.contains(&"--user-data-dir=/tmp/p".to_string()));
        assert!(!v.iter().any(|a| !a.starts_with('-')), "no positional args");
    }

    /// The software-rasterised variant differs from the honest one in exactly
    /// one place.
    #[test]
    fn software_differs_by_one_flag() {
        let a = argv("u", Path::new("/tmp/p"), 1);
        let b = argv_software("u", Path::new("/tmp/p"), 1);
        assert_eq!(b.len(), a.len() + 1);
        assert_eq!(b[..a.len()], a[..]);
        assert_eq!(b[a.len()], "--disable-gpu");
    }
}
