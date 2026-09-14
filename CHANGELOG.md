# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-14

First release. A Wayland compositor whose only output is textures.

### Added

- `Glasspane::new(device, queue, caps)` — a compositor on a private socket,
  borrowing the caller's `wgpu::Device`; `dispatch`, `tick`, `tenants`,
  `events`, `configure`, `close`, `spawn`.
- `Tenant`: id, pid, title, app_id, size, a stable `wgpu::Texture` and a
  commit generation.
- Buffer import for `wl_shm` and for single-plane `ARGB8888`/`XRGB8888`
  dma-buf, the latter through `wgpu-dmabuf`, with an import-once registry
  keyed by `wl_buffer` identity.
- Explicit acquire fencing: every dma-buf commit exports a `sync_file` from
  the buffer and waits on it before the blit that reads it. A device that
  cannot import such a fence is refused at construction unless
  `Caps::allow_unfenced_dmabuf` says otherwise.
- Input injection in tenant-local buffer pixels: pointer enter/motion/leave,
  button, axis, keyboard focus and keys in evdev codes.
- `tenant::browser::chromium(url, profile_dir)` — the argv and environment
  that make a Chromium a tenant, plus a free CDP port.
- `examples/spawn_foot.rs`: a terminal, headless, to a PNG.

[Unreleased]: https://github.com/oneilltomhq/glasspane/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/oneilltomhq/glasspane/releases/tag/v0.1.0
