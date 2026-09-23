# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `xdg_popup`: `Glasspane::popups`, `popup`, `Popup` (id, parent, root,
  parent-relative position, size, texture, generation), and the
  `Event::Popup*` family. A popup is hosted like a toplevel — own id, own
  stable texture — and never appears in `tenants()`.
- The positioner is resolved against the root tenant's own rectangle, through
  smithay's `get_unconstrained_geometry`, so a popup near an edge flips or
  slides rather than running off the pane.
- `xdg_popup.grab` is honoured: a pointer button pressed outside a grabbing
  popup's chain sends `popup_done` and is swallowed. `Glasspane::dismiss_popup`
  and `dismiss_popups_of` do the same without a click.
- `xdg_popup.reposition` is answered, and the popup moves.
- `input::Target` and the `*_at` input entry points, for aiming pointer and
  keyboard at a popup in its own buffer pixels. The `TenantId` entry points
  are unchanged.
- `Error::NoSuchPopup`.
- Primary selection: the `zwp_primary_selection_device_manager_v1` global,
  offered to whichever tenant has the keyboard, as the clipboard already was.
  Select in one tenant and middle-click in another and the text crosses.

### Changed

- The focused tenant is told it is focused: `xdg_toplevel` carries the
  `activated` state, and keeps it while one of its own popups has the
  keyboard. A browser only offers a menu to a window it believes is active.
- Every surface is sent `wl_surface.enter` for the one output when it is
  mapped, toplevel or popup. A client with no `enter` has to guess its scale.

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
