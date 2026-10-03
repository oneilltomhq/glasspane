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
- `Glasspane::set_scale`: ask a tenant, and its popups, to draw at a scale,
  for a shell that draws it larger than its logical size. Sent as
  `wp_fractional_scale_v1.preferred_scale` and, rounded up, as
  `wl_surface.preferred_buffer_scale`; the `wp_fractional_scale_manager_v1`
  and `wp_viewporter` globals are offered and `wl_compositor` is version 6
  for it. `Error::BadScale` for a scale that is not a number above zero.
- `Tenant::buffer_size`, `Tenant::scale` and `Popup::buffer_size`.
- A `wp_viewport` source rectangle is honoured: the texture is that part of
  the buffer. Chromium, offered `wp_viewporter`, keeps a bigger buffer when a
  window shrinks and crops it, and without this was drawn whole and squeezed.
- `Glasspane::tick_where`: answer the frame callbacks of the tenants a
  predicate picks and hold the rest, so a shell can pace panes nobody sees
  (a held callback is answered the tick its tenant is next picked). A popup's
  callbacks count as its root tenant's. `tick` is `tick_where` picking all.
- `Tenant::callbacks_sent`: the frame callbacks answered for a tenant and its
  popups.

### Changed

- The focused tenant is told it is focused: `xdg_toplevel` carries the
  `activated` state, and keeps it while one of its own popups has the
  keyboard. A browser only offers a menu to a window it believes is active.
- Every surface is sent `wl_surface.enter` for the one output when it is
  mapped, toplevel or popup. A client with no `enter` has to guess its scale.
- Every toplevel is configured tiled on all four edges, so it takes the size
  it is given. A floating foot rounded its size down to whole cells, and on a
  scale change kept its grid and resized itself around the new font.
- `Tenant::size`, `Popup::size`, popup positions and every input coordinate
  are **logical** pixels: the configured size, and what `wl_pointer` speaks.
  At scale 1 nothing moves. Above it the texture is larger than the size, by
  the scale; sample it by UV and hit-test by `size`.
- `Event::Resized` and `Event::PopupResized` fire when the logical size or
  the buffer's moves, so a repaint at a new scale is announced with the same
  `size`. Their fields are unchanged.

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
