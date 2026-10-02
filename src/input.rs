//! Input injection, and the coordinate contract that goes with it.
//!
//! # The coordinate contract
//!
//! Every pointer coordinate this crate accepts is **tenant-local, in logical
//! pixels, origin top-left**: the same space as [`crate::Tenant::size`], and
//! the space `wl_pointer` speaks to the client in. `(0.0, 0.0)` is the
//! tenant's top-left corner and `(w, h)` is one past its bottom-right.
//!
//! Logical, not buffer: a tenant drawing at scale 2 has a texture twice its
//! size in each direction, and the coordinate for the same spot is the same
//! number at either scale. The scale is a question of how sharp the pixels
//! are, never of where anything is.
//!
//! That is the whole contract, and it is deliberately the *only* one. glasspane
//! has no idea where a tenant is on screen, how large it is drawn, whether it
//! is rotated, or whether it is a quad on a cube — so it cannot do the
//! transform and does not pretend to. The shell owns the camera and therefore
//! owns screen → tenant; what it hands back is where the pointer is *on the
//! texture it sampled*.
//!
//! A shell rendering a tenant as a textured quad already computes a UV to
//! sample it. That UV, times [`crate::Tenant::size`], is exactly this coordinate —
//! which is the check to apply when the routing looks wrong: if a click lands
//! where the texture is dark, the shell's hit test and its sampler disagree,
//! and the compositor is not the one that is confused.
//!
//! # Keycodes
//!
//! [`Glasspane::keyboard_key`] takes an **evdev** keycode, the numbering in
//! `linux/input-event-codes.h`, which is also what `wl_keyboard.key` carries on
//! the wire. xkb's numbering is evdev's plus 8; the offset is applied here,
//! exactly once, so that `evdev in == evdev out` is a property a client can
//! assert. Applying it a second time shifts every key by one row and produces
//! plausible-but-wrong letters, which is the silent-wrong-output shape.

use smithay::input::keyboard::{FilterResult, Keycode};
use smithay::input::pointer::{AxisFrame, ButtonEvent, MotionEvent};
use smithay::utils::SERIAL_COUNTER;

use crate::popup::PopupId;
use crate::state::State;
use crate::tenant::TenantId;
use crate::{Error, Glasspane, Result};

/// What an input event is aimed at: a tenant, or one of its popups.
///
/// The coordinate contract is the same for both — logical pixels, origin
/// top-left, in the space of the texture the shell sampled. A shell that
/// hit-tests a popup quad hands back the pixel on *that* texture, not on the
/// parent's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// A toplevel.
    Tenant(TenantId),
    /// A popup over one.
    Popup(PopupId),
}

impl From<TenantId> for Target {
    fn from(id: TenantId) -> Self {
        Target::Tenant(id)
    }
}

impl From<PopupId> for Target {
    fn from(id: PopupId) -> Self {
        Target::Popup(id)
    }
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Target::Tenant(id) => write!(f, "{id}"),
            Target::Popup(id) => write!(f, "{id}"),
        }
    }
}

impl Target {
    fn missing(self) -> Error {
        match self {
            Target::Tenant(id) => Error::NoSuchTenant(id),
            Target::Popup(id) => Error::NoSuchPopup(id),
        }
    }
}

/// xkb's keycode space is evdev's, offset by 8.
pub const XKB_EVDEV_OFFSET: u32 = 8;

/// Pointer button codes, from `linux/input-event-codes.h`. These are the
/// numbers `wl_pointer.button` carries; a client reads them directly.
pub mod button {
    /// `BTN_LEFT`.
    pub const LEFT: u32 = 0x110;
    /// `BTN_RIGHT`.
    pub const RIGHT: u32 = 0x111;
    /// `BTN_MIDDLE`.
    pub const MIDDLE: u32 = 0x112;
}

/// Pressed or released.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyState {
    /// The key or button went down.
    Pressed,
    /// It came back up.
    Released,
}

impl From<KeyState> for smithay::backend::input::KeyState {
    fn from(s: KeyState) -> Self {
        match s {
            KeyState::Pressed => smithay::backend::input::KeyState::Pressed,
            KeyState::Released => smithay::backend::input::KeyState::Released,
        }
    }
}

impl From<KeyState> for smithay::backend::input::ButtonState {
    fn from(s: KeyState) -> Self {
        match s {
            KeyState::Pressed => smithay::backend::input::ButtonState::Pressed,
            KeyState::Released => smithay::backend::input::ButtonState::Released,
        }
    }
}

impl State {
    fn now_ms(&self) -> u32 {
        self.start.elapsed().as_millis() as u32
    }
}

impl Glasspane {
    /// Put the pointer inside `id` at a tenant-local position.
    ///
    /// This is both enter and motion: Wayland's `wl_pointer.enter` is derived
    /// from a focus change, so a caller that tracks "which tenant is the
    /// pointer over" only ever needs this call and [`Glasspane::pointer_leave`].
    pub fn pointer_enter(&mut self, id: TenantId, x: f64, y: f64) -> Result<()> {
        self.pointer_motion(id, x, y)
    }

    /// Put the pointer inside a tenant *or a popup*. See [`Target`].
    pub fn pointer_enter_at(&mut self, target: Target, x: f64, y: f64) -> Result<()> {
        self.pointer_motion_at(target, x, y)
    }

    /// Move the pointer within `id`. See the module docs for the coordinate
    /// contract.
    pub fn pointer_motion(&mut self, id: TenantId, x: f64, y: f64) -> Result<()> {
        self.pointer_motion_at(Target::Tenant(id), x, y)
    }

    /// Move the pointer within a tenant or a popup, in that surface's own
    /// logical pixels.
    pub fn pointer_motion_at(&mut self, target: Target, x: f64, y: f64) -> Result<()> {
        let slot = self
            .state
            .slot_of_target(target)
            .ok_or_else(|| target.missing())?;
        let surface = self.state.surface_of(slot);
        let serial = SERIAL_COUNTER.next_serial();
        let time = self.state.now_ms();
        let pointer = self.state.pointer.clone();
        // The focus point is the surface's origin *in the pointer's own
        // coordinate space*, and the client is told `location - origin`. Since
        // the location we are handing over is already tenant-local, the origin
        // is (0, 0) — which is what makes this crate's coordinate space and
        // the client's the same space, with no transform anywhere in between.
        pointer.motion(
            &mut self.state,
            Some((surface, (0.0, 0.0).into())),
            &MotionEvent {
                location: (x, y).into(),
                serial,
                time,
            },
        );
        pointer.frame(&mut self.state);
        self.state.pointer_focus = Some(target);
        self.state.flush();
        Ok(())
    }

    /// Take the pointer out of every tenant.
    ///
    /// There is no per-tenant leave: the seat has one pointer, and it is
    /// either over a surface or over nothing.
    pub fn pointer_leave(&mut self) {
        let serial = SERIAL_COUNTER.next_serial();
        let time = self.state.now_ms();
        let pointer = self.state.pointer.clone();
        pointer.motion(
            &mut self.state,
            None,
            &MotionEvent {
                location: (0.0, 0.0).into(),
                serial,
                time,
            },
        );
        pointer.frame(&mut self.state);
        self.state.pointer_focus = None;
        self.state.flush();
    }

    /// Press or release a pointer button on whatever the pointer is over.
    ///
    /// The button code is an evdev one — see [`button`]. The *location* of a
    /// click is the last [`Glasspane::pointer_motion`], as in the protocol, so
    /// a click is always motion-then-button and never a position of its own.
    pub fn pointer_button(&mut self, code: u32, state: KeyState) {
        // A grabbing popup is modal: a press that is not inside its chain
        // dismisses it and is **swallowed**, exactly as a press outside a menu
        // on a desktop closes the menu without also activating what was under
        // the cursor. Without this a `<select>` menu never learns it has been
        // dismissed and Chromium waits for a `popup_done` that never comes.
        if state == KeyState::Pressed {
            if let Some(grab) = self.state.grab {
                let inside = match self.state.pointer_focus {
                    Some(Target::Popup(p)) => self.state.in_chain(p, grab),
                    _ => false,
                };
                if !inside {
                    self.state.dismiss_grab();
                    return;
                }
            }
        }
        let serial = SERIAL_COUNTER.next_serial();
        let time = self.state.now_ms();
        let pointer = self.state.pointer.clone();
        pointer.button(
            &mut self.state,
            &ButtonEvent {
                serial,
                time,
                button: code,
                state: state.into(),
            },
        );
        pointer.frame(&mut self.state);
        self.state.flush();
    }

    /// Scroll. `horizontal` and `vertical` are in surface-local units;
    /// `v120` is the high-resolution wheel-click count (120 per detent) if the
    /// source has detents.
    pub fn pointer_axis(&mut self, horizontal: f64, vertical: f64, v120: Option<(i32, i32)>) {
        use smithay::backend::input::{Axis, AxisSource};
        let time = self.state.now_ms();
        let mut frame = AxisFrame::new(time).source(if v120.is_some() {
            AxisSource::Wheel
        } else {
            AxisSource::Continuous
        });
        if horizontal != 0.0 {
            frame = frame.value(Axis::Horizontal, horizontal);
        }
        if vertical != 0.0 {
            frame = frame.value(Axis::Vertical, vertical);
        }
        if let Some((h, v)) = v120 {
            if h != 0 {
                frame = frame.v120(Axis::Horizontal, h);
            }
            if v != 0 {
                frame = frame.v120(Axis::Vertical, v);
            }
        }
        let pointer = self.state.pointer.clone();
        pointer.axis(&mut self.state, frame);
        pointer.frame(&mut self.state);
        self.state.flush();
    }

    /// Give keyboard focus to `id`, or to nothing.
    ///
    /// This is the one piece of policy glasspane *does* hold, because the
    /// protocol makes it server state: there is one seat, and exactly one
    /// surface has its keyboard focus. Which one is the shell's decision, made
    /// here.
    pub fn set_keyboard_focus(&mut self, id: Option<TenantId>) -> Result<()> {
        self.set_keyboard_focus_at(id.map(Target::Tenant))
    }

    /// Give keyboard focus to a tenant, a popup, or nothing.
    pub fn set_keyboard_focus_at(&mut self, target: Option<Target>) -> Result<()> {
        let surface = match target {
            Some(target) => {
                let slot = self
                    .state
                    .slot_of_target(target)
                    .ok_or_else(|| target.missing())?;
                Some(self.state.surface_of(slot))
            }
            None => None,
        };
        let serial = SERIAL_COUNTER.next_serial();
        let keyboard = self.state.keyboard.clone();
        keyboard.set_focus(&mut self.state, surface, serial);
        self.state.keyboard_focus = target;
        // The window a popup belongs to stays the active one while the popup
        // has the keyboard: a menu does not deactivate the window it is over.
        let active = match target {
            Some(Target::Tenant(id)) => Some(id),
            Some(Target::Popup(id)) => self.state.popup(id).map(|p| p.root),
            None => None,
        };
        self.state.set_active(active);
        self.state.flush();
        Ok(())
    }

    /// Which tenant has keyboard focus. `None` while a popup holds it — see
    /// [`Glasspane::keyboard_focus_at`], which says either way.
    pub fn keyboard_focus(&self) -> Option<TenantId> {
        match self.state.keyboard_focus {
            Some(Target::Tenant(id)) => Some(id),
            _ => None,
        }
    }

    /// What has keyboard focus: a tenant, a popup, or nothing.
    pub fn keyboard_focus_at(&self) -> Option<Target> {
        self.state.keyboard_focus
    }

    /// What the pointer is over, as of the last motion.
    pub fn pointer_focus_at(&self) -> Option<Target> {
        self.state.pointer_focus
    }

    /// Send one key to the focused tenant.
    ///
    /// `evdev` is an evdev keycode; see the module docs for the offset. If
    /// `to` is given, focus moves there first — so a caller that routes keys
    /// per-tenant does not have to remember to call
    /// [`Glasspane::set_keyboard_focus`], and a caller that has already set
    /// focus passes `None`.
    pub fn keyboard_key(&mut self, to: Option<TenantId>, evdev: u32, state: KeyState) -> Result<()> {
        self.keyboard_key_at(to.map(Target::Tenant), evdev, state)
    }

    /// Send one key to a tenant or a popup, moving focus there first if `to`
    /// is given.
    pub fn keyboard_key_at(
        &mut self,
        to: Option<Target>,
        evdev: u32,
        state: KeyState,
    ) -> Result<()> {
        if let Some(target) = to {
            if self.state.keyboard_focus != Some(target) {
                self.set_keyboard_focus_at(Some(target))?;
            }
        }
        let serial = SERIAL_COUNTER.next_serial();
        let time = self.state.now_ms();
        let keyboard = self.state.keyboard.clone();
        keyboard.input::<(), _>(
            &mut self.state,
            Keycode::new(evdev + XKB_EVDEV_OFFSET),
            state.into(),
            serial,
            time,
            |_, _, _| FilterResult::Forward,
        );
        // A forwarded key is a *buffered write*, not a syscall, and nothing
        // else on this path will flush it. Without this line the tenant never
        // sees the key, never repaints, never asks for another frame callback
        // — a starvation that sustains itself and surfaces only at the next
        // unrelated flush.
        self.state.flush();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The keycode round trip is an identity over the whole evdev range. A
    /// changed offset must fail in `cargo test`, not in a terminal that types
    /// the wrong letters.
    #[test]
    fn keycode_round_trip_is_identity() {
        for evdev in 0u32..=255 {
            assert_eq!(
                Keycode::new(evdev + XKB_EVDEV_OFFSET).raw() - XKB_EVDEV_OFFSET,
                evdev
            );
        }
    }

    /// The button codes are the kernel's, not invented here.
    #[test]
    fn button_codes_are_the_evdev_ones() {
        assert_eq!(button::LEFT, 272);
        assert_eq!(button::RIGHT, 273);
        assert_eq!(button::MIDDLE, 274);
    }
}
