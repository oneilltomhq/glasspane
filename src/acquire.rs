//! The acquire wait, on the commit path.
//!
//! **Sharing is not ordering.** Importing a client's dma-buf tells the driver
//! where the bytes are; it says nothing about *when* they are valid. On Mesa
//! `anv` a narrow implicit-sync conjunction — i915 KMD, dedicated allocation,
//! a `COLOR_ATTACHMENT` producer, imported memory — supplies that ordering for
//! free, which is exactly why an unfenced compositor passes its own test and
//! proves nothing. Every term of that conjunction is outside this crate's
//! control.
//!
//! So the wait is explicit, and it happens here:
//!
//! ```text
//!   client commits a dmabuf-backed wl_buffer
//!        ├─ export_sync_file(plane 0 fd, Read)   ← the kernel's fences
//!        ├─ AcquireSemaphore::import(fence)
//!        └─ wait_on_next_submit(queue)
//!                └─ the next vkQueueSubmit on the shell's queue carries it —
//!                   which is the blit into the tenant texture, and after that
//!                   the shell's own frame.
//! ```
//!
//! # Four things that are easy to get wrong here
//!
//! **One semaphore per process is one too few.** `SYNC_FD` import is
//! necessarily *temporary*: importing a second payload replaces the first
//! before any wait consumed it, silently dropping a client's fence. The pool
//! is therefore keyed per `wl_buffer` — bounded by live buffers, not by
//! frames.
//!
//! **`vkImportSemaphoreFdKHR` needs the consuming submit retired**
//! (`VUID-vkImportSemaphoreFdKHR-semaphore-01142`), and wgpu never retires a
//! submit unless something polls. One `device.poll` before the import clears
//! it, skipped on a buffer's first commit where nothing can be referencing a
//! semaphore created two lines ago. The poll is *to that submit*, the blit
//! [`AcquireGate::carried_by`] was told about, and not to the end of the
//! queue: draining the queue waits for every other tenant's blit, each gated
//! on its own client's GPU frame, and for the shell's frame besides. With a
//! terminal and a browser both busy that was over a millisecond on a fifth of
//! commits and 67 ms at worst, spent in one tenant's commit on another's
//! rendering.
//!
//! **A binary semaphore waited twice in one submit deadlocks.** A client can
//! commit many times between two submits; naively staging a wait per commit
//! puts N copies of one `VkSemaphore` in one submit's wait list, the first
//! consumes the single signal, and the queue wedges with no error at all. So
//! the previous wait is *withdrawn* first — which loses nothing, because
//! `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` returns all the buffer's outstanding write
//! fences at the moment it is called, so the new fence subsumes the old.
//!
//! **The wait can be free, and counting waits would not show it.** The ioctl
//! substitutes an already-signalled stub for an empty reservation object, so a
//! run can export, import, wait, count N waits and have ordered nothing.
//! `AcquireGate::already_signalled` is the number that says so.
//!
//! # Scope
//!
//! This is an acquire path for **implicit-sync clients**. `DMA_BUF_SYNC_READ`
//! resolves to the `DMA_RESV_USAGE_WRITE` fences; a client that has opted into
//! explicit synchronisation attaches its fences as `DMA_RESV_USAGE_BOOKKEEP`,
//! which this ioctl does not report. glasspane therefore does **not** advertise
//! `linux-drm-syncobj-v1`: doing so would make Mesa and Chromium switch, and
//! this gate would go hollow while every counter stayed green.

use std::collections::HashMap;
use std::os::fd::BorrowedFd;

use wgpu_dmabuf::sync::{self, AcquireSemaphore, SyncDirection};

use crate::registry::BufferKey;

/// Where the wait blocks.
///
/// `ALL_COMMANDS` rather than `FRAGMENT_SHADER` because the very next thing
/// this crate does with a client texture is a *copy* into the tenant texture,
/// not a sample. Gating only the fragment stage would leave that transfer
/// unguarded, which is the almost-right shape that passes every test.
const WAIT_STAGE: wgpu_dmabuf::ash::vk::PipelineStageFlags =
    wgpu_dmabuf::ash::vk::PipelineStageFlags::ALL_COMMANDS;

struct Slot {
    semaphore: AcquireSemaphore,
    /// Which submit consumed this semaphore's last staged wait, and so has to
    /// retire before the next import.
    carrier: Carrier,
}

/// What a slot's last staged wait was carried by.
enum Carrier {
    /// Nothing was ever staged: no submit can be referencing the semaphore,
    /// so no retire poll is owed.
    Never,
    /// Staged, and not followed by a blit this gate was told about: an import
    /// that failed after the wait went up leaves it for whichever submit comes
    /// next. Retiring that means draining the queue.
    Unknown,
    /// Staged and consumed by this submit, which is the only one that has to
    /// retire.
    Submit(wgpu::SubmissionIndex),
}

impl Carrier {
    /// Wait until no submit refers to the semaphore any more. `superseded`
    /// says the last wait was withdrawn before any submit took it, in which
    /// case the one before it was retired ahead of the import that replaced
    /// it, and nothing is owed.
    fn retire(&self, device: &wgpu::Device, superseded: bool) -> Result<(), wgpu::PollError> {
        if superseded {
            return Ok(());
        }
        let submission_index = match self {
            Carrier::Never => return Ok(()),
            Carrier::Unknown => None,
            Carrier::Submit(index) => Some(index.clone()),
        };
        device
            .poll(wgpu::PollType::Wait {
                submission_index,
                timeout: None,
            })
            .map(drop)
    }
}

/// The per-`wl_buffer` acquire-semaphore pool and its ledger.
pub(crate) struct AcquireGate {
    device: wgpu::Device,
    queue: wgpu::Queue,
    slots: HashMap<BufferKey, Slot>,
    /// `sync_file`s exported from client dma-bufs; one per dma-buf commit.
    pub exports: u64,
    /// Waits actually staged on the shell's queue.
    pub waits: u64,
    /// Of `waits`, how many were withdrawn before any submit consumed them.
    /// `waits - superseded` is the number a `vkQueueSubmit` actually carried —
    /// "we staged 300 waits" and "300 submits waited" are very different
    /// claims, and only the second is about ordering.
    pub superseded: u64,
    /// Of `exports`, how many were already signalled at export time: the
    /// anti-vacuity number.
    pub already_signalled: u64,
}

impl AcquireGate {
    /// Mint the gate, or say why this device cannot fence.
    ///
    /// A probe semaphore is created and dropped here purely so that
    /// "`VK_KHR_external_semaphore_fd` is enabled and `SYNC_FD` is importable"
    /// is answered at construction, on one line, rather than on the first
    /// client commit in the middle of a dispatch. Stock `wgpu::request_device`
    /// never enables that extension; open the device with
    /// [`wgpu_dmabuf::device::open_device`].
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, wgpu_dmabuf::Error> {
        drop(AcquireSemaphore::new(device)?);
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            slots: HashMap::new(),
            exports: 0,
            waits: 0,
            superseded: 0,
            already_signalled: 0,
        })
    }

    /// Export `plane`'s implicit fences and stage them as a wait on the
    /// shell's queue, so nothing touches this buffer before the client's GPU
    /// work has landed.
    ///
    /// Every failure is returned, never logged-and-continued. `ENOTTY` means
    /// the kernel predates dma-buf explicit sync; anything else means the
    /// driver refused a fence we are contractually required to wait on. Both
    /// are the same answer: this commit is rejected.
    pub fn wait_for(&mut self, key: BufferKey, plane: BorrowedFd<'_>) -> Result<(), String> {
        let fence = sync::export_sync_file(plane, SyncDirection::Read).map_err(|e| {
            format!(
                "DMA_BUF_IOCTL_EXPORT_SYNC_FILE on a client buffer failed: {e}. \
                 glasspane hard-requires it; sampling client memory with nothing \
                 ordering it is not a fallback."
            )
        })?;
        self.exports += 1;
        if fence.already_signalled {
            self.already_signalled += 1;
        }

        let first = !self.slots.contains_key(&key);
        if first {
            let semaphore = AcquireSemaphore::new(&self.device)
                .map_err(|e| format!("acquire semaphore for a client buffer: {e}"))?;
            self.slots.insert(
                key,
                Slot {
                    semaphore,
                    carrier: Carrier::Never,
                },
            );
        }
        let slot = self.slots.get_mut(&key).expect("just inserted");

        let superseded = slot
            .semaphore
            .cancel_pending_wait(&self.queue)
            .map_err(|e| format!("withdrawing the previous acquire wait: {e}"))?;
        if superseded {
            self.superseded += 1;
        }
        // The previous wait was consumed by a submit that may not have
        // retired, and importing a SYNC_FD payload into a semaphore an
        // unretired submit refers to is a validation error. One poll clears
        // it.
        slot.carrier
            .retire(&self.device, superseded)
            .map_err(|e| format!("retire poll before acquire import: {e:?}"))?;
        slot.semaphore
            .import(fence.sync_file)
            .map_err(|e| format!("importing a client sync_file: {e}"))?;
        slot.semaphore
            .wait_on_next_submit(&self.queue, WAIT_STAGE)
            .map_err(|e| format!("staging a client acquire wait: {e}"))?;
        slot.carrier = Carrier::Unknown;
        self.waits += 1;
        Ok(())
    }

    /// Say which submit carried the wait [`Self::wait_for`] just staged for
    /// `key`: the blit straight after it. The next commit of the same buffer
    /// then polls to this submit and no further.
    ///
    /// Call it only for the submit that immediately followed the staging. A
    /// later one would let the next import go ahead with the real carrier
    /// still in flight.
    pub fn carried_by(&mut self, key: BufferKey, index: wgpu::SubmissionIndex) {
        if let Some(slot) = self.slots.get_mut(&key) {
            slot.carrier = Carrier::Submit(index);
        }
    }

    /// Drop the semaphore for a destroyed `wl_buffer`.
    ///
    /// **Two things happen before `vkDestroySemaphore`, not one.** The submit
    /// that carried its last wait is retired, because destroying a semaphore
    /// an unretired submit refers to is a use-after-free the validation layer
    /// catches and the driver does not. But `poll` retires *submits*; it does
    /// not touch the queue's list of waits staged and not yet submitted, and a
    /// buffer is routinely destroyed with one still staged. So the staged wait
    /// is withdrawn first, and only then is the carrier retired and the
    /// semaphore dropped.
    pub fn forget(&mut self, key: BufferKey) {
        if let Some(slot) = self.slots.remove(&key) {
            if !matches!(slot.carrier, Carrier::Never) {
                let superseded = slot
                    .semaphore
                    .cancel_pending_wait(&self.queue)
                    .unwrap_or(false);
                if superseded {
                    self.superseded += 1;
                }
                let _ = slot.carrier.retire(&self.device, superseded);
            }
        }
    }

    /// Live semaphores — i.e. dma-buf `wl_buffer`s seen and not yet destroyed.
    /// This is what makes "the pool is bounded by buffers, not by frames"
    /// something a run can assert rather than claim.
    pub fn live_semaphores(&self) -> usize {
        self.slots.len()
    }

    /// Release every semaphore, with every staged wait withdrawn and the queue
    /// drained first.
    ///
    /// Called from `Drop`, because the last commit before shutdown almost
    /// always stages a wait that no submit ever consumes, and without the
    /// withdrawal the `VkSemaphore` is destroyed while its handle is still in
    /// the hal queue's pending-wait list.
    pub fn shutdown(&mut self) {
        if !self.slots.is_empty() {
            let mut withdrawn = 0;
            for slot in self.slots.values() {
                if !matches!(slot.carrier, Carrier::Never)
                    && slot
                        .semaphore
                        .cancel_pending_wait(&self.queue)
                        .unwrap_or(false)
                {
                    withdrawn += 1;
                }
            }
            self.superseded += withdrawn;
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
        self.slots.clear();
    }
}

/// A run's acquire ledger, for a harness that wants to fail on a gate that
/// passed for the wrong reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AcquireReport {
    /// `sync_file`s exported from client dma-bufs.
    pub exports: u64,
    /// Waits staged on the queue.
    pub waits: u64,
    /// Waits withdrawn before any submit consumed them.
    pub superseded: u64,
    /// Waits a `vkQueueSubmit` actually carried.
    pub carried: u64,
    /// Exports that were already signalled, and so could not have blocked.
    pub already_signalled: u64,
    /// Live per-buffer semaphores.
    pub live_semaphores: usize,
}

impl std::fmt::Display for AcquireReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "acquire: exports={} waits={} carried={} superseded={} already_signalled={} \
             live_semaphores={}",
            self.exports,
            self.waits,
            self.carried,
            self.superseded,
            self.already_signalled,
            self.live_semaphores
        )
    }
}

impl AcquireGate {
    /// This gate's ledger.
    pub fn report(&self) -> AcquireReport {
        AcquireReport {
            exports: self.exports,
            waits: self.waits,
            superseded: self.superseded,
            carried: self.waits.saturating_sub(self.superseded),
            already_signalled: self.already_signalled,
            live_semaphores: self.live_semaphores(),
        }
    }
}
