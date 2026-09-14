//! The import-once registry: one `VkImage` per `wl_buffer`, for the life of
//! that buffer.
//!
//! **The rule.** A client that re-commits the same `wl_buffer` — which is what
//! every double-buffered client does, forever — must produce exactly one
//! dma-buf import and then cache hits. Importing again per commit would leak a
//! `VkImage` per frame and would also hand the consumer a different
//! `wgpu::Texture` for the same pixels, which is the shape of a scene that
//! rebinds every frame for no reason.
//!
//! This is a cache of *source* textures. It is not what the consumer sees:
//! that is the tenant's own stable texture (see [`crate::Tenant::texture`]),
//! which these are copied into. Keeping the two separate is what lets both
//! laws hold at once — import-once per buffer, and one stable texture handle
//! per tenant.
//!
//! `wl_shm` buffers never enter here. An shm buffer is CPU memory that has to
//! be copied whatever happens, so there is no import to amortise; it goes
//! straight into the tenant texture through `write_texture`.

use std::collections::HashMap;
use std::sync::Arc;

/// Key for the buffer cache, derived from the protocol object id.
///
/// The id carries wayland-backend's reuse counter, so a destroyed-and-recreated
/// `wl_buffer` never aliases the old entry's texture — which would be a texture
/// that samples cleanly and shows another buffer's picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferKey(pub u64);

/// Process-unique, monotonically issued, never reused within a run.
///
/// Identity, not count: a caller that wants to know whether an import happened
/// diffs *ids*, because "the last entry" of a map is not "the newest entry".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureId(pub u64);

impl std::fmt::Display for TextureId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tex#{}", self.0)
    }
}

/// One imported client buffer.
#[derive(Debug, Clone)]
pub struct Entry {
    /// This import's identity.
    pub id: TextureId,
    /// The imported image. Shared because a commit hands it out while the
    /// registry keeps it.
    pub texture: Arc<wgpu::Texture>,
    /// Commits that resolved to this entry without re-importing.
    pub cache_hits: u64,
}

/// Owns imported textures for the lifetime of the buffers behind them.
#[derive(Debug, Default)]
pub struct Registry {
    by_buffer: HashMap<BufferKey, Entry>,
    next_id: u64,
    imports: u64,
    hits: u64,
}

impl Registry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Import-or-lookup. `import` is called only on a miss; that is the whole
    /// point of this type.
    ///
    /// A failing import does not poison the cache: nothing is inserted, the
    /// error is returned, and the next commit of the same buffer tries again.
    pub fn get_or_import<E>(
        &mut self,
        key: BufferKey,
        import: impl FnOnce() -> std::result::Result<wgpu::Texture, E>,
    ) -> std::result::Result<(TextureId, Arc<wgpu::Texture>, bool), E> {
        if let Some(entry) = self.by_buffer.get_mut(&key) {
            entry.cache_hits += 1;
            self.hits += 1;
            return Ok((entry.id, entry.texture.clone(), false));
        }
        let texture = Arc::new(import()?);
        self.next_id += 1;
        let id = TextureId(self.next_id);
        self.imports += 1;
        self.by_buffer.insert(
            key,
            Entry {
                id,
                texture: texture.clone(),
                cache_hits: 0,
            },
        );
        Ok((id, texture, true))
    }

    /// Drop the entry for a destroyed `wl_buffer`, releasing the texture.
    pub fn forget(&mut self, key: BufferKey) -> Option<Entry> {
        self.by_buffer.remove(&key)
    }

    /// Look up an entry without importing.
    pub fn get(&self, key: BufferKey) -> Option<&Entry> {
        self.by_buffer.get(&key)
    }

    /// Live entries — i.e. client buffers imported and not yet destroyed.
    pub fn len(&self) -> usize {
        self.by_buffer.len()
    }

    /// Whether nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.by_buffer.is_empty()
    }

    /// Imports performed over the run (cache misses).
    pub fn import_count(&self) -> u64 {
        self.imports
    }

    /// Commits that hit the cache.
    pub fn cache_hit_count(&self) -> u64 {
        self.hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cache decision is pure, so it is testable without a GPU. A failed
    /// import must leave the cache untouched — otherwise one transient driver
    /// error would pin a broken entry for the buffer's whole life.
    #[test]
    fn a_failed_import_does_not_poison_the_cache() {
        let mut reg = Registry::new();
        let mut calls = 0u32;
        for _ in 0..3 {
            let r = reg.get_or_import::<()>(BufferKey(1), || {
                calls += 1;
                Err(())
            });
            assert!(r.is_err());
        }
        assert_eq!(calls, 3, "every commit must retry a failed import");
        assert_eq!(reg.import_count(), 0);
        assert_eq!(reg.cache_hit_count(), 0);
        assert!(reg.is_empty());
    }

}
