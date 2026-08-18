//! Knowledge base storage — Phase 3.
//!
//! Optimisations layered in this module:
//!  1. **memmap2**        — zero-copy memory-mapped I/O for JSON loading.
//!  2. **f16 quantisation** — vectors stored as 16-bit floats (50 % memory).
//!  3. **rayon**          — parallel linear scan for mid-size datasets.
//!  4. **HnswIndex**      — O(log N) approximate search for large datasets.
//!  5. **SQLite cache**   — avoids re-parsing JSON on every restart.

use half::f16;
use memmap2::MmapOptions;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;

use crate::engine_error::EngineError;
use crate::hnsw_index::HnswIndex;

// ─── Public data types ────────────────────────────────────────────────────────

/// Full deserialization target — used only during JSON loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeItem {
    pub id: Option<String>,
    /// Full-precision f32 vector (only present in JSON source).
    pub vector: Vec<f32>,
    #[serde(flatten)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// Slim runtime representation — metadata only, vector stored separately.
#[derive(Debug, Clone)]
pub struct SlimItem {
    pub id: Option<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

// ─── KnowledgeStore ───────────────────────────────────────────────────────────

/// The central in-memory knowledge base with pluggable search backends.
pub struct KnowledgeStore {
    /// Metadata for each item (id + arbitrary JSON fields).
    pub items: Vec<SlimItem>,
    /// f16-quantised vectors — parallel to `items`.
    vectors_f16: Vec<Vec<f16>>,
    /// HNSW index (built on demand via `build_index`).
    hnsw: Option<HnswIndex>,
    /// Expected vector dimensionality (set on first load).
    pub dim: usize,
}

impl Default for KnowledgeStore {
    fn default() -> Self {
        KnowledgeStore::new()
    }
}

impl KnowledgeStore {
    pub fn new() -> Self {
        KnowledgeStore {
            items: Vec::new(),
            vectors_f16: Vec::new(),
            hnsw: None,
            dim: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Whether an HNSW index has been built.
    pub fn has_index(&self) -> bool {
        self.hnsw.is_some()
    }

    // ── Loading ───────────────────────────────────────────────────────────────

    /// Load knowledge base from a JSON file using zero-copy memory-mapped I/O.
    ///
    /// The file is parsed once; vectors are quantised to f16 on the fly.
    /// Returns the number of items loaded.
    pub fn load_from_path(&mut self, path: &str) -> Result<usize, EngineError> {
        let file = fs::File::open(path)?;
        // SAFETY: we do not mutate the file while the mmap lives.
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let raw: Vec<KnowledgeItem> = serde_json::from_slice(&mmap)?;
        let count = raw.len();

        if count == 0 {
            return Ok(0);
        }

        let new_dim = raw[0].vector.len();

        // Reset state.
        self.dim = new_dim;
        self.hnsw = None;
        self.items = Vec::with_capacity(count);
        self.vectors_f16 = Vec::with_capacity(count);

        for item in raw {
            // Dimension guard.
            if item.vector.len() != self.dim {
                return Err(EngineError::DimensionMismatch {
                    expected: self.dim,
                    got: item.vector.len(),
                });
            }
            // Quantise f32 → f16.
            let f16_vec: Vec<f16> = item.vector.iter().map(|&x| f16::from_f32(x)).collect();

            self.vectors_f16.push(f16_vec);
            self.items.push(SlimItem {
                id: item.id,
                metadata: item.metadata,
            });
        }

        Ok(count)
    }

    // ── Index building ────────────────────────────────────────────────────────

    /// Build (or rebuild) the HNSW approximate-nearest-neighbour index.
    ///
    /// **CPU-intensive** — call this inside `tokio::task::spawn_blocking`.
    pub fn build_index(&mut self) {
        // Dequantise for HNSW build (needed once; not kept afterwards).
        let vecs_f32: Vec<Vec<f32>> = self
            .vectors_f16
            .iter()
            .map(|v| v.iter().map(|&x| f32::from(x)).collect())
            .collect();

        self.hnsw = Some(HnswIndex::build(&vecs_f32));
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /// HNSW O(log N) search — use after calling `build_index`.
    ///
    /// Falls back to parallel linear scan if no index exists.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        if let Some(hnsw) = &self.hnsw {
            let raw = hnsw.search(query, top_k);
            raw.into_iter()
                .map(|(score, idx)| self.make_result(score, idx))
                .collect()
        } else {
            self.linear_search(query, top_k)
        }
    }

    /// Parallel rayon linear scan — O(N) with SIMD dot product.
    ///
    /// Suitable for ≤ ~10 K items or when HNSW is not yet built.
    pub fn linear_search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut scores: Vec<(f32, usize)> = self
            .vectors_f16
            .par_iter()
            .enumerate()
            .map(|(idx, v)| {
                let score = dot_product_f16(query, v);
                (score, idx)
            })
            .collect();

        // O(N) partial selection — avoids full sort.
        let n = scores.len();
        if n > top_k {
            scores.select_nth_unstable_by(top_k, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
            });
            scores.truncate(top_k);
        }
        scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        scores
            .into_iter()
            .map(|(score, idx)| self.make_result(score, idx))
            .collect()
    }

    // ── SQLite persistence ────────────────────────────────────────────────────

    /// Persist the current knowledge base metadata to a SQLite database.
    ///
    /// Subsequent startups can call `load_from_sqlite` for faster startup
    /// (avoids re-parsing the JSON file and re-quantising vectors).
    pub fn save_to_sqlite(&self, db_path: &str) -> Result<(), EngineError> {
        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| EngineError::Database(e.to_string()))?;

        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS kb_items (
                idx       INTEGER PRIMARY KEY,
                id        TEXT,
                metadata  TEXT NOT NULL,
                vector_f16 BLOB NOT NULL
            );
            DELETE FROM kb_items;
        ")
        .map_err(|e| EngineError::Database(e.to_string()))?;

        let mut stmt = conn.prepare(
            "INSERT INTO kb_items (idx, id, metadata, vector_f16) VALUES (?1,?2,?3,?4)"
        ).map_err(|e| EngineError::Database(e.to_string()))?;

        for (i, (item, f16v)) in self.items.iter().zip(self.vectors_f16.iter()).enumerate() {
            let metadata_json = serde_json::to_string(&item.metadata)?;
            // Store f16 vector as raw bytes (little-endian).
            let blob: Vec<u8> = f16v.iter().flat_map(|x| x.to_le_bytes()).collect();
            stmt.execute(rusqlite::params![i as i64, item.id, metadata_json, blob])
                .map_err(|e| EngineError::Database(e.to_string()))?;
        }

        Ok(())
    }

    /// Load knowledge base from a SQLite cache (fast restart path).
    pub fn load_from_sqlite(&mut self, db_path: &str) -> Result<usize, EngineError> {
        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| EngineError::Database(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT id, metadata, vector_f16 FROM kb_items ORDER BY idx"
        ).map_err(|e| EngineError::Database(e.to_string()))?;

        let rows: Vec<(Option<String>, String, Vec<u8>)> = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .map_err(|e| EngineError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        self.items.clear();
        self.vectors_f16.clear();
        self.hnsw = None;

        for (id, meta_json, blob) in rows {
            let metadata: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(&meta_json)?;
            let f16v: Vec<f16> = blob
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]))
                .collect();

            if self.dim == 0 {
                self.dim = f16v.len();
            }
            self.items.push(SlimItem { id, metadata });
            self.vectors_f16.push(f16v);
        }

        Ok(self.items.len())
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn make_result(&self, score: f32, idx: usize) -> SearchResult {
        let item = &self.items[idx];
        SearchResult {
            idx,
            score,
            id: item.id.clone(),
            metadata: item.metadata.clone(),
        }
    }
}

// ─── Result type ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub idx: usize,
    pub score: f32,
    pub id: Option<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl SearchResult {
    /// Convert to a `serde_json::Value` for JSON-RPC responses.
    pub fn to_json_value(&self) -> serde_json::Value {
        let mut map = self.metadata.clone();
        if let Some(id) = &self.id {
            map.insert("id".to_string(), serde_json::json!(id));
        }
        map.insert("score".to_string(), serde_json::json!(self.score));
        serde_json::Value::Object(map)
    }
}

// ─── SIMD-friendly f16 dot product ───────────────────────────────────────────

/// Compute dot product between an f32 query and an f16-stored vector.
/// 8-way unroll so the compiler can issue 256-bit SIMD loads.
#[inline(always)]
fn dot_product_f16(query: &[f32], stored: &[f16]) -> f32 {
    let n = query.len().min(stored.len());
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;

    let chunks = n / 8;
    for i in 0..chunks {
        let b = i * 8;
        unsafe {
            s0 += query.get_unchecked(b)     * f32::from(*stored.get_unchecked(b));
            s1 += query.get_unchecked(b + 1) * f32::from(*stored.get_unchecked(b + 1));
            s2 += query.get_unchecked(b + 2) * f32::from(*stored.get_unchecked(b + 2));
            s3 += query.get_unchecked(b + 3) * f32::from(*stored.get_unchecked(b + 3));
            s4 += query.get_unchecked(b + 4) * f32::from(*stored.get_unchecked(b + 4));
            s5 += query.get_unchecked(b + 5) * f32::from(*stored.get_unchecked(b + 5));
            s6 += query.get_unchecked(b + 6) * f32::from(*stored.get_unchecked(b + 6));
            s7 += query.get_unchecked(b + 7) * f32::from(*stored.get_unchecked(b + 7));
        }
    }

    let mut sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    for i in (chunks * 8)..n {
        unsafe {
            sum += query.get_unchecked(i) * f32::from(*stored.get_unchecked(i));
        }
    }
    sum
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store(n: usize, dim: usize) -> KnowledgeStore {
        let mut store = KnowledgeStore::new();
        store.dim = dim;
        for i in 0..n {
            let v: Vec<f16> = (0..dim)
                .map(|j| f16::from_f32(if j == i % dim { 1.0 } else { 0.0 }))
                .collect();
            store.vectors_f16.push(v);
            store.items.push(SlimItem {
                id: Some(format!("item-{i}")),
                metadata: serde_json::Map::new(),
            });
        }
        store
    }

    #[test]
    fn linear_search_returns_correct_top1() {
        let store = make_store(50, 64);
        let mut query = vec![0.0f32; 64];
        query[0] = 1.0;
        let results = store.linear_search(&query, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].idx, 0);
    }

    #[test]
    fn hnsw_search_returns_approximate_top1() {
        let mut store = make_store(200, 64);
        store.build_index();
        let mut query = vec![0.0f32; 64];
        query[5] = 1.0;
        let results = store.search(&query, 3);
        assert!(!results.is_empty());
        // Node 5 should appear in top results (it is an exact unit vector match).
        assert!(results.iter().any(|r| r.idx == 5));
    }
}
