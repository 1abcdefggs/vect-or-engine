//! vect-or-engine library crate.
//!
//! All domain modules are declared here and re-exported so that:
//!   - `src/main.rs` (stdin/stdout IPC binary) and
//!   - `napi/src/lib.rs` (napi-rs native addon, Phase 2)
//! can both depend on this library without code duplication.

pub mod engine_error;
pub mod profile;
pub mod validator;
pub mod translator;
pub mod hnsw_index;
pub mod knowledge_store;

// Convenient top-level re-exports.
pub use engine_error::EngineError;
pub use hnsw_index::HnswIndex;
pub use knowledge_store::{KnowledgeItem, KnowledgeStore, SearchResult};
pub use profile::Profile;
pub use translator::{TranslationMatch, Translator};
pub use validator::{ValidationMarker, ValidationResult, Validator};

#[cfg(test)]
mod tests {
    #[test]
    fn lib_compiles() {
        // Smoke test — ensures all modules are reachable.
        let _ = super::Profile::default();
    }
}
