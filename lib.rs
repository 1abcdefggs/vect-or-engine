// Copyright (c) 2026 1abcdefggs
// SPDX-License-Identifier: MIT
// https://github.com/1abcdefggs/vect-or-engine

//! NAPI-rs bindings for vect-or-engine (Phase 2).
//! Exposes the core Rust engine functions to Node.js as a native addon.

#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

use napi::bindgen_prelude::*;
use once_cell::sync::Lazy;
use std::sync::Arc;
use tokio::sync::RwLock;
use vect_or_engine_lib::{
    EngineError, KnowledgeStore, Profile, SearchResult, Translator, Validator, ValidationMarker,
    ValidationResult,
};

// --- Global Engine State ---

/// The single, global, thread-safe engine instance.
struct Engine {
    profile: Profile,
    validator: Validator,
    translator: Translator,
    knowledge_store: KnowledgeStore,
}

impl Engine {
    fn new(profile: Profile) -> Self {
        let validator = Validator::new(&profile);
        let translator = Translator::new(&profile);
        Self {
            profile,
            validator,
            translator,
            knowledge_store: KnowledgeStore::new(),
        }
    }
}

static ENGINE: Lazy<Arc<RwLock<Engine>>> =
    Lazy::new(|| Arc::new(RwLock::new(Engine::new(Profile::default()))));
/// A simple ping function to verify the NAPI setup.
#[napi]
pub fn ping(name: String) -> String {
  format!("pong from rust: {name}!")
}

// --- N-API Data Structures ---
// These structs mirror the internal engine structs but are annotated for NAPI.

#[napi(object)]
pub struct JsValidationMarker {
  pub line: u32,
  pub cols: Vec<u32>,
  pub rule_id: String,
  pub message: String,
}

#[napi(object)]
pub struct JsValidationResult {
  pub is_valid: bool,
  pub markers: Vec<JsValidationMarker>,
}

#[napi(object)]
pub struct JsSearchResult {
    pub idx: u32,
    pub score: f32,
    pub id: Option<String>,
    pub metadata: serde_json::Value,
}

impl From<&SearchResult> for JsSearchResult {
    fn from(r: &SearchResult) -> Self {
        Self {
            idx: r.idx as u32,
            score: r.score,
            id: r.id.clone(),
            metadata: serde_json::Value::Object(r.metadata.clone()),
        }
    }
}

#[napi(object)]
pub struct JsKbInfo {
    pub count: u32,
    pub dim: u32,
    #[napi(js_name = "hnswReady")]
    pub hnsw_ready: bool,
}

// --- Conversion Implementations ---
// Convert from internal engine types to N-API JS types.

impl From<ValidationMarker> for JsValidationMarker {
    fn from(marker: ValidationMarker) -> Self {
        Self {
            // Safe cast, line numbers won't exceed u32::MAX
            line: marker.line as u32,
            cols: marker.cols.iter().map(|&c| c as u32).collect(),
            rule_id: marker.rule_id,
            message: marker.message,
        }
    }
}

impl From<ValidationResult> for JsValidationResult {
    fn from(result: ValidationResult) -> Self {
        Self {
            is_valid: result.is_valid,
            markers: result.markers.into_iter().map(Into::into).collect(),
        }
    }
}

// --- Exposed N-API Functions ---

/// Validates a document using the currently loaded profile.
#[napi]
pub async fn validate(text: String) -> Result<JsValidationResult> {
    let engine = ENGINE.read().await;
    let result = engine.validator.validate(&text);
    Ok(result.into())
}

#[napi(task)]
pub async fn load_profile(path: String) -> Result<()> {
    verify_safe_path(&path)?;
    let content = tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to read profile: {e}")))?;
    let profile = Profile::load_from_json_str(&content)
        .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to parse profile: {e}")))?;

    let mut engine = ENGINE.write().await;
    *engine = Engine::new(profile);
    Ok(())
}

#[napi(task)]
pub async fn load_knowledge_base(path: String) -> Result<u32> {
    verify_safe_path(&path)?;
    let mut engine = ENGINE.write().await;
    let count = engine
        .knowledge_store
        .load_from_path(&path)
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(count as u32)
}

#[napi(task)]
pub async fn load_kb_cache(path: String) -> Result<u32> {
    verify_safe_path(&path)?;
    let mut engine = ENGINE.write().await;
    let count = engine
        .knowledge_store
        .load_from_sqlite(&path)
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    Ok(count as u32)
}

#[napi(task)]
pub async fn save_kb_cache(path: String) -> Result<()> {
    verify_safe_path(&path)?;
    let engine = ENGINE.read().await;
    engine
        .knowledge_store
        .save_to_sqlite(&path)
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))
}

#[napi(task)]
pub async fn build_index() -> Result<u32> {
    // Clone Arc to move it into the blocking task for CPU-intensive work.
    let engine_arc = ENGINE.clone();
    tokio::task::spawn_blocking(move || {
        // Re-lock inside the new thread.
        let mut engine = engine_arc.blocking_write();
        if engine.knowledge_store.is_empty() {
            return Err("Knowledge base is empty.".to_string());
        }
        engine.knowledge_store.build_index();
        Ok(engine.knowledge_store.len() as u32)
    })
    .await
    .map_err(|e| Error::new(Status::JoinError, e.to_string()))? // Handle task join error
    .map_err(|e| Error::new(Status::GenericFailure, e)) // Handle our custom error
}

#[napi(ts_args_type = "query: Float32Array, topK: number")]
#[napi(task)]
pub async fn search(query: Float32Array, top_k: u32) -> Result<Vec<JsSearchResult>> {
    let engine = ENGINE.read().await;
    // Pass query by reference to avoid allocation.
    let results = engine.knowledge_store.search(&query, top_k as usize);
    Ok(results.iter().map(Into::into).collect())
}

#[napi(task)]
pub async fn kb_info() -> Result<JsKbInfo> {
    let engine = ENGINE.read().await;
    Ok(JsKbInfo {
        count: engine.knowledge_store.len() as u32,
        dim: engine.knowledge_store.dim as u32,
        hnsw_ready: engine.knowledge_store.has_index(),
    })
}

/// SECURITY: Basic path safety check to prevent traversal attacks.
/// Disallows absolute paths and parent directory components (`..`).
fn verify_safe_path(path_str: &str) -> Result<()> {
    let path = std::path::Path::new(path_str);
    if path.is_absolute() {
        return Err(Error::new(
            Status::InvalidArg,
            EngineError::InvalidPath("Absolute paths are not allowed".to_string()).to_string(),
        ));
    }
    if path.components().any(|c| c == std::path::Component::ParentDir) {
        return Err(Error::new(
            Status::InvalidArg,
            EngineError::InvalidPath("Directory traversal ('..') is not allowed".to_string())
                .to_string(),
        ));
    }
    Ok(())
}