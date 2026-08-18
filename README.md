# VectOrEngine (`vect-or-engine`)

> **High-Performance Rust-Powered Vector Indexing (HNSW) & Semantic Validation Engine**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75+-DEA584?style=flat-square&logo=rust&logoColor=black)](https://www.rust-lang.org/)
[![N-API](https://img.shields.io/badge/Node.js-N--API_Native-green?style=flat-square&logo=node.js&logoColor=white)](https://napi.rs/)

**VectOrEngine** is a standalone, ultra-low latency native core engine written in Rust. It provides approximate nearest neighbor search (HNSW), rule-driven semantic linter validation, and zero-copy memory-mapped JSON/SQLite knowledge management for Node.js, Electron, and Rust native applications.

---

## ✨ Features

- **⚡ Hierarchical Navigable Small World (HNSW)**: $O(\log N)$ approximate nearest neighbor cosine similarity search with SIMD auto-vectorization.
- **💾 Memory-Efficient Representation**:
  - `f16` half-precision float vector quantization (50% RAM reduction).
  - `memmap2` zero-copy memory-mapped I/O for instant multi-gigabyte knowledge loading.
- **🚀 Real-Time Multi-Pattern Linter**: Parallel rule evaluation and keyword conflict analysis using Aho-Corasick automaton and regex.
- **🔗 Native Node.js Bindings (N-API)**: Direct native bindings via `napi-rs` with zero IPC serialization overhead.
- **🗃️ Pluggable Local Cache**: Bundled SQLite cache backend for fast warm-restarts.

---

## 🛠️ Project Structure

```
vect-or-engine/
├── Cargo.toml          # Rust crate configuration & optimization profiles
├── lib.rs              # N-API bindings & Node.js native interface
├── index.js            # Node.js export loader
├── index.d.ts          # TypeScript type definitions
└── src/
    ├── hnsw_index.rs       # HNSW graph indexing & search implementation
    ├── knowledge_store.rs  # Vector quantization, mmap, and SQLite store
    ├── validator.rs        # Real-time static linter & rule evaluator
    ├── translator.rs       # Term mapping & colloquial pattern matching
    ├── profile.rs          # Schema-driven profile deserializer
    ├── engine_error.rs     # Typed engine error definitions
    ├── lib.rs              # Core Rust library entry point
    ├── main.rs             # Stdin/stdout JSON-RPC server binary
    └── index.ts            # TypeScript IPC client wrapper
```

---

## 🚀 Building from Source

### Prerequisites

- [Rust](https://www.rust-lang.org/) (2021 edition, `cargo` v1.75+)
- [Node.js](https://nodejs.org/) (v18+)

### Build Native Addon

```bash
# Install dependencies
npm install

# Compile release binary (.node)
npm run build
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
