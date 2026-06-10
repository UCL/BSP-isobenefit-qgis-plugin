//! Isobenefit Urbanism simulation core.
//!
//! Pure compute, arrays-in / arrays-out. The algorithm modules
//! ([`neighbours`], [`access`], [`density`], [`sim`]) are plain Rust and are unit
//! tested with `cargo test` (no Python needed). The PyO3 bindings live in
//! `py_bindings` and are compiled only with `--features python` (used by maturin).

pub mod access;
pub mod density;
pub mod neighbours;
pub mod sim;

#[cfg(feature = "python")]
mod py_bindings;
