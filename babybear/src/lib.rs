//! This crate implements Babybear field with modulus 2^64 - 2^32 + 1
//! Credit: the majority of the code is borrowed or inspired from Plonky2 with modifications.

pub use fp::Babybear;
pub use fp3::BabybearExt3;

pub use ff::{Field, PrimeField};
pub use fp::MODULUS;
pub use halo2curves::serde::SerdeObject;

pub mod p3_compatability;

#[macro_use]
mod derive;
mod field;
mod fp;
mod fp3;
mod util;

#[cfg(test)]
mod tests;
