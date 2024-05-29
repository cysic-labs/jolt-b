use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use goldilocks::Goldilocks;
use poseidon::Poseidon;
use std::mem::{size_of, transmute};

pub const OCTOPOS_HASH_OUTPUT_WIDTH: usize = 4;

#[derive(Debug, Default, PartialEq, Eq, Clone, Copy, CanonicalSerialize, CanonicalDeserialize)]
pub struct Digest(pub [Goldilocks; OCTOPOS_HASH_OUTPUT_WIDTH]);

impl Digest {
    pub fn as_u8s(&self) -> &[u8; OCTOPOS_HASH_OUTPUT_WIDTH * size_of::<Goldilocks>()] {
        unsafe {
            transmute::<
                &[Goldilocks; OCTOPOS_HASH_OUTPUT_WIDTH],
                &[u8; OCTOPOS_HASH_OUTPUT_WIDTH * size_of::<Goldilocks>()],
            >(&self.0)
        }
    }
}

// NOTE: this const for poseidon hash state width is invariant of hash leaves' fields,
// namely, this is only with respect to the Goldilocks base field size, rather than
// arbitrary extension fields or curve element.
pub const OCTOPOS_HASH_STATE_GOLDILOCKS: usize = 12;
pub const OCTOPOS_HASH_RATE: usize = 11;

pub type OctoposHasher = Poseidon<Goldilocks, OCTOPOS_HASH_STATE_GOLDILOCKS, OCTOPOS_HASH_RATE>;

pub const OCTOPOS_LEAF_GOLDILOCKS: usize = 8;

pub const OCTOPOS_LEAF_BYTES: usize = OCTOPOS_LEAF_GOLDILOCKS * size_of::<Goldilocks>();

pub const OCTOPOS_HASH_FULL_ROUNDS: usize = 8;

pub const OCTOPOS_HASH_PARTIAL_ROUNDS: usize = 22;

pub fn new_octopos_hasher() -> OctoposHasher {
    OctoposHasher::new(OCTOPOS_HASH_FULL_ROUNDS, OCTOPOS_HASH_PARTIAL_ROUNDS)
}
