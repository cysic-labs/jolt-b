use goldilocks::{Goldilocks, GoldilocksExt2};

use super::JoltField;

impl JoltField for Goldilocks {
    const NUM_BYTES: usize = 8;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        <Self as goldilocks::Field>::random(rng)
    }

    fn is_zero(&self) -> bool {
        <Self as goldilocks::Field>::is_zero_vartime(self)
    }

    fn is_one(&self) -> bool {
        self == &<Self as goldilocks::Field>::ONE
    }

    fn zero() -> Self {
        <Self as goldilocks::Field>::ZERO
    }

    fn one() -> Self {
        <Self as goldilocks::Field>::ONE
    }

    fn from_u64(n: u64) -> Option<Self> {
        // According to its usage, it should reject instead of taking the modula.
        assert!(n < goldilocks::MODULUS);
        Some(<Self as From<u64>>::from(n))
    }

    fn square(&self) -> Self {
        <Self as goldilocks::Field>::square(self)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::NUM_BYTES);
        <Self as goldilocks::SerdeObject>::from_raw_bytes(bytes).unwrap()
    }
}

impl JoltField for GoldilocksExt2 {
    const NUM_BYTES: usize = 16;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        <Self as goldilocks::Field>::random(rng)
    }

    fn is_zero(&self) -> bool {
        <Self as goldilocks::Field>::is_zero_vartime(self)
    }

    fn is_one(&self) -> bool {
        self == &<Self as goldilocks::Field>::ONE
    }

    fn zero() -> Self {
        <Self as goldilocks::Field>::ZERO
    }

    fn one() -> Self {
        <Self as goldilocks::Field>::ONE
    }

    fn from_u64(n: u64) -> Option<Self> {
        // According to its usage, it should reject instead of taking the modula.
        assert!(n < goldilocks::MODULUS);
        Some(<Self as From<u64>>::from(n))
    }

    fn square(&self) -> Self {
        <Self as goldilocks::Field>::square(self)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::NUM_BYTES);
        <Self as goldilocks::SerdeObject>::from_raw_bytes(bytes).unwrap()
    }
}
