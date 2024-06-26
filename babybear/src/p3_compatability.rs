use ff::PrimeField;
use num_bigint::BigUint;
use p3_field::{AbstractField, Field, Packable, TwoAdicField};

use crate::{Babybear, MODULUS};

impl AbstractField for Babybear {
    type F = Self;

    fn zero() -> Self {
        <Self as ff::Field>::ZERO
    }

    fn from_bool(b: bool) -> Self {
        Self(u32::from(b))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self(u32::from(n))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self(u32::from(n))
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self::from(n)
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self(u32::from(n))
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::from(n as u64)
    }

    fn from_f(f: Self::F) -> Self {
        f
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self(u32::from(n))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self::from(n)
    }

    fn generator() -> Self {
        <Self as PrimeField>::MULTIPLICATIVE_GENERATOR
    }

    fn neg_one() -> Self {
        Self(MODULUS - 1)
    }

    fn one() -> Self {
        <Self as ff::Field>::ONE
    }

    fn square(&self) -> Self {
        <Self as ff::Field>::square(self)
    }

    fn two() -> Self {
        Self(2)
    }
}

impl Packable for Babybear {}

impl Field for Babybear {
    type Packing = Self;

    fn order() -> BigUint {
        MODULUS.into()
    }

    fn try_inverse(&self) -> Option<Self> {
        <Self as ff::Field>::invert(self).into()
    }
}

const BABYBEAR_TWO_ADIC_GENERATORS: [Babybear; Babybear::TWO_ADICITY] = [
    Babybear(0x1a427a41),
    Babybear(0x3a26eef8),
    Babybear(0x4483d85a),
    Babybear(0x3bd57996),
    Babybear(0x4b859b3d),
    Babybear(0x21fd55bc),
    Babybear(0x18adc27d),
    Babybear(0xba067a3),
    Babybear(0x3e9430e8),
    Babybear(0x5cf5713f),
    Babybear(0x4cabd6a6),
    Babybear(0x54c131f4),
    Babybear(0x77cad399),
    Babybear(0x62c3d2b1),
    Babybear(0x11c33e2a),
    Babybear(0x4c734715),
    Babybear(0x4fe61226),
    Babybear(0x145e952d),
    Babybear(0x688442f9),
    Babybear(0x67456167),
    Babybear(0x17b56c64),
    Babybear(0x669d6090),
    Babybear(0x2d4cc4da),
    Babybear(0xbb4c4e4),
    Babybear(0x5ee99486),
    Babybear(0x67055c21),
    Babybear(0x78000000),
];

impl TwoAdicField for Babybear {
    const TWO_ADICITY: usize = <Self as PrimeField>::S as usize;

    #[inline]
    fn two_adic_generator(bits: usize) -> Self {
        BABYBEAR_TWO_ADIC_GENERATORS[Self::TWO_ADICITY - bits]
    }
}
