use ff::PrimeField;
use num_bigint::BigUint;
use num_traits::One;
use p3_field::{AbstractField, Field, Packable, TwoAdicField};
use std::fmt::{Display, Formatter};

use crate::{Goldilocks, GoldilocksExt2, MODULUS};

impl AbstractField for Goldilocks {
    type F = Self;

    fn zero() -> Self {
        <Self as ff::Field>::ZERO
    }

    fn from_bool(b: bool) -> Self {
        Self(u64::from(b))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self(u64::from(n))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self(u64::from(n))
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self::from(n)
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self(u64::from(n))
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::from(n as u64)
    }

    fn from_f(f: Self::F) -> Self {
        f
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self(u64::from(n))
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

impl AbstractField for GoldilocksExt2 {
    type F = Self;

    fn zero() -> Self {
        <Self as ff::Field>::ZERO
    }

    fn from_bool(b: bool) -> Self {
        Self([Goldilocks::from_bool(b), Goldilocks::zero()])
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self([Goldilocks::from_canonical_u16(n), Goldilocks::zero()])
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self([Goldilocks::from_canonical_u32(n), Goldilocks::zero()])
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self([Goldilocks::from_canonical_u64(n), Goldilocks::zero()])
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self([Goldilocks::from_canonical_u8(n), Goldilocks::zero()])
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self([Goldilocks::from_canonical_usize(n), Goldilocks::zero()])
    }

    fn from_f(f: Self::F) -> Self {
        Self(f.0)
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self([Goldilocks::from_wrapped_u32(n), Goldilocks::zero()])
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self([Goldilocks::from_wrapped_u64(n), Goldilocks::zero()])
    }

    fn generator() -> Self {
        Self([
            Goldilocks(18081566051660590251),
            Goldilocks(16121475356294670766),
        ])
    }

    fn neg_one() -> Self {
        -Self::one()
    }

    fn one() -> Self {
        <Self as ff::Field>::ONE
    }

    fn square(&self) -> Self {
        <Self as ff::Field>::square(self)
    }

    fn two() -> Self {
        Self([Goldilocks::two(), Goldilocks::zero()])
    }
}

impl Packable for Goldilocks {}

impl Field for Goldilocks {
    type Packing = Self;

    fn order() -> BigUint {
        MODULUS.into()
    }

    fn try_inverse(&self) -> Option<Self> {
        <Self as ff::Field>::invert(self).into()
    }
}

impl Packable for GoldilocksExt2 {}

impl Display for GoldilocksExt2 {
    fn fmt(&self, w: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(w, "[{}, {}]", self.0[0], self.0[1])
    }
}

impl Field for GoldilocksExt2 {
    type Packing = Self;

    fn order() -> BigUint {
        let modulus_big_int: BigUint = MODULUS.into();
        modulus_big_int.pow(2) - BigUint::one()
    }

    fn try_inverse(&self) -> Option<Self> {
        <Self as ff::Field>::invert(self).into()
    }
}

const GOLDILOCKS_TWO_ADIC_GENERATORS: [Goldilocks; Goldilocks::TWO_ADICITY] = [
    Goldilocks(0x185629dcda58878c),
    Goldilocks(0x400a7f755588e659),
    Goldilocks(0x7e9bd009b86a0845),
    Goldilocks(0xdfa8c93ba46d2666),
    Goldilocks(0x59049500004a4485),
    Goldilocks(0x10d78dd8915a171d),
    Goldilocks(0xed41d05b78d6e286),
    Goldilocks(0x4bbaf5976ecfefd8),
    Goldilocks(0x86cdcc31c307e171),
    Goldilocks(0xea9d5a1336fbc98b),
    Goldilocks(0x4b2a18ade67246b5),
    Goldilocks(0xf502aef532322654),
    Goldilocks(0x30ba2ecd5e93e76d),
    Goldilocks(0xfbd41c6b8caa3302),
    Goldilocks(0x81281a7b05f9beac),
    Goldilocks(0xabd0a6e8aa3d8a0e),
    Goldilocks(0x54df9630bf79450e),
    Goldilocks(0xf6b2cffe2306baac),
    Goldilocks(0xe0ee099310bba1e2),
    Goldilocks(0x1544ef2335d17997),
    Goldilocks(0xf2c35199959dfcb6),
    Goldilocks(0x653b4801da1c8cf),
    Goldilocks(0x9d8f2ad78bfed972),
    Goldilocks(0x1905d02a5c411f4e),
    Goldilocks(0xbf79143ce60ca966),
    Goldilocks(0xf80007ff08000001),
    Goldilocks(0x8000000000),
    Goldilocks(0x3fffffffc000),
    Goldilocks(0xefffffff00000001),
    Goldilocks(0xfffffffeff000001),
    Goldilocks(0x1000000000000),
    Goldilocks(0xffffffff00000000),
];

impl TwoAdicField for Goldilocks {
    const TWO_ADICITY: usize = <Self as PrimeField>::S as usize;

    #[inline]
    fn two_adic_generator(bits: usize) -> Self {
        GOLDILOCKS_TWO_ADIC_GENERATORS[Self::TWO_ADICITY - bits]
    }
}

const GOLDILOCKS_EXT2_TWO_ADIC_GENERATOR: [GoldilocksExt2; GoldilocksExt2::TWO_ADICITY] = [
    GoldilocksExt2([Goldilocks(0x0), Goldilocks(0xd95051a31cf4a6ef)]),
    GoldilocksExt2([Goldilocks(0x185629dcda58878c), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x400a7f755588e659), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x7e9bd009b86a0845), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xdfa8c93ba46d2666), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x59049500004a4485), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x10d78dd8915a171d), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xed41d05b78d6e286), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x4bbaf5976ecfefd8), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x86cdcc31c307e171), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xea9d5a1336fbc98b), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x4b2a18ade67246b5), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xf502aef532322654), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x30ba2ecd5e93e76d), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xfbd41c6b8caa3302), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x81281a7b05f9beac), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xabd0a6e8aa3d8a0e), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x54df9630bf79450e), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xf6b2cffe2306baac), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xe0ee099310bba1e2), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x1544ef2335d17997), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xf2c35199959dfcb6), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x653b4801da1c8cf), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x9d8f2ad78bfed972), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x1905d02a5c411f4e), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xbf79143ce60ca966), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xf80007ff08000001), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x8000000000), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x3fffffffc000), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xefffffff00000001), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xfffffffeff000001), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0x1000000000000), Goldilocks(0x0)]),
    GoldilocksExt2([Goldilocks(0xffffffff00000000), Goldilocks(0x0)]),
];

impl TwoAdicField for GoldilocksExt2 {
    const TWO_ADICITY: usize = <Self as PrimeField>::S as usize + 1;

    #[inline]
    fn two_adic_generator(bits: usize) -> Self {
        GOLDILOCKS_EXT2_TWO_ADIC_GENERATOR[Self::TWO_ADICITY - bits]
    }
}
