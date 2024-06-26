//! This module defines our customized field trait.

use ff::Field;
use goldilocks::SmallField;
use halo2curves::serde::SerdeObject;
use rand_core::RngCore;

use crate::{Babybear, BabybearExt3};

impl SmallField for Babybear {
    type BaseField = Self;

    const DEGREE: usize = 1;
    const NAME: &'static str = "Babybear";

    fn bytes_to_field_elements(bytes: &[u8]) -> Vec<Self> {
        bytes
            .chunks(8)
            .map(Self::from_raw_bytes_unchecked)
            .collect::<Vec<_>>()
    }

    fn to_canonical_u64_vec(&self) -> Vec<u64> {
        vec![self.to_canonical_u32() as u64]
    }

    fn to_limbs(&self) -> Vec<Babybear> {
        vec![*self]
    }

    fn from_limbs(limbs: &[Babybear]) -> Self {
        limbs[0]
    }

    fn sample_base(mut rng: impl RngCore) -> Self {
        Self::random(&mut rng)
    }
    fn from_base(b: &Self::BaseField) -> Self {
        *b
    }

    /// Mul-assign self by a base field element
    fn mul_assign_base(&mut self, rhs: &Self::BaseField) {
        *self *= rhs;
    }
}

impl SmallField for BabybearExt3 {
    type BaseField = Babybear;

    const DEGREE: usize = 3;
    const NAME: &'static str = "BabybearExt3";

    fn bytes_to_field_elements(bytes: &[u8]) -> Vec<Self> {
        bytes
            .chunks(24)
            .map(Self::from_raw_bytes_unchecked)
            .collect::<Vec<_>>()
    }

    fn to_canonical_u64_vec(&self) -> Vec<u64> {
        self.0
            .iter()
            .map(|a| a.to_canonical_u32() as u64)
            .collect::<Vec<_>>()
    }

    fn to_limbs(&self) -> Vec<Babybear> {
        self.0.to_vec()
    }

    fn from_limbs(limbs: &[Babybear]) -> Self {
        Self([limbs[0], limbs[1], limbs[2]])
    }

    fn sample_base(mut rng: impl RngCore) -> Self {
        Self::BaseField::random(&mut rng).into()
    }

    fn from_base(b: &Self::BaseField) -> Self {
        Self([*b, Babybear::ZERO, Babybear::ZERO])
    }

    /// Mul-assign self by a base field element
    fn mul_assign_base(&mut self, rhs: &Self::BaseField) {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
        self.0[2] *= rhs;
    }
}
