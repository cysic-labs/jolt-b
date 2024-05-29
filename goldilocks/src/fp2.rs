//! This module implements Goldilocks quadratic extension field mod x^2 - 7

use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::hash::Hash;
use std::io::{Read, Write};
use std::ops::Div;

use ff::{Field, FromUniformBytes, PrimeField};
use halo2curves::serde::SerdeObject;
use rand_core::RngCore;
use serde::{Deserialize, Serialize};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use crate::field::SmallField;
use crate::Goldilocks;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};

/// Degree 3 Goldilocks extension field mod x^2 - 7
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct GoldilocksExt2(pub [Goldilocks; 2]);

/// For a = (a1, a2) and b = (b1, b2)
/// The multiplication is define as
/// c := a * b = a(x) * b(x) % (x^2 - 7)
///    = x*a2*b1 + x*a1*b2
///    + a1*b1 + 7*a2*b2

/// This requires 9 multiplications and 6 1 additions
fn mul_internal(a: &GoldilocksExt2, b: &GoldilocksExt2) -> GoldilocksExt2 {
    // todo: optimizations?
    let a1b1 = a.0[0] * b.0[0];
    let a1b2 = a.0[0] * b.0[1];
    let a2b1 = a.0[1] * b.0[0];
    let a2b2 = a.0[1] * b.0[1];

    let c1 = a1b1 + Goldilocks(7) * a2b2;
    let c2 = a2b1 + a1b2;
    GoldilocksExt2([c1, c2])
}

impl_extension_field!(Goldilocks, GoldilocksExt2, 2);

impl SerdeObject for GoldilocksExt2 {
    /// The purpose of unchecked functions is to read the internal memory representation
    /// of a type from bytes as quickly as possible. No sanitization checks are performed
    /// to ensure the bytes represent a valid object. As such this function should only be
    /// used internally as an extension of machine memory. It should not be used to deserialize
    /// externally provided data.
    fn from_raw_bytes_unchecked(bytes: &[u8]) -> Self {
        Self([
            Goldilocks::from_raw_bytes_unchecked(bytes[..8].as_ref()),
            Goldilocks::from_raw_bytes_unchecked(bytes[8..].as_ref()),
        ])
    }

    fn from_raw_bytes(bytes: &[u8]) -> Option<Self> {
        let a1 = match Goldilocks::from_raw_bytes(bytes[..8].as_ref()) {
            Some(p) => p,
            None => return None,
        };
        let a2 = match Goldilocks::from_raw_bytes(bytes[8..].as_ref()) {
            Some(p) => p,
            None => return None,
        };

        Some(Self([a1, a2]))
    }

    fn to_raw_bytes(&self) -> Vec<u8> {
        [self.0[0].to_raw_bytes(), self.0[1].to_raw_bytes()].concat()
    }

    /// The purpose of unchecked functions is to read the internal memory representation
    /// of a type from disk as quickly as possible. No sanitization checks are performed
    /// to ensure the bytes represent a valid object. This function should only be used
    /// internally when some machine state cannot be kept in memory (e.g., between runs)
    /// and needs to be reloaded as quickly as possible.
    fn read_raw_unchecked<R: Read>(reader: &mut R) -> Self {
        let a1 = Goldilocks::read_raw_unchecked(reader);
        let a2 = Goldilocks::read_raw_unchecked(reader);
        Self([a1, a2])
    }
    fn read_raw<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let a1 = Goldilocks::read_raw(reader)?;
        let a2 = Goldilocks::read_raw(reader)?;
        Ok(Self([a1, a2]))
    }

    fn write_raw<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0[0].write_raw(writer)?;
        self.0[1].write_raw(writer)
    }
}

impl Field for GoldilocksExt2 {
    /// The zero element of the field, the additive identity.
    const ZERO: Self = Self([Goldilocks::ZERO; 2]);

    /// The one element of the field, the multiplicative identity.
    const ONE: Self = Self([Goldilocks::ONE, Goldilocks::ZERO]);

    /// Returns an element chosen uniformly at random using a user-provided RNG.
    /// Note: this sampler is not constant time!
    fn random(mut rng: impl RngCore) -> Self {
        let a1 = Goldilocks::random(&mut rng);
        let a2 = Goldilocks::random(&mut rng);

        Self([a1, a2])
    }

    /// Squares this element.
    #[must_use]
    fn square(&self) -> Self {
        *self * *self
    }

    /// Cubes this element.
    #[must_use]
    fn cube(&self) -> Self {
        self.square() * self
    }

    /// Doubles this element.
    #[must_use]
    fn double(&self) -> Self {
        *self + *self
    }

    /// Computes the multiplicative inverse of this element,
    /// failing if the element is zero.
    fn invert(&self) -> CtOption<Self> {
        if self.is_zero_vartime() {
            return CtOption::new(Self::default(), (false as u8).into());
        }

        let a_pow_r_minus_1 = self.frobenius();
        let a_pow_r = a_pow_r_minus_1 * *self;
        debug_assert!(a_pow_r.0[1] == Goldilocks::ZERO);
        let a_pow_r_inv = a_pow_r.0[0].invert().expect("inverse does not exist");

        let res = [
            a_pow_r_minus_1.0[0] * a_pow_r_inv,
            a_pow_r_minus_1.0[1] * a_pow_r_inv,
        ];

        CtOption::new(Self(res), Choice::from(1))
    }

    /// Returns the square root of the field element, if it is
    /// quadratic residue.
    fn sqrt(&self) -> CtOption<Self> {
        unimplemented!()
    }

    /// Computes:
    ///
    /// - $(\textsf{true}, \sqrt{\textsf{num}/\textsf{div}})$, if $\textsf{num}$ and
    ///   $\textsf{div}$ are nonzero and $\textsf{num}/\textsf{div}$ is a square in the
    ///   field;
    /// - $(\textsf{true}, 0)$, if $\textsf{num}$ is zero;
    /// - $(\textsf{false}, 0)$, if $\textsf{num}$ is nonzero and $\textsf{div}$ is zero;
    /// - $(\textsf{false}, \sqrt{G_S \cdot \textsf{num}/\textsf{div}})$, if
    ///   $\textsf{num}$ and $\textsf{div}$ are nonzero and $\textsf{num}/\textsf{div}$ is
    ///   a nonsquare in the field;
    ///
    /// where $G_S$ is a non-square.
    ///
    /// # Warnings
    ///
    /// - The choice of root from `sqrt` is unspecified.
    /// - The value of $G_S$ is unspecified, and cannot be assumed to have any specific
    ///   value in a generic context.
    fn sqrt_ratio(_: &Self, _: &Self) -> (Choice, Self) {
        unimplemented!()
    }
}

impl ConditionallySelectable for GoldilocksExt2 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self([
            Goldilocks::conditional_select(&a.0[0], &b.0[0], choice),
            Goldilocks::conditional_select(&a.0[1], &b.0[1], choice),
        ])
    }
}

// impl ConstantTimeEq for GoldilocksExt2 {
//     fn ct_eq(&self, other: &Self) -> Choice {
//         self.to_canonical_u64_array()
//             .ct_eq(&other.to_canonical_u64_array())
//     }
// }
impl Neg for GoldilocksExt2 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1]])
    }
}

impl Add for GoldilocksExt2 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl Sub for GoldilocksExt2 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl AsMut<[u8]> for GoldilocksExt2 {
    fn as_mut(&mut self) -> &mut [u8] {
        let ptr = self as *mut Self as *mut u8;
        // SAFETY Self is repr(transparent) and u64 is 8 bytes wide,
        // with alignment greater than that of u8
        unsafe { std::slice::from_raw_parts_mut(ptr, 16) }
    }
}

impl AsRef<[u8]> for GoldilocksExt2 {
    fn as_ref(&self) -> &[u8] {
        let ptr = self as *const Self as *const u8;
        // SAFETY Self is repr(transparent) and u64 is 8 bytes wide,
        // with alignment greater than that of u8
        unsafe { std::slice::from_raw_parts(ptr, 16) }
    }
}

impl From<Goldilocks> for GoldilocksExt2 {
    fn from(a: Goldilocks) -> Self {
        Self([a, Goldilocks::ZERO])
    }
}

/// This represents an element of a prime field.
impl PrimeField for GoldilocksExt2 {
    /// The prime field can be converted back and forth into this binary
    /// representation.
    type Repr = Self;

    /// Modulus of the field written as a string for debugging purposes.
    ///
    /// The encoding of the modulus is implementation-specific. Generic users of the
    /// `PrimeField` trait should treat this string as opaque.
    const MODULUS: &'static str = "0xffffffff00000001";

    /// How many bits are needed to represent an element of this field.
    const NUM_BITS: u32 = 128;

    /// How many bits of information can be reliably stored in the field element.
    ///
    /// This is usually `Self::NUM_BITS - 1`.
    const CAPACITY: u32 = 126;

    /// An integer `s` satisfying the equation `2^s * t = modulus - 1` with `t` odd.
    ///
    /// This is the number of leading zero bits in the little-endian bit representation of
    /// `modulus - 1`.
    const S: u32 = 32;

    /// Inverse of $2$ in the field.
    const TWO_INV: Self = GoldilocksExt2([Goldilocks::TWO_INV, Goldilocks::ZERO]);

    /// A fixed multiplicative generator of `modulus - 1` order. This element must also be
    /// a quadratic nonresidue.
    ///
    /// It can be calculated using [SageMath] as `GF(modulus).primitive_element()`.
    ///
    /// Implementations of this trait MUST ensure that this is the generator used to
    /// derive `Self::ROOT_OF_UNITY`.
    ///
    /// [SageMath]: https://www.sagemath.org/
    const MULTIPLICATIVE_GENERATOR: Self =
        GoldilocksExt2([Goldilocks::MULTIPLICATIVE_GENERATOR, Goldilocks::ZERO]);

    /// The `2^s` root of unity.
    ///
    /// It can be calculated by exponentiating `Self::MULTIPLICATIVE_GENERATOR` by `t`,
    /// where `t = (modulus - 1) >> Self::S`.
    const ROOT_OF_UNITY: Self = GoldilocksExt2([Goldilocks::ROOT_OF_UNITY, Goldilocks::ZERO]);

    /// Inverse of [`Self::ROOT_OF_UNITY`].
    const ROOT_OF_UNITY_INV: Self =
        GoldilocksExt2([Goldilocks::ROOT_OF_UNITY_INV, Goldilocks::ZERO]);

    /// Generator of the `t-order` multiplicative subgroup.
    ///
    /// It can be calculated by exponentiating [`Self::MULTIPLICATIVE_GENERATOR`] by `2^s`,
    /// where `s` is [`Self::S`].
    const DELTA: Self = GoldilocksExt2([Goldilocks::DELTA, Goldilocks::ZERO]);

    /// Attempts to convert a byte representation of a field element into an element of
    /// this prime field, failing if the input is not canonical (is not smaller than the
    /// field's modulus).
    ///
    /// The byte representation is interpreted with the same endianness as elements
    /// returned by [`PrimeField::to_repr`].
    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        CtOption::new(repr, Choice::from(1))
    }

    /// Attempts to convert a byte representation of a field element into an element of
    /// this prime field, failing if the input is not canonical (is not smaller than the
    /// field's modulus).
    ///
    /// The byte representation is interpreted with the same endianness as elements
    /// returned by [`PrimeField::to_repr`].
    ///
    /// # Security
    ///
    /// This method provides **no** constant-time guarantees. Implementors of the
    /// `PrimeField` trait **may** optimise this method using non-constant-time logic.
    fn from_repr_vartime(repr: Self::Repr) -> Option<Self> {
        Self::from_repr(repr).into()
    }

    /// Converts an element of the prime field into the standard byte representation for
    /// this field.
    ///
    /// The endianness of the byte representation is implementation-specific. Generic
    /// encodings of field elements should be treated as opaque.
    fn to_repr(&self) -> Self::Repr {
        *self
    }

    /// Returns true iff this element is odd.
    fn is_odd(&self) -> Choice {
        unimplemented!()
    }
}

impl FromUniformBytes<64> for GoldilocksExt2 {
    fn from_uniform_bytes(bytes: &[u8; 64]) -> Self {
        Self([
            <Goldilocks as FromUniformBytes<32>>::from_uniform_bytes(
                bytes[..32].try_into().unwrap(),
            ),
            <Goldilocks as FromUniformBytes<32>>::from_uniform_bytes(
                bytes[32..].try_into().unwrap(),
            ),
        ])
    }
}

impl GoldilocksExt2 {
    /// FrobeniusField automorphisms: x -> x^p, where p is the order of BaseField.
    fn frobenius(&self) -> Self {
        self.repeated_frobenius(1)
    }

    /// Repeated Frobenius automorphisms: x -> x^(p^count).
    ///
    /// Follows precomputation suggestion in Section 11.3.3 of the
    /// Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn repeated_frobenius(&self, count: usize) -> Self {
        if count == 0 {
            return *self;
        } else if count >= 2 {
            // x |-> x^(p^D) is the identity, so x^(p^count) ==
            // x^(p^(count % D))
            return self.repeated_frobenius(count % 2);
        }
        let arr = self.0;

        // z0 = DTH_ROOT^count = W^(k * count) where k = floor((p^D-1)/D)
        let mut z0 = Goldilocks(18446744069414584320);
        for _ in 1..count {
            z0 *= Goldilocks(18446744069414584320);
        }
        let z0square = z0 * z0;

        let mut res = [Goldilocks::ZERO; 2];

        res[0] = arr[0] * z0;
        res[1] = arr[1] * z0square;

        Self(res)
    }
}

impl TryInto<Goldilocks> for GoldilocksExt2 {
    /// The type returned in the event of a conversion error.
    type Error = &'static str;

    /// Performs the conversion.
    fn try_into(self) -> Result<Goldilocks, Self::Error> {
        if self.0[1].is_zero_vartime() {
            Ok(self.0[0])
        } else {
            Err("extension field is not zero")
        }
    }
}

impl CanonicalDeserialize for GoldilocksExt2 {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let a = Goldilocks::deserialize_with_mode(&mut reader, compress, validate)?;
        let b = Goldilocks::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self([a, b]))
    }
}

impl Valid for GoldilocksExt2 {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        Ok(())
    }
}
impl CanonicalSerialize for GoldilocksExt2 {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.0[0].serialize_with_mode(&mut writer, compress)?;
        self.0[1].serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        2 * core::mem::size_of::<u64>()
    }
}

impl Div for GoldilocksExt2 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        self * rhs.invert().unwrap()
    }
}
