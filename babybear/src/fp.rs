use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::ops::Div;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid, Validate};
use ff::{Field, FromUniformBytes, PrimeField};
use halo2curves::serde::SerdeObject;
use itertools::Itertools;
use rand_core::RngCore;
use serde::{Deserialize, Serialize};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use crate::util::sqrt_tonelli_shanks;

/// Babybear field with modulus 2^31 - 2^27 + 1.
/// A Babybear field may store a non-canonical form of the element
/// where the value can be between 0 and 2^32.
/// For unique representation of its form, use `to_canonical_u32`
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Babybear(pub u32);

impl SerdeObject for Babybear {
    /// The purpose of unchecked functions is to read the internal memory representation
    /// of a type from bytes as quickly as possible. No sanitization checks are performed
    /// to ensure the bytes represent a valid object. As such this function should only be
    /// used internally as an extension of machine memory. It should not be used to deserialize
    /// externally provided data.
    fn from_raw_bytes_unchecked(bytes: &[u8]) -> Self {
        let mut tmp = u32::from_le_bytes(
            bytes
                .iter()
                .pad_using(8, |_| &0u8)
                .take(8)
                .cloned()
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap(),
        );
        if tmp >= MODULUS {
            tmp -= MODULUS
        }

        Self(tmp)
    }

    fn from_raw_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() > 4 {
            return None;
        }
        let tmp = u32::from_le_bytes(
            bytes
                .iter()
                .pad_using(4, |_| &0u8)
                .take(4)
                .cloned()
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap(),
        );
        if tmp >= MODULUS {
            None
        } else {
            Some(Self(tmp))
        }
    }

    fn to_raw_bytes(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }

    /// The purpose of unchecked functions is to read the internal memory representation
    /// of a type from disk as quickly as possible. No sanitization checks are performed
    /// to ensure the bytes represent a valid object. This function should only be used
    /// internally when some machine state cannot be kept in memory (e.g., between runs)
    /// and needs to be reloaded as quickly as possible.
    fn read_raw_unchecked<R: Read>(reader: &mut R) -> Self {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).unwrap();
        Self(u32::from_le_bytes(buf))
    }
    fn read_raw<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        let tmp = u32::from_le_bytes(buf);
        if tmp >= MODULUS {
            // todo: wrap the error
            panic!("Not a field element")
        } else {
            Ok(Self(tmp))
        }
    }

    fn write_raw<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(self.as_ref())
    }
}

impl PartialEq for Babybear {
    fn eq(&self, other: &Babybear) -> bool {
        self.to_canonical_u32() == other.to_canonical_u32()
    }
}

impl Eq for Babybear {}

impl Hash for Babybear {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.to_canonical_u32().hash(hasher);
    }
}

impl Display for Babybear {
    fn fmt(&self, w: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(w, "{}", self.0)
    }
}

/// 2^31 - 2^27 + 1
pub const MODULUS: u32 = 0x78000001;

impl FromUniformBytes<64> for Babybear {
    fn from_uniform_bytes(bytes: &[u8; 64]) -> Self {
        <Self as FromUniformBytes<32>>::from_uniform_bytes(bytes[0..32].try_into().unwrap())
    }
}

impl FromUniformBytes<32> for Babybear {
    fn from_uniform_bytes(bytes: &[u8; 32]) -> Self {
        // FIXME: this is biased.
        Babybear(u32::from_le_bytes(bytes[..8].try_into().unwrap()))
    }
}

impl FromUniformBytes<16> for Babybear {
    fn from_uniform_bytes(bytes: &[u8; 16]) -> Self {
        // FIXME: this is also biased.
        Babybear(u32::from_le_bytes(bytes[..8].try_into().unwrap()))
    }
}

impl FromUniformBytes<8> for Babybear {
    fn from_uniform_bytes(bytes: &[u8; 8]) -> Self {
        // FIXME: this is also biased.
        Babybear(u32::from_le_bytes(bytes[..8].try_into().unwrap()))
    }
}

impl Field for Babybear {
    /// The zero element of the field, the additive identity.
    const ZERO: Self = Self(0);

    /// The one element of the field, the multiplicative identity.
    const ONE: Self = Self(1);

    /// Returns an element chosen uniformly at random using a user-provided RNG.
    /// Note: this sampler is not constant time!
    fn random(mut rng: impl RngCore) -> Self {
        let mut res = rng.next_u32();
        while res >= MODULUS {
            res = rng.next_u32();
        }
        Self(res)
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
    ///
    /// credit: https://github.com/Plonky3/Plonky3/blob/7bb6db50594e159010f11c97d110aa3ee121069b/baby-bear/src/baby_bear.rs#L220
    fn invert(&self) -> CtOption<Self> {
        if self.is_zero().into() {
            return CtOption::new(Self::ZERO, Choice::from(0));
        }
        let a = self.pow_vartime([(MODULUS - 2) as u64]);

        // From Fermat's little theorem, in a prime field `F_p`, the inverse of `a` is `a^(p-2)`.
        // Here p-2 = 2013265919 = 1110111111111111111111111111111_2.
        // Uses 30 Squares + 7 Multiplications => 37 Operations total.

        let p1 = *self;
        let p100000000 = p1.exp_power_of_2(8);
        println!("1 {:?}, {:?}, ", self, p100000000);
        let p100000001 = p100000000 * p1;
        println!("2 {:?}, ", p100000001);
        let p10000000000000000 = p100000000.exp_power_of_2(8);
        println!("3 {:?}, ", p100000001);
        let p10000000100000001 = p10000000000000000 * p100000001;
        let p10000000100000001000 = p10000000100000001.exp_power_of_2(3);
        let p1000000010000000100000000 = p10000000100000001000.exp_power_of_2(5);
        let p1000000010000000100000001 = p1000000010000000100000000 * p1;
        let p1000010010000100100001001 = p1000000010000000100000001 * p10000000100000001000;
        let p10000000100000001000000010 = p1000000010000000100000001.square();
        let p11000010110000101100001011 = p10000000100000001000000010 * p1000010010000100100001001;
        let p100000001000000010000000100 = p10000000100000001000000010.square();
        let p111000011110000111100001111 =
            p100000001000000010000000100 * p11000010110000101100001011;
        let p1110000111100001111000011110000 = p111000011110000111100001111.exp_power_of_2(4);
        let p1110111111111111111111111111111 =
            p1110000111100001111000011110000 * p111000011110000111100001111;

        println!(
            "{:?}, {:?}, {:?}",
            self, a, p1110111111111111111111111111111
        );
        // CtOption::new(p1110111111111111111111111111111, Choice::from(1))
        assert_eq!(a, p1110111111111111111111111111111);

        CtOption::new(a, Choice::from(1))
    }

    /// Returns the square root of the field element, if it is
    /// quadratic residue.
    fn sqrt(&self) -> CtOption<Self> {
        // TODO: better algorithm taking advantage of (t-1)/2 has a nice structure
        sqrt_tonelli_shanks(self, 7)
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

impl Babybear {
    fn exp_power_of_2(&self, power_log: usize) -> Self {
        let mut res = self.clone();
        for _ in 0..power_log {
            res = res.square();
        }
        res
    }
}

impl AsRef<u32> for Babybear {
    fn as_ref(&self) -> &u32 {
        &self.0
    }
}

impl AsMut<[u8]> for Babybear {
    fn as_mut(&mut self) -> &mut [u8] {
        let ptr = self as *mut Self as *mut u8;
        // SAFETY Self is repr(transparent) and u64 is 8 bytes wide,
        // with alignment greater than that of u8
        unsafe { std::slice::from_raw_parts_mut(ptr, 4) }
    }
}

impl AsRef<[u8]> for Babybear {
    fn as_ref(&self) -> &[u8] {
        let ptr = self as *const Self as *const u8;
        // SAFETY Self is repr(transparent) and u64 is 8 bytes wide,
        // with alignment greater than that of u8
        unsafe { std::slice::from_raw_parts(ptr, 4) }
    }
}

/// This represents an element of a prime field.
impl PrimeField for Babybear {
    /// The prime field can be converted back and forth into this binary
    /// representation.
    type Repr = Self;

    /// Modulus of the field written as a string for debugging purposes.
    ///
    /// The encoding of the modulus is implementation-specific. Generic users of the
    /// `PrimeField` trait should treat this string as opaque.
    const MODULUS: &'static str = "0x78000001";

    /// How many bits are needed to represent an element of this field.
    const NUM_BITS: u32 = 31;

    /// How many bits of information can be reliably stored in the field element.
    ///
    /// This is usually `Self::NUM_BITS - 1`.
    const CAPACITY: u32 = 30;

    /// An integer `s` satisfying the equation `2^s * t = modulus - 1` with `t` odd.
    ///
    /// This is the number of leading zero bits in the little-endian bit representation of
    /// `modulus - 1`.
    const S: u32 = 27;

    /// Inverse of $2$ in the field.
    const TWO_INV: Self = Self(0x3c000001);

    /// A fixed multiplicative generator of `modulus - 1` order. This element must also be
    /// a quadratic nonresidue.
    ///
    /// It can be calculated using [SageMath] as `GF(modulus).primitive_element()`.
    ///
    /// Implementations of this trait MUST ensure that this is the generator used to
    /// derive `Self::ROOT_OF_UNITY`.
    ///
    /// [SageMath]: https://www.sagemath.org/
    const MULTIPLICATIVE_GENERATOR: Self = Self(31);

    /// The `2^s` root of unity.
    ///
    /// It can be calculated by exponentiating `Self::MULTIPLICATIVE_GENERATOR` by `t`,
    /// where `t = (modulus - 1) >> Self::S`.
    const ROOT_OF_UNITY: Self = Self(0x1a427a41);

    /// Inverse of [`Self::ROOT_OF_UNITY`].
    const ROOT_OF_UNITY_INV: Self = Self(0x662731d4);

    /// Generator of the `t-order` multiplicative subgroup.
    ///
    /// It can be calculated by exponentiating [`Self::MULTIPLICATIVE_GENERATOR`] by `2^s`,
    /// where `s` is [`Self::S`].
    const DELTA: Self = Self(0x76f07a0c);

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
        Choice::from((self.0 & 1) as u8)
    }
}

impl From<u64> for Babybear {
    fn from(input: u64) -> Self {
        assert!(input < MODULUS as u64);
        Self(input as u32)
    }
}

impl From<Babybear> for u64 {
    fn from(input: Babybear) -> Self {
        input.0 as u64
    }
}

impl ConditionallySelectable for Babybear {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self(u32::conditional_select(&a.0, &b.0, choice))
    }
}

impl ConstantTimeEq for Babybear {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.to_canonical_u32().ct_eq(&other.to_canonical_u32())
    }
}

impl Neg for Babybear {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        if self.0 == 0 {
            self
        } else {
            Self(MODULUS - self.to_canonical_u32())
        }
    }
}

impl Add for Babybear {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut sum = self.0 + rhs.0;
        let (corr_sum, over) = sum.overflowing_sub(MODULUS);
        if !over {
            sum = corr_sum;
        }
        Self(sum)
    }
}

impl<'a> Add<&'a Babybear> for Babybear {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &'a Babybear) -> Self::Output {
        self + *rhs
    }
}

impl<'a> Add<&'a mut Babybear> for Babybear {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &'a mut Babybear) -> Self::Output {
        self + *rhs
    }
}

impl AddAssign for Babybear {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<'a> AddAssign<&'a Babybear> for Babybear {
    #[inline]
    fn add_assign(&mut self, rhs: &'a Babybear) {
        *self = *self + *rhs;
    }
}

impl<'a> AddAssign<&'a mut Babybear> for Babybear {
    #[inline]
    fn add_assign(&mut self, rhs: &'a mut Babybear) {
        *self = *self + *rhs;
    }
}

impl Sub for Babybear {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        let (mut diff, over) = self.0.overflowing_sub(rhs.0);
        let corr = if over { MODULUS } else { 0 };
        diff = diff.wrapping_add(corr);
        Self(diff)
    }
}

impl<'a> Sub<&'a Babybear> for Babybear {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &'a Babybear) -> Self::Output {
        self - *rhs
    }
}

impl<'a> Sub<&'a mut Babybear> for Babybear {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &'a mut Babybear) -> Self::Output {
        self - *rhs
    }
}

impl SubAssign for Babybear {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<'a> SubAssign<&'a Babybear> for Babybear {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a Babybear) {
        *self = *self - *rhs;
    }
}

impl<'a> SubAssign<&'a mut Babybear> for Babybear {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a mut Babybear) {
        *self = *self - *rhs;
    }
}

impl<T: ::core::borrow::Borrow<Babybear>> Sum<T> for Babybear {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        // This is faster than iter.reduce(|x, y| x + y).unwrap_or(Self::zero()) for iterators of length > 2.
        // There might be a faster reduction method possible for lengths <= 16 which avoids %.

        // This sum will not overflow so long as iter.len() < 2^33.
        let sum = iter.map(|x| x.borrow().0 as u64).sum::<u64>();
        Babybear((sum % MODULUS as u64) as u32)
    }
}

impl Mul for Babybear {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(((self.0 as u64) * (rhs.0 as u64) % MODULUS as u64) as u32)
    }
}

impl<'a> Mul<&'a Babybear> for Babybear {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &'a Babybear) -> Self::Output {
        self * *rhs
    }
}

impl<'a> Mul<&'a mut Babybear> for Babybear {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &'a mut Babybear) -> Self::Output {
        self * *rhs
    }
}

impl MulAssign for Babybear {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a> MulAssign<&'a Babybear> for Babybear {
    #[inline]
    fn mul_assign(&mut self, rhs: &'a Babybear) {
        *self = *self * *rhs;
    }
}

impl<'a> MulAssign<&'a mut Babybear> for Babybear {
    #[inline]
    fn mul_assign(&mut self, rhs: &'a mut Babybear) {
        *self = *self * *rhs;
    }
}

impl Div for Babybear {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        self * rhs.invert().unwrap()
    }
}

impl<T: ::core::borrow::Borrow<Babybear>> Product<T> for Babybear {
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, item| acc * item.borrow())
    }
}

impl Babybear {
    #[inline]
    pub fn to_canonical_u32(&self) -> u32 {
        let mut c = self.0;
        // We only need two condition subtraction, since 3 * ORDER would not fit in a u64.
        if c >= MODULUS {
            c -= MODULUS;
        }
        // We only need one condition subtraction, since 2 * ORDER would not fit in a u64.
        if c >= MODULUS {
            c -= MODULUS;
        }
        c
    }

    pub const fn size() -> usize {
        4
    }

    pub fn legendre(&self) -> LegendreSymbol {
        // s = self^((modulus - 1) // 2)
        // 1006632960
        let s = 0x3c000000;
        let s = self.pow_vartime([s]);
        if s == Self::ZERO {
            LegendreSymbol::Zero
        } else if s == Self::ONE {
            LegendreSymbol::QuadraticResidue
        } else {
            LegendreSymbol::QuadraticNonResidue
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum LegendreSymbol {
    Zero = 0,
    QuadraticResidue = 1,
    QuadraticNonResidue = -1,
}

impl CanonicalDeserialize for Babybear {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        _compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let mut bytes = [0u8; core::mem::size_of::<u32>()];
        reader.read_exact(&mut bytes)?;
        let value_u32 = u32::from_le_bytes(bytes);
        if validate == Validate::Yes {
            if value_u32 >= super::MODULUS {
                return Err(ark_serialize::SerializationError::InvalidData);
            }
        }
        Ok(Babybear(u32::from_le_bytes(bytes)))
    }
}

impl Valid for Babybear {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for Babybear {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        let bytes = self.to_canonical_u32().to_le_bytes();
        writer.write_all(&bytes)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        core::mem::size_of::<u32>()
    }
}
