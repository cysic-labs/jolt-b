macro_rules! impl_extension_field {
    ($base_field:ident, $ext_field:ident, $degree: expr) => {
        impl $ext_field {
            pub fn new_from_base(base_elems: &[$base_field]) -> Self {
                Self(base_elems.try_into().unwrap())
            }
        }

        impl ConstantTimeEq for $ext_field {
            fn ct_eq(&self, other: &Self) -> Choice {
                self.to_canonical_u64_vec()
                    .ct_eq(&other.to_canonical_u64_vec())
            }
        }

        impl<'a> Add<&'a $ext_field> for $ext_field {
            type Output = Self;

            #[inline]
            fn add(self, rhs: &'a $ext_field) -> Self::Output {
                self + *rhs
            }
        }

        impl<'a> Add<&'a mut $ext_field> for $ext_field {
            type Output = Self;

            #[inline]
            fn add(self, rhs: &'a mut $ext_field) -> Self::Output {
                self + *rhs
            }
        }

        impl AddAssign for $ext_field {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl<'a> AddAssign<&'a $ext_field> for $ext_field {
            #[inline]
            fn add_assign(&mut self, rhs: &'a $ext_field) {
                *self = *self + *rhs;
            }
        }

        impl<'a> AddAssign<&'a mut $ext_field> for $ext_field {
            #[inline]
            fn add_assign(&mut self, rhs: &'a mut $ext_field) {
                *self = *self + *rhs;
            }
        }

        impl<'a> Sub<&'a $ext_field> for $ext_field {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: &'a $ext_field) -> Self::Output {
                self - *rhs
            }
        }

        impl<'a> Sub<&'a mut $ext_field> for $ext_field {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: &'a mut $ext_field) -> Self::Output {
                self - *rhs
            }
        }

        impl SubAssign for $ext_field {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl<'a> SubAssign<&'a $ext_field> for $ext_field {
            #[inline]
            fn sub_assign(&mut self, rhs: &'a $ext_field) {
                *self = *self - *rhs;
            }
        }

        impl<'a> SubAssign<&'a mut $ext_field> for $ext_field {
            #[inline]
            fn sub_assign(&mut self, rhs: &'a mut $ext_field) {
                *self = *self - *rhs;
            }
        }

        impl<T: ::core::borrow::Borrow<$ext_field>> Sum<T> for $ext_field {
            fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
                iter.fold(Self::ZERO, |acc, item| acc + item.borrow())
            }
        }

        impl Mul for $ext_field {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self {
                mul_internal(&self, &rhs)
            }
        }

        impl<'a> Mul<&'a $ext_field> for $ext_field {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: &'a $ext_field) -> Self::Output {
                self * *rhs
            }
        }

        impl<'a> Mul<&'a mut $ext_field> for $ext_field {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: &'a mut $ext_field) -> Self::Output {
                self * *rhs
            }
        }

        impl MulAssign for $ext_field {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl<'a> MulAssign<&'a $ext_field> for $ext_field {
            #[inline]
            fn mul_assign(&mut self, rhs: &'a $ext_field) {
                *self = *self * *rhs;
            }
        }

        impl<'a> MulAssign<&'a mut $ext_field> for $ext_field {
            #[inline]
            fn mul_assign(&mut self, rhs: &'a mut $ext_field) {
                *self = *self * *rhs;
            }
        }

        impl<T: ::core::borrow::Borrow<$ext_field>> Product<T> for $ext_field {
            fn product<I: Iterator<Item = T>>(iter: I) -> Self {
                iter.fold(Self::ONE, |acc, item| acc * item.borrow())
            }
        }

        impl From<u64> for $ext_field {
            fn from(a: u64) -> Self {
                Babybear::from(a).into()
            }
        }
    };
}
