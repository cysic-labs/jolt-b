use ff::{Field, PrimeField};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use crate::fp::Babybear;

pub(crate) fn sqrt_tonelli_shanks(f: &Babybear, tm1d2: u32) -> CtOption<Babybear> {
    // w = self^((t - 1) // 2)
    let w = f.pow_vartime([tm1d2 as u64]);

    let mut v = Babybear::S;
    let mut x = w * f;
    let mut b = x * w;

    // Initialize z as the 2^S root of unity.
    let mut z = Babybear::ROOT_OF_UNITY;

    for max_v in (1..=Babybear::S).rev() {
        let mut k = 1;
        let mut tmp = b.square();
        let mut j_less_than_v: Choice = 1.into();

        for j in 2..max_v {
            let tmp_is_one = tmp.ct_eq(&Babybear::ONE);
            let squared = Babybear::conditional_select(&tmp, &z, tmp_is_one).square();
            tmp = Babybear::conditional_select(&squared, &tmp, tmp_is_one);
            let new_z = Babybear::conditional_select(&z, &squared, tmp_is_one);
            j_less_than_v &= !j.ct_eq(&v);
            k = u32::conditional_select(&j, &k, tmp_is_one);
            z = Babybear::conditional_select(&z, &new_z, j_less_than_v);
        }

        let result = x * z;
        x = Babybear::conditional_select(&result, &x, b.ct_eq(&Babybear::ONE));
        z = z.square();
        b *= z;
        v = k;
    }
    CtOption::new(
        x,
        (x * x).ct_eq(f), // Only return Some if it's the square root.
    )
}
