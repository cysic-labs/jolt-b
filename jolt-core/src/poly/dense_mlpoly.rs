#![allow(clippy::too_many_arguments)]
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::utils::{self, compute_dotproduct, compute_dotproduct_low_optimized};

use crate::poly::field::JoltField;
use crate::utils::math::Math;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use core::ops::Index;
use itertools::Itertools;
use rayon::prelude::*;
use std::ops::AddAssign;

#[derive(Debug, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub struct DensePolynomial<F: JoltField> {
    num_vars: usize, // the number of variables in the multilinear polynomial
    len: usize,
    Z: Vec<F>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
}

impl<F: JoltField> DensePolynomial<F> {
    pub fn new(Z: Vec<F>) -> Self {
        assert!(
            utils::is_power_of_two(Z.len()),
            "Dense multi-linear polynomials must be made from a power of 2 (not {})",
            Z.len()
        );

        DensePolynomial {
            num_vars: Z.len().log_2(),
            len: Z.len(),
            Z,
        }
    }

    pub fn new_padded(evals: Vec<F>) -> Self {
        // Pad non-power-2 evaluations to fill out the dense multilinear polynomial
        let mut poly_evals = evals;
        if !utils::is_power_of_two(poly_evals.len()) {
            let next_po2 = poly_evals.len().next_power_of_two();
            poly_evals.resize(next_po2, F::zero())
        }

        DensePolynomial {
            num_vars: poly_evals.len().log_2(),
            len: poly_evals.len(),
            Z: poly_evals,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn split(&self, idx: usize) -> (Self, Self) {
        assert!(idx < self.len());
        (
            Self::new(self.Z[..idx].to_vec()),
            Self::new(self.Z[idx..2 * idx].to_vec()),
        )
    }

    pub fn split_evals(&self, idx: usize) -> (&[F], &[F]) {
        (&self.Z[..idx], &self.Z[idx..])
    }

    pub fn bound_poly_var_top(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
            *a += *r * (*b - *a);
        });

        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_top_par(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        left.par_iter_mut()
            .zip(right.par_iter())
            .for_each(|(a, b)| {
                *a += *r * (*b - *a);
            });

        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_top_many_ones(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        left.iter_mut()
            .zip(right.iter())
            .filter(|(&mut a, &b)| a != b)
            .for_each(|(a, b)| {
                let m = *b - *a;
                if m.is_one() {
                    *a += *r;
                } else {
                    *a += *r * m;
                }
            });

        self.num_vars -= 1;
        self.len = n;
    }

    /// Bounds the polynomial's most significant index bit to 'r' optimized for a
    /// high P(eval = 0).
    #[tracing::instrument(skip_all)]
    pub fn bound_poly_var_top_zero_optimized(&mut self, r: &F) {
        let n = self.len() / 2;

        let (left, right) = self.Z.split_at_mut(n);

        left.par_iter_mut()
            .zip(right.par_iter())
            .filter(|(&mut a, &b)| a != b)
            .for_each(|(a, b)| {
                *a += *r * (*b - *a);
            });

        self.Z.resize(n, F::zero());
        self.num_vars -= 1;
        self.len = n;
    }

    #[tracing::instrument(skip_all)]
    pub fn new_poly_from_bound_poly_var_top(&self, r: &F) -> Self {
        let n = self.len() / 2;
        let mut new_evals: Vec<F> = unsafe_allocate_zero_vec(n);

        for i in 0..n {
            // let low' = low + r * (high - low)
            let low = self.Z[i];
            let high = self.Z[i + n];
            if !(low.is_zero() && high.is_zero()) {
                let m = high - low;
                new_evals[i] = low + *r * m;
            }
        }
        let num_vars = self.num_vars - 1;
        let len = n;

        Self {
            num_vars,
            len,
            Z: new_evals,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn new_poly_from_bound_poly_var_top_flags(&self, r: &F) -> Self {
        let n = self.len() / 2;
        let mut new_evals: Vec<F> = unsafe_allocate_zero_vec(n);

        for i in 0..n {
            // let low' = low + r * (high - low)
            // Special truth table here
            //         high 0   high 1
            // low 0     0        r
            // low 1   (1-r)      1
            let low = self.Z[i];
            let high = self.Z[i + n];

            if low.is_zero() {
                if high.is_one() {
                    new_evals[i] = *r;
                } else if !high.is_zero() {
                    panic!("Shouldn't happen for a flag poly");
                }
            } else if low.is_one() {
                if high.is_one() {
                    new_evals[i] = F::one();
                } else if high.is_zero() {
                    new_evals[i] = F::one() - r;
                } else {
                    panic!("Shouldn't happen for a flag poly");
                }
            }
        }
        let num_vars = self.num_vars - 1;
        let len = n;

        Self {
            num_vars,
            len,
            Z: new_evals,
        }
    }

    pub fn bound_poly_var_bot(&mut self, r: &F) {
        let n = self.len() / 2;
        for i in 0..n {
            self.Z[i] = self.Z[2 * i] + *r * (self.Z[2 * i + 1] - self.Z[2 * i]);
        }
        self.num_vars -= 1;
        self.len = n;
    }

    // returns Z(r) in O(n) time
    pub fn evaluate(&self, r: &[F]) -> F {
        // r must have a value for each variable
        assert_eq!(r.len(), self.get_num_vars());
        let chis = EqPolynomial::new(r.to_vec()).evals();
        assert_eq!(chis.len(), self.Z.len());
        compute_dotproduct(&self.Z, &chis)
    }

    pub fn interpolate_over_hypercube_impl(evals: &[F]) -> Vec<F> {
        let mut coeffs = evals.to_vec();
        let num_vars = coeffs.len().log_2();

        for i in 1..=num_vars {
            let chunk_size = 1 << i;

            coeffs.chunks_mut(chunk_size).for_each(|chunk| {
                let half_chunk = chunk_size >> 1;
                let (left, right) = chunk.split_at_mut(half_chunk);
                right.iter_mut().zip(left.iter()).for_each(|(a, b)| *a -= b);
            })
        }

        coeffs
    }

    // interpolate Z evaluations over boolean hypercube {0, 1}^n
    pub fn interpolate_over_hypercube(&self) -> Vec<F> {
        // Take eq poly as an example:
        //
        // The evaluation format of an eq poly over {0, 1}^2 follows:
        // eq(\vec{r}, \vec{x}) with \vec{x} order in x0 x1
        //
        //     00             01            10          11
        // (1-r0)(1-r1)    (1-r0)r1      r0(1-r1)      r0r1
        //
        // The interpolated version over x0 x1 (ordered in x0 x1) follows:
        //
        //     00               01                  10                11
        // (1-r0)(1-r1)    (1-r0)(2r1-1)      (2r0-1)(1-r1)     (2r0-1)(2r1-1)

        // NOTE(Hang): I think the original implementation of this dense multilinear
        // polynomial requires a resizing of coeffs by num vars,
        // e.g., when sumchecking - the num_var reduces, while Z evals can reuse the
        // whole space, which means we cannot simply relying Z's size itself.
        Self::interpolate_over_hypercube_impl(&self.Z[..self.len()])
    }

    // NOTE: we are assuming that the polys are of the same size, but the polys
    // can be (and should be) of different sized, yet our implementation only
    // provide a simplified version.
    pub fn low_degree_extension(polys: &[&Self]) -> Self {
        // NOTE: the order is reversed w.r.t. bit index, i.e.,
        // supposing polys share x_0 to x_n, while LDE append k variables,
        // then polynomials are ordered by x_0 ... x_{k - 1}, x_k, ... x_{n + k},
        // where the original coefficients shift right by k.

        let new_z = polys.iter().map(|p| p.evals()).concat();
        Self::new_padded(new_z)
    }

    // on given an evaluation point x_0...x_n,
    // output a tensor product of x_j product of order:
    // 1, x_n, x_{n-1}, x_n x_{n-1}, x_{n-2} ..., \prod x_i
    //
    // the inner product of the tensor product vec with boolean hypercube interpolation
    // should yield the evaluation of the polynomial on the point.
    #[allow(unused)]
    pub fn eval_point_tensor_product(point: &[F]) -> Vec<F> {
        let tensor_prod_size = 1 << point.len();
        let mut eval_tensor_prod: Vec<F> = vec![F::one(); tensor_prod_size];

        for packed_indices in 0..tensor_prod_size {
            for (j, x_j) in point.iter().rev().enumerate() {
                if (packed_indices >> j) & 1 != 0 {
                    eval_tensor_prod[packed_indices] *= x_j;
                }
            }
        }

        eval_tensor_prod
    }

    pub fn evaluate_at_chi(&self, chis: &[F]) -> F {
        compute_dotproduct(&self.Z, chis)
    }

    pub fn evaluate_at_chi_low_optimized(&self, chis: &[F]) -> F {
        assert_eq!(self.Z.len(), chis.len());
        compute_dotproduct_low_optimized(&self.Z, chis)
    }

    pub fn evals(&self) -> Vec<F> {
        self.Z.clone()
    }

    pub fn evals_ref(&self) -> &[F] {
        self.Z.as_ref()
    }

    #[tracing::instrument(skip_all, name = "DensePoly::flatten")]
    pub fn flatten(polys: &[DensePolynomial<F>]) -> Vec<F> {
        let poly_len = polys[0].len();
        polys
            .iter()
            .for_each(|poly| assert_eq!(poly_len, poly.len()));

        let num_polys = polys.len();
        let flat_len = num_polys * poly_len;
        let mut flat: Vec<F> = unsafe_allocate_zero_vec(flat_len);
        flat.par_chunks_mut(poly_len)
            .enumerate()
            .for_each(|(poly_index, result)| {
                let evals = polys[poly_index].evals_ref();
                for (eval_index, eval) in evals.iter().enumerate() {
                    result[eval_index] = *eval;
                }
            });
        flat
    }

    #[tracing::instrument(skip_all, name = "DensePolynomial::from")]
    pub fn from_usize(Z: &[usize]) -> Self {
        DensePolynomial::new(
            (0..Z.len())
                .map(|i| F::from_u64(Z[i] as u64).unwrap())
                .collect::<Vec<F>>(),
        )
    }

    #[tracing::instrument(skip_all, name = "DensePolynomial::from")]
    pub fn from_u64(Z: &[u64]) -> Self {
        DensePolynomial::new(
            (0..Z.len())
                .map(|i| F::from_u64(Z[i]).unwrap())
                .collect::<Vec<F>>(),
        )
    }
}

impl<F: JoltField> Clone for DensePolynomial<F> {
    fn clone(&self) -> Self {
        Self::new(self.Z[0..self.len].to_vec())
    }
}

impl<F: JoltField> Index<usize> for DensePolynomial<F> {
    type Output = F;

    #[inline(always)]
    fn index(&self, _index: usize) -> &F {
        &(self.Z[_index])
    }
}

impl<F: JoltField> AsRef<DensePolynomial<F>> for DensePolynomial<F> {
    fn as_ref(&self) -> &DensePolynomial<F> {
        self
    }
}

impl<F: JoltField> AddAssign<&DensePolynomial<F>> for DensePolynomial<F> {
    fn add_assign(&mut self, rhs: &DensePolynomial<F>) {
        assert_eq!(self.num_vars, rhs.num_vars);
        assert_eq!(self.len, rhs.len);
        let summed_evaluations: Vec<F> = self.Z.iter().zip(&rhs.Z).map(|(a, b)| *a + *b).collect();

        *self = Self {
            num_vars: self.num_vars,
            len: self.len,
            Z: summed_evaluations,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::poly::commitment::hyrax::matrix_dimensions;

    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_core::RngCore;

    fn random_vec<F: JoltField>(size: usize, rng: &mut impl RngCore) -> Vec<F> {
        (0..size).map(|_| F::random(rng)).collect()
    }

    fn evaluate_with_LR<F: JoltField>(Z: &[F], r: &[F]) -> F {
        let ell = r.len();
        let (L_size, _R_size) = matrix_dimensions(ell, 1);
        let eq = EqPolynomial::<F>::new(r.to_vec());
        let (L, R) = eq.compute_factored_evals(L_size);

        // ensure ell is even
        assert!(ell % 2 == 0);
        // compute n = 2^\ell
        let n = ell.pow2();
        // compute m = sqrt(n) = 2^{\ell/2}
        let m = n.square_root();

        // compute vector-matrix product between L and Z viewed as a matrix
        let LZ = (0..m)
            .map(|i| (0..m).map(|j| L[j] * Z[j * m + i]).sum())
            .collect::<Vec<F>>();

        // compute dot product between LZ and R
        compute_dotproduct(&LZ, &R)
    }

    #[test]
    fn check_polynomial_evaluation() {
        check_polynomial_evaluation_helper::<Fr>()
    }

    fn check_polynomial_evaluation_helper<F: JoltField>() {
        // Z = [1, 2, 1, 4]
        let Z = vec![
            F::one(),
            F::from_u64(2u64).unwrap(),
            F::one(),
            F::from_u64(4u64).unwrap(),
        ];

        // r = [4,3]
        let r = vec![F::from_u64(4u64).unwrap(), F::from_u64(3u64).unwrap()];

        let eval_with_LR = evaluate_with_LR::<F>(&Z, &r);
        let poly = DensePolynomial::new(Z);

        let eval = poly.evaluate(&r);
        assert_eq!(eval, F::from_u64(28u64).unwrap());
        assert_eq!(eval_with_LR, eval);
    }

    pub fn compute_factored_chis_at_r<F: JoltField>(r: &[F]) -> (Vec<F>, Vec<F>) {
        let mut L: Vec<F> = Vec::new();
        let mut R: Vec<F> = Vec::new();

        let ell = r.len();
        assert!(ell % 2 == 0); // ensure ell is even
        let n = ell.pow2();
        let m = n.square_root();

        // compute row vector L
        for i in 0..m {
            let mut chi_i = F::one();
            for j in 0..ell / 2 {
                let bit_j = ((m * i) & (1 << (r.len() - j - 1))) > 0;
                if bit_j {
                    chi_i *= r[j];
                } else {
                    chi_i *= F::one() - r[j];
                }
            }
            L.push(chi_i);
        }

        // compute column vector R
        for i in 0..m {
            let mut chi_i = F::one();
            for j in ell / 2..ell {
                let bit_j = (i & (1 << (r.len() - j - 1))) > 0;
                if bit_j {
                    chi_i *= r[j];
                } else {
                    chi_i *= F::one() - r[j];
                }
            }
            R.push(chi_i);
        }
        (L, R)
    }

    pub fn compute_chis_at_r<F: JoltField>(r: &[F]) -> Vec<F> {
        let ell = r.len();
        let n = ell.pow2();
        let mut chis: Vec<F> = Vec::new();
        for i in 0..n {
            let mut chi_i = F::one();
            for j in 0..r.len() {
                let bit_j = (i & (1 << (r.len() - j - 1))) > 0;
                if bit_j {
                    chi_i *= r[j];
                } else {
                    chi_i *= F::one() - r[j];
                }
            }
            chis.push(chi_i);
        }
        chis
    }

    pub fn compute_outerproduct<F: JoltField>(L: &[F], R: &[F]) -> Vec<F> {
        assert_eq!(L.len(), R.len());
        (0..L.len())
            .map(|i| (0..R.len()).map(|j| L[i] * R[j]).collect::<Vec<F>>())
            .collect::<Vec<Vec<F>>>()
            .into_iter()
            .flatten()
            .collect::<Vec<F>>()
    }

    #[test]
    fn check_memoized_chis() {
        check_memoized_chis_helper::<Fr>()
    }

    fn check_memoized_chis_helper<F: JoltField>() {
        let mut prng = test_rng();

        let r = random_vec(10, &mut prng);
        let chis = compute_chis_at_r::<F>(&r);
        let chis_m = EqPolynomial::<F>::new(r).evals();
        assert_eq!(chis, chis_m);
    }

    #[test]
    fn check_factored_chis() {
        check_factored_chis_helper::<Fr>()
    }

    fn check_factored_chis_helper<F: JoltField>() {
        let mut prng = test_rng();

        let r: Vec<F> = random_vec(10, &mut prng);
        let chis = EqPolynomial::new(r.clone()).evals();
        let (L_size, _R_size) = matrix_dimensions(r.len(), 1);
        let (L, R) = EqPolynomial::new(r).compute_factored_evals(L_size);
        let O = compute_outerproduct(&L, &R);
        assert_eq!(chis, O);
    }

    #[test]
    fn check_memoized_factored_chis() {
        check_memoized_factored_chis_helper::<Fr>()
    }

    fn check_memoized_factored_chis_helper<F: JoltField>() {
        let mut prng = test_rng();

        let r: Vec<F> = random_vec(10, &mut prng);
        let (L_size, _R_size) = matrix_dimensions(r.len(), 1);
        let (L, R) = compute_factored_chis_at_r(&r);
        let eq = EqPolynomial::new(r);
        let (L2, R2) = eq.compute_factored_evals(L_size);
        assert_eq!(L, L2);
        assert_eq!(R, R2);
    }

    #[test]
    fn evaluation() {
        let num_evals = 4;
        let mut evals: Vec<Fr> = Vec::with_capacity(num_evals);
        for _ in 0..num_evals {
            evals.push(Fr::from(8));
        }
        let dense_poly: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());

        // Evaluate at 3:
        // (0, 0) = 1
        // (0, 1) = 1
        // (1, 0) = 1
        // (1, 1) = 1
        // g(x_0,x_1) => c_0*(1 - x_0)(1 - x_1) + c_1*(1-x_0)(x_1) + c_2*(x_0)(1-x_1) + c_3*(x_0)(x_1)
        // g(3, 4) = 8*(1 - 3)(1 - 4) + 8*(1-3)(4) + 8*(3)(1-4) + 8*(3)(4) = 48 + -64 + -72 + 96  = 8
        // g(5, 10) = 8*(1 - 5)(1 - 10) + 8*(1 - 5)(10) + 8*(5)(1-10) + 8*(5)(10) = 96 + -16 + -72 + 96  = 8
        assert_eq!(
            dense_poly.evaluate(vec![Fr::from(3), Fr::from(4)].as_slice()),
            Fr::from(8)
        );
    }

    #[test]
    fn test_interpolation_evaulation_eq() {
        let mut rng = test_rng();

        let num_var = 10;
        let r_vec: Vec<Fr> = random_vec(num_var, &mut rng);

        let eq_poly = EqPolynomial::new(r_vec);
        let eq_evals = eq_poly.evals();

        let dense_poly = DensePolynomial::new(eq_evals.clone());
        let eval_point: Vec<Fr> = random_vec(num_var, &mut rng);
        let ideal_eval: Fr = dense_poly.evaluate(&eval_point);

        let eval_tensor_prod = DensePolynomial::eval_point_tensor_product(&eval_point);
        let dense_poly_interpolation = dense_poly.interpolate_over_hypercube();
        let eval = compute_dotproduct(&eval_tensor_prod, &dense_poly_interpolation);

        assert_eq!(eval, ideal_eval);
    }
}
