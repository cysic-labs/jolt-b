use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, start_timer};
use itertools::Itertools;
use octopos::{
    hash::{Digest as OctoposDigest, OctoposHasherTrait},
    path::OctoposPath,
    tree::{AbstractOracle, OctoposTree},
};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use std::{cmp, marker::PhantomData};

use crate::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, field::JoltField},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{
        compute_dotproduct,
        errors::ProofVerifyError,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};

use super::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};

#[derive(Clone)]
pub struct BasefoldCommitmentScheme<
    F: JoltField + TwoAdicField,
    ExtF: JoltField + TwoAdicField + From<F>,
    H: OctoposHasherTrait + Sync,
> {
    _marker_f: PhantomData<F>,
    _marker_ext_f: PhantomData<ExtF>,
    _marker_h: PhantomData<H>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BasefoldCommitment<F: JoltField + TwoAdicField> {
    merkle: OctoposTree<F>,
}

impl<F: JoltField + TwoAdicField> AppendToTranscript for BasefoldCommitment<F> {
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript) {
        transcript.append_message(label, b"poly_commitment_begin");
        transcript.append_bytes(label, self.merkle.root().as_u8s());
        transcript.append_message(label, b"poly_commitment_end");
    }
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct BasefoldIOPPQuerySingleRound {
    left: OctoposPath,
    right: OctoposPath,
}

impl BasefoldIOPPQuerySingleRound {
    pub fn check_expected_codeword<F: JoltField>(
        &self,
        entry_index: usize,
        oracle_len: usize,
        entry: &F,
    ) -> bool {
        let side = &match entry_index & (oracle_len >> 1) {
            0 => self.left.leaf_value::<F>(),
            _ => self.right.leaf_value::<F>(),
        };

        *side == *entry
    }
}

#[derive(Debug, CanonicalDeserialize, CanonicalSerialize)]
pub struct BasefoldIOPPQuery {
    // NOTE: the folding r's are in sumcheck verification, deriving from Fiat-Shamir.
    iopp_round_query: Vec<BasefoldIOPPQuerySingleRound>,
}

impl BasefoldIOPPQuery {
    fn verify_iopp_query<
        'a,
        F: JoltField + TwoAdicField,
        ExtF: JoltField + TwoAdicField + From<F>,
        H: OctoposHasherTrait + Sync + Send,
    >(
        iopp_round_query: &[BasefoldIOPPQuerySingleRound],
        setup: &BasefoldPP<F, ExtF, H>,
        challenge_point: usize,
        oracles: impl IndexedParallelIterator<Item = &'a OctoposDigest> + Clone,
        folding_rs: &[ExtF],
        is_leading_over_f: bool,
    ) -> bool {
        let hasher = H::new_instance();
        let num_vars = folding_rs.len();

        let ((iopp_round_query_f, oracles_f), (iopp_round_query_extf, oracles_extf)) =
            match is_leading_over_f {
                true => (
                    (&iopp_round_query[..1], oracles.clone().take(1)),
                    (&iopp_round_query[1..], oracles.skip(1)),
                ),
                _ => (
                    (&iopp_round_query[..0], oracles.clone().take(0)),
                    (&iopp_round_query[0..], oracles.skip(0)),
                ),
            };

        // check merkle trees against base field or extension field
        let mt_verify_f =
            iopp_round_query_f
                .par_iter()
                .zip(oracles_f)
                .all(|(round_i_query, root_i)| {
                    let left = round_i_query.left.verify::<F, _>(root_i, &hasher);
                    let right = round_i_query.right.verify::<F, _>(root_i, &hasher);

                    left && right
                });

        let mt_verify_extf =
            iopp_round_query_extf
                .par_iter()
                .zip(oracles_extf)
                .all(|(round_i_query, root_i)| {
                    let left = round_i_query.left.verify::<ExtF, _>(root_i, &hasher);
                    let right = round_i_query.right.verify::<ExtF, _>(root_i, &hasher);

                    left && right
                });

        if !mt_verify_f || !mt_verify_extf {
            return false;
        }

        // Check IOPP query results
        let iopp_query_f_res_iter = iopp_round_query_f.iter().map(|q| {
            let l_f = q.left.leaf_value::<F>();
            let r_f = q.right.leaf_value::<F>();

            let l_ef: ExtF = <F>::into(l_f);
            let r_ef: ExtF = <F>::into(r_f);

            (l_ef, r_ef)
        });

        let iopp_query_ef_res_iter = iopp_round_query_extf
            .iter()
            .map(|q| (q.left.leaf_value::<ExtF>(), q.right.leaf_value::<ExtF>()));

        let iopp_query_res_iter = iopp_query_f_res_iter.chain(iopp_query_ef_res_iter);

        let mut point = challenge_point;
        iopp_query_res_iter
            .tuple_windows()
            .enumerate()
            .all(|(round_i, ((c1, c2), (nc1, nc2)))| {
                let oracle_rhs_start = 1 << (setup.codeword_bits(num_vars) - round_i - 1);
                let sibling_point = point ^ oracle_rhs_start;
                let left_index = cmp::min(point, sibling_point);

                let g1: ExtF = setup.t_term::<ExtF>(num_vars, round_i, left_index).into();
                let g2 = -g1;

                // interpolate y = b + kx form
                let k = (c2 - c1) / (g2 - g1);
                let b = c1 - k * g1;
                let expected_codeword = b + k * folding_rs[round_i];

                point = left_index;

                let actual_codeword = match left_index & (oracle_rhs_start >> 1) {
                    0 => nc1,
                    _ => nc2,
                };

                actual_codeword == expected_codeword
            })
    }

    pub fn verify<
        'a,
        F: JoltField + TwoAdicField,
        ExtF: JoltField + TwoAdicField + From<F>,
        H: OctoposHasherTrait + Send + Sync,
    >(
        &self,
        setup: &BasefoldPP<F, ExtF, H>,
        challenge_point: usize,
        oracles: impl IndexedParallelIterator<Item = &'a OctoposDigest> + Clone,
        folding_rs: &[ExtF],
    ) -> bool {
        Self::verify_iopp_query(
            &self.iopp_round_query,
            setup,
            challenge_point,
            oracles,
            folding_rs,
            true,
        )
    }
}

pub struct BasefoldVirtualOracle<'a, 'b, F: JoltField + TwoAdicField> {
    commitments: &'a [&'b BasefoldCommitment<F>],
}

impl<'a, 'b, F: JoltField + TwoAdicField> BasefoldVirtualOracle<'a, 'b, F> {
    fn new(commitments: &'a [&'b BasefoldCommitment<F>]) -> Self {
        Self { commitments }
    }
}

impl<'a, 'b, F: JoltField + TwoAdicField> AbstractOracle for BasefoldVirtualOracle<'a, 'b, F> {
    type QueryResult = Vec<OctoposPath>;

    #[inline]
    fn index_query(&self, index: usize) -> Self::QueryResult {
        self.commitments
            .par_iter()
            .map(|o| o.merkle.index_query(index))
            .collect()
    }

    #[inline]
    fn size(&self) -> usize {
        self.commitments[0].merkle.size()
    }
}

#[derive(Debug, CanonicalDeserialize, CanonicalSerialize)]
pub struct BasefoldVirtualIOPPQuery {
    virtual_queries: Vec<BasefoldIOPPQuerySingleRound>,
    iopp_query: BasefoldIOPPQuery,
}

impl BasefoldVirtualIOPPQuery {
    #[inline]
    fn deteriorate_to_basefold_iopp_query(&self) -> BasefoldIOPPQuery {
        // NOTE: the deterioration happens only when there is only one virtual query,
        // namely, using batch for one single polynomial.
        assert_eq!(self.virtual_queries.len(), 1);

        let mut iopp_round_query = self.virtual_queries.clone();
        iopp_round_query.extend_from_slice(&self.iopp_query.iopp_round_query);
        BasefoldIOPPQuery { iopp_round_query }
    }

    pub fn verify<
        'a,
        F: JoltField + TwoAdicField,
        ExtF: JoltField + TwoAdicField + From<F>,
        H: OctoposHasherTrait + Send + Sync,
    >(
        &self,
        setup: &BasefoldPP<F, ExtF, H>,
        challenge_point: usize,
        virtual_oracles: impl IndexedParallelIterator<Item = &'a OctoposDigest> + Clone,
        query_oracles: impl IndexedParallelIterator<Item = &'a OctoposDigest> + Clone,
        batching_rs: &[ExtF],
        folding_rs: &[ExtF],
    ) -> bool {
        if virtual_oracles.len() == 1 {
            // directly clone for there is only one query and one oracle
            let vanilla_iopp_query = self.deteriorate_to_basefold_iopp_query();
            let vanilla_oracles = virtual_oracles.chain(query_oracles);
            return vanilla_iopp_query.verify(setup, challenge_point, vanilla_oracles, folding_rs);
        }

        let hasher = H::new_instance();
        let num_vars = query_oracles.len();

        let virtual_mt_verify =
            self.virtual_queries
                .par_iter()
                .zip(virtual_oracles)
                .all(|(query, root)| {
                    let left = query.left.verify::<F, _>(root, &hasher);
                    let right = query.right.verify::<F, _>(root, &hasher);
                    left && right
                });

        if !virtual_mt_verify {
            return false;
        }

        let oracle_rhs_start = 1 << (setup.codeword_bits(num_vars) - 1);
        let sibling_point = challenge_point ^ oracle_rhs_start;
        let left_index = cmp::min(challenge_point, sibling_point);

        let g1: ExtF = setup.t_term::<F>(num_vars, 0, left_index).into();
        let g2: ExtF = -g1;

        let (c1s, c2s): (Vec<ExtF>, Vec<ExtF>) = self
            .virtual_queries
            .iter()
            .map(|q| {
                let left_q = q.left.leaf_value::<F>();
                let right_q = q.right.leaf_value::<F>();
                let left_lifted: ExtF = left_q.into();
                let right_lifted: ExtF = right_q.into();
                (left_lifted, right_lifted)
            })
            .unzip();

        let c1 = compute_dotproduct(&c1s, &batching_rs);
        let c2 = compute_dotproduct(&c2s, &batching_rs);

        let k = (c2 - c1) / (g2 - g1);
        let b = c1 - k * g1;
        let expected_codeword = b + k * folding_rs[0];

        if !self.iopp_query.iopp_round_query[0].check_expected_codeword(
            left_index,
            oracle_rhs_start,
            &expected_codeword,
        ) {
            return false;
        }

        BasefoldIOPPQuery::verify_iopp_query(
            &self.iopp_query.iopp_round_query,
            setup,
            left_index,
            query_oracles,
            &folding_rs[1..],
            false,
        )
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct BasefoldProof<ExtF: JoltField + TwoAdicField> {
    sumcheck_transcript: SumcheckInstanceProof<ExtF>,
    iopp_oracles: Vec<OctoposDigest>,
    iopp_last_oracle_message: Vec<ExtF>,
    iopp_queries: Vec<BasefoldIOPPQuery>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct BasefoldBatchedProof<ExtF: JoltField + TwoAdicField> {
    sumcheck_transcript: SumcheckInstanceProof<ExtF>,
    iopp_oracles: Vec<OctoposDigest>,
    iopp_last_oracle_message: Vec<ExtF>,
    iopp_queries: Vec<BasefoldVirtualIOPPQuery>,
}

pub const BASEFOLD_ADDITIONAL_RATE_BITS: usize = 3;

pub const SECURITY_BITS: usize = 100;

pub const MERGE_POLY_DEG: usize = 2;

pub const BASEFOLD_IOPP_CHALLENGE_TAG: &[u8] = b"basefold IOPP query challenge";

pub const BASEFOLD_BATCH_OPENING_CHALLENGE_TAG: &[u8] = b"basefold batch opening challenge";

#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct BasefoldPP<
    F: JoltField + TwoAdicField,
    ExtF: JoltField + TwoAdicField + From<F>,
    H: OctoposHasherTrait + Sync + Send,
> {
    pub rate_bits: usize,
    pub verifier_queries: usize,
    _marker_f: PhantomData<F>,
    _marker_ext_f: PhantomData<ExtF>,
    _marker_h: PhantomData<H>,
}

impl<
        F: JoltField + TwoAdicField,
        ExtF: JoltField + TwoAdicField + From<F>,
        H: OctoposHasherTrait + Sync + Send,
    > BasefoldPP<F, ExtF, H>
{
    pub fn new(rate_bits: usize) -> Self {
        let verifier_queries = 33;

        Self {
            rate_bits,
            verifier_queries,
            _marker_f: PhantomData,
            _marker_ext_f: PhantomData,
            _marker_h: PhantomData,
        }
    }

    #[inline]
    pub fn iopp_challenges(&self, num_vars: usize, transcript: &mut ProofTranscript) -> Vec<usize> {
        let iopp_challenge_bitmask = (1 << self.codeword_bits(num_vars)) - 1;
        // NOTE: Fiat-Shamir sampling an IOPP query point ranging [0, 2^codeword_bits - 1].
        transcript
            .challenge_usizes(BASEFOLD_IOPP_CHALLENGE_TAG, self.verifier_queries)
            .par_iter()
            .map(|c| c & iopp_challenge_bitmask)
            .collect()
    }

    #[inline]
    pub fn codeword_bits(&self, num_vars: usize) -> usize {
        self.rate_bits + num_vars
    }

    #[inline]
    pub fn t_term<T: JoltField + TwoAdicField>(
        &self,
        num_vars: usize,
        round: usize,
        index: usize,
    ) -> T {
        let round_gen = T::two_adic_generator(self.codeword_bits(num_vars) - round);
        round_gen.exp_u64(index as u64)
    }

    pub fn reed_solomon_from_coeffs<T: JoltField + TwoAdicField>(
        &self,
        mut coeffs: Vec<T>,
    ) -> Vec<T> {
        plonky2_util::reverse_index_bits_in_place(&mut coeffs);
        let extended_length = coeffs.len() << self.rate_bits;
        coeffs.resize(extended_length, <T as JoltField>::zero());
        p3_dft::Radix2DitParallel.dft(coeffs)
    }

    /// Performs dft in batch. returns a vector that is concatenated from all the dft results.
    fn batch_reed_solomon_from_coeff_vecs<T: JoltField + TwoAdicField>(
        &self,
        mut coeff_vecs: Vec<Vec<T>>,
    ) -> Vec<Vec<T>> {
        let length = coeff_vecs[0].len();
        let num_poly = coeff_vecs.len();
        let extended_length = length << self.rate_bits;

        let timer = start_timer!(|| "reverse index bits in batch rs code");
        coeff_vecs.par_iter_mut().for_each(|coeffs| {
            plonky2_util::reverse_index_bits_in_place(coeffs);
        });
        end_timer!(timer);

        let timer = start_timer!(|| "dft in batch rs code");
        // transpose the vector to make it suitable for batch dft
        // somehow manually transpose the vector is faster than DenseMatrix.transpose()
        let mut buf = vec![<T as JoltField>::zero(); coeff_vecs.len() * extended_length];
        coeff_vecs.iter().enumerate().for_each(|(i, coeffs)| {
            coeffs.iter().enumerate().for_each(|(j, &coeff)| {
                buf[num_poly * j + i] = coeff;
            });
        });
        drop(coeff_vecs);

        let dft_res = p3_dft::Radix2DitParallel
            .dft_batch(RowMajorMatrix::new(buf, num_poly))
            .to_row_major_matrix()
            .values;
        end_timer!(timer);

        let timer = start_timer!(|| "transpose vector in batch rs code");
        // somehow manually transpose the vector is faster than DenseMatrix.transpose()
        let mut res = vec![Vec::with_capacity(extended_length); num_poly];
        res.par_iter_mut().enumerate().for_each(|(i, r)| {
            dft_res.chunks_exact(num_poly).for_each(|chunk| {
                r.push(chunk[i]);
            });
        });
        end_timer!(timer);
        res
    }

    fn batch_basefold_oracle_from_slices(&self, evals: &[&[F]]) -> Vec<OctoposTree<F>> {
        let timer = start_timer!(|| "interpolate over hypercube");
        let coeffs: Vec<Vec<F>> = evals
            .par_iter()
            .map(|&evals| DensePolynomial::interpolate_over_hypercube_impl(evals))
            .collect();
        end_timer!(timer);

        let timer = start_timer!(|| "batch rs from coeffs");
        let rs_codes = self.batch_reed_solomon_from_coeff_vecs(coeffs);
        end_timer!(timer);

        let timer = start_timer!(|| "new from leaves");
        let hasher = H::new_instance();
        let trees = rs_codes
            .par_iter()
            .map(|codeword| OctoposTree::new_from_leaves(codeword.to_vec(), &hasher))
            .collect::<Vec<_>>();
        end_timer!(timer);
        trees
    }

    pub fn basefold_oracle_from_poly(&self, poly: &DensePolynomial<F>) -> OctoposTree<F> {
        let timer =
            start_timer!(|| format!("basefold oracle from poly of {} vars", poly.get_num_vars()));
        let timer2 = start_timer!(|| "interpolate over hypercube");
        let coeffs = poly.interpolate_over_hypercube();
        end_timer!(timer2);

        let timer2 = start_timer!(|| "reed solomon from coeffs");
        let codeword = self.reed_solomon_from_coeffs(coeffs);
        end_timer!(timer2);

        let timer2 = start_timer!(|| "new from leaves");
        let hasher = H::new_instance();
        let tree = OctoposTree::new_from_leaves(codeword, &hasher);
        end_timer!(timer2);
        end_timer!(timer);
        tree
    }
}

impl<
        F: JoltField + TwoAdicField,
        ExtF: JoltField + TwoAdicField + From<F>,
        H: OctoposHasherTrait + Sync + Send + Clone + 'static,
    > CommitmentScheme for BasefoldCommitmentScheme<F, ExtF, H>
{
    type Field = F;
    type Setup = BasefoldPP<F, ExtF, H>;
    type Commitment = BasefoldCommitment<F>;
    type Proof = BasefoldProof<ExtF>;
    type BatchedProof = BasefoldBatchedProof<ExtF>;

    fn setup(_shapes: &[CommitShape]) -> Self::Setup {
        BasefoldPP::new(BASEFOLD_ADDITIONAL_RATE_BITS)
    }

    fn commit(poly: &DensePolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        let timer =
            start_timer!(|| format!("basefold commit poly with {} vars", poly.get_num_vars()));
        let commit = BasefoldCommitment {
            merkle: setup.basefold_oracle_from_poly(poly),
        };
        end_timer!(timer);

        commit
    }

    /// Commit a set of poly in a batch. Slightly faster than committing one by one.
    fn batch_commit(
        evals: &[&[Self::Field]],
        setup: &Self::Setup,
        _batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
        let timer = start_timer!(|| format!(
            "basefold batch commit {} polys of length {}",
            evals.len(),
            evals[0].len()
        ));
        let trees = setup.batch_basefold_oracle_from_slices(evals);

        let res = trees
            .into_iter()
            .map(|t| BasefoldCommitment { merkle: t })
            .collect();

        end_timer!(timer);
        res
    }

    fn prove(
        poly: &DensePolynomial<Self::Field>,
        setup: &Self::Setup,
        commitment: &Self::Commitment,
        opening_point: &[Self::Field],
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let timer =
            start_timer!(|| format!("basefold prove poly with {} vars", poly.get_num_vars()));

        let shift_z =
            EqPolynomial::new(opening_point.iter().cloned().map(Into::into).collect_vec())
                .to_dense();
        let mut sumcheck_poly_vec = vec![poly.lift_to::<ExtF>(), shift_z];
        let merge_function = |x: &[ExtF]| x.iter().product::<ExtF>();

        let num_vars = poly.get_num_vars();

        let mut sumcheck_polys: Vec<_> = Vec::with_capacity(num_vars);
        let mut iopp_codewords: Vec<_> = Vec::with_capacity(num_vars);

        let hasher = H::new_instance();

        (0..num_vars).for_each(|_| {
            // NOTE: sumcheck a single step, r_i start from x_0 towards x_n
            let (sc_univariate_poly_i, _, _) = SumcheckInstanceProof::prove_arbitrary(
                &<ExtF as JoltField>::zero(),
                1,
                &mut sumcheck_poly_vec,
                merge_function,
                MERGE_POLY_DEG,
                transcript,
            );
            sumcheck_polys.push(sc_univariate_poly_i.compressed_polys[0].clone());
            drop(sc_univariate_poly_i);

            let coeffs = sumcheck_poly_vec[0].interpolate_over_hypercube();
            iopp_codewords.push(setup.reed_solomon_from_coeffs(coeffs));
        });

        let iopp_oracles = OctoposTree::batch_tree_for_recursive_oracles(iopp_codewords, &hasher);

        let iopp_last_oracle_message = iopp_oracles[iopp_oracles.len() - 1].leaves.clone();
        let iopp_challenges = setup.iopp_challenges(num_vars, transcript);

        let iopp_queries = (0..setup.verifier_queries)
            .zip(iopp_challenges)
            .map(|(_, mut point)| {
                let mut iopp_round_query = Vec::with_capacity(iopp_oracles.len() + 1);

                // Merkle queries over F
                let oracle_rhs_start = commitment.merkle.size() >> 1;
                let sibling_point = point ^ oracle_rhs_start;
                let left = cmp::min(point, sibling_point);
                let right = oracle_rhs_start + left;
                point = left;

                iopp_round_query.push(BasefoldIOPPQuerySingleRound {
                    left: commitment.merkle.index_query(left),
                    right: commitment.merkle.index_query(right),
                });

                // Merkle queries over ExtF
                iopp_oracles.iter().for_each(|oracle| {
                    // NOTE: since the oracle length is always a power of 2,
                    // so the right hand side starts from directly div by 2.
                    let oracle_rhs_start = oracle.size() >> 1;

                    // NOTE: dirty trick, oracle rhs starting index is a pow of 2.
                    // now that we want to find a sibling point w.r.t the index,
                    // by plus (or minus) against point, so xor should work.
                    let sibling_point = point ^ oracle_rhs_start;

                    let left = cmp::min(point, sibling_point);
                    let right = oracle_rhs_start + left;

                    // NOTE: update point for next round of IOPP querying
                    point = left;

                    iopp_round_query.push(BasefoldIOPPQuerySingleRound {
                        left: oracle.index_query(left),
                        right: oracle.index_query(right),
                    })
                });

                BasefoldIOPPQuery { iopp_round_query }
            })
            .collect();
        end_timer!(timer);

        BasefoldProof {
            sumcheck_transcript: SumcheckInstanceProof::new(sumcheck_polys),
            iopp_oracles: iopp_oracles.iter().map(|t| t.root()).collect(),
            iopp_last_oracle_message,
            iopp_queries,
        }
    }

    fn batch_prove(
        polynomials: &[&DensePolynomial<Self::Field>],
        setup: &Self::Setup,
        commitments: &[&Self::Commitment],
        opening_point: &[Self::Field],
        _openings: &[Self::Field],
        _batch_type: BatchType,
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        let timer = start_timer!(|| format!(
            "basefold batch prove {} polys of {} vars",
            polynomials.len(),
            polynomials[0].get_num_vars()
        ));
        // NOTE: first sample from Fiat-Shamir for the t vector that combines
        // multiple multilinear polynomials.
        let t_len: usize = polynomials.len() - 1;

        // NOTE: compose equality polynomial of joining opening point and t_vec
        let num_vars = opening_point.len();

        let (t_vec, opening_point_lifted) = {
            let mut t_vec =
                transcript.challenge_vector::<ExtF>(BASEFOLD_BATCH_OPENING_CHALLENGE_TAG, t_len);
            t_vec.insert(0, <ExtF as JoltField>::one());
            let point_lifted: Vec<ExtF> =
                opening_point.iter().cloned().map(Into::into).collect_vec();
            (t_vec, point_lifted)
        };

        let timer2 = start_timer!(|| "compose eq polynomial");
        let eq_z_t = EqPolynomial::new(opening_point_lifted).to_dense();
        end_timer!(timer2);

        // NOTE: batch polynomials into one, and group them for sumcheck inputs
        let timer2 = start_timer!(|| "many to 1 merging");
        let lifted_polys: Vec<_> = polynomials.iter().map(|p| p.lift_to()).collect_vec();
        let lde_polynomials = DensePolynomial::linear_combination(lifted_polys, &t_vec);
        end_timer!(timer2);

        let mut sumcheck_poly_vec = vec![lde_polynomials, eq_z_t];
        let merge_function = |x: &[ExtF]| x.iter().product::<ExtF>();

        // NOTE: declare sumcheck related variables
        let mut sumcheck_polys: Vec<_> = Vec::with_capacity(num_vars);
        let mut iopp_codewords: Vec<_> = Vec::with_capacity(num_vars);
        let virtual_oracle = BasefoldVirtualOracle::new(commitments);

        let hasher = H::new_instance();

        (0..num_vars).for_each(|_| {
            let (sc_univariate_poly_i, _, _) = SumcheckInstanceProof::prove_arbitrary(
                &<ExtF as JoltField>::zero(),
                1,
                &mut sumcheck_poly_vec,
                merge_function,
                MERGE_POLY_DEG,
                transcript,
            );
            sumcheck_polys.push(sc_univariate_poly_i.compressed_polys[0].clone());

            let coeffs = sumcheck_poly_vec[0].interpolate_over_hypercube();
            iopp_codewords.push(setup.reed_solomon_from_coeffs(coeffs));
        });

        let iopp_oracles = OctoposTree::batch_tree_for_recursive_oracles(iopp_codewords, &hasher);
        let iopp_last_oracle_message = iopp_oracles[iopp_oracles.len() - 1].leaves.clone();
        let iopp_challenges = setup.iopp_challenges(num_vars, transcript);

        let iopp_queries = (0..setup.verifier_queries)
            .into_par_iter()
            .zip(iopp_challenges)
            .map(|(_, mut point)| {
                // NOTE: run the first round in virtual oracle from multiple MTs
                let oracle_rhs_start = virtual_oracle.size() >> 1;
                let sibling_point = point ^ oracle_rhs_start;

                let left = cmp::min(point, sibling_point);
                let right = oracle_rhs_start + left;

                let left_queries = virtual_oracle.index_query(left);
                let right_queries = virtual_oracle.index_query(right);

                let virtual_queries = left_queries
                    .into_iter()
                    .zip(right_queries)
                    .map(|(l, r)| BasefoldIOPPQuerySingleRound { left: l, right: r })
                    .collect();

                point = left;

                let iopp_round_query = iopp_oracles
                    .iter()
                    .map(|oracle| {
                        // NOTE: since the oracle length is always a power of 2,
                        // so the right hand side starts from directly div by 2.
                        let oracle_rhs_start = oracle.size() >> 1;

                        // NOTE: dirty trick, oracle rhs starting index is a pow of 2.
                        // now that we want to find a sibling point w.r.t the index,
                        // by plus (or minus) against point, so xor should work.
                        let sibling_point = point ^ oracle_rhs_start;

                        let left = cmp::min(point, sibling_point);
                        let right = oracle_rhs_start + left;

                        // NOTE: update point for next round of IOPP querying
                        point = left;

                        BasefoldIOPPQuerySingleRound {
                            left: oracle.index_query(left),
                            right: oracle.index_query(right),
                        }
                    })
                    .collect();

                BasefoldVirtualIOPPQuery {
                    virtual_queries,
                    iopp_query: BasefoldIOPPQuery { iopp_round_query },
                }
            })
            .collect();

        end_timer!(timer);
        BasefoldBatchedProof {
            sumcheck_transcript: SumcheckInstanceProof::new(sumcheck_polys),
            iopp_oracles: iopp_oracles.iter().map(|t| t.root()).collect(),
            iopp_last_oracle_message,
            iopp_queries,
        }
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let num_vars = opening_point.len();

        let opening_lifted: ExtF = <Self::Field as Into<ExtF>>::into(*opening);
        let opening_point_lifted: Vec<ExtF> =
            opening_point.iter().cloned().map(Into::into).collect_vec();

        // NOTE: check sumcheck statement:
        // f(z) = \sum_{r \in {0, 1}^n} (f(r) \eq(r, z)) can be reduced to
        // f_r_eq_zr = f(rs) \eq(rs, z)
        let (f_r_eq_zr, rs) = proof.sumcheck_transcript.verify(
            opening_lifted,
            num_vars,
            MERGE_POLY_DEG,
            transcript,
        )?;

        let eq_zr = EqPolynomial::new(opening_point_lifted).evaluate(&rs);
        let f_r = f_r_eq_zr / eq_zr;

        // NOTE: Basefold IOPP fold each round with rs (backwards),
        // so the last round of RS code should be all f(rs).
        if proof.iopp_last_oracle_message.len() != 1 << setup.rate_bits {
            return Err(ProofVerifyError::InternalError);
        }

        if proof.iopp_last_oracle_message.iter().any(|&x| x != f_r) {
            return Err(ProofVerifyError::InternalError);
        }

        let commitment_root = commitment.merkle.root();
        let oracles = rayon::iter::once(&commitment_root)
            .chain(proof.iopp_oracles.par_iter())
            .take(num_vars)
            .into_par_iter();

        let points = setup.iopp_challenges(num_vars, transcript);

        if !proof
            .iopp_queries
            .par_iter()
            .enumerate()
            .all(|(i, iopp_query)| iopp_query.verify(setup, points[i], oracles.clone(), &rs))
        {
            return Err(ProofVerifyError::InternalError);
        }

        Ok(())
    }

    fn batch_verify(
        batch_proof: &Self::BatchedProof,
        setup: &Self::Setup,
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        commitments: &[&Self::Commitment],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let t_len = commitments.len() - 1;
        let openings_lifted: Vec<ExtF> = openings.iter().cloned().map(Into::into).collect();

        let (t_vec, opening_point_lifted) = {
            let mut t_vec =
                transcript.challenge_vector::<ExtF>(BASEFOLD_BATCH_OPENING_CHALLENGE_TAG, t_len);
            t_vec.insert(0, <ExtF as JoltField>::one());
            let point_lifted: Vec<ExtF> =
                opening_point.iter().cloned().map(Into::into).collect_vec();
            (t_vec, point_lifted)
        };

        let sumcheck_claimed_sum = compute_dotproduct(&t_vec, &openings_lifted);

        let num_vars = opening_point.len();
        // NOTE: run sumcheck and retrieve all sumcheck challenges
        let (combined_f_eq, rs) = batch_proof.sumcheck_transcript.verify(
            sumcheck_claimed_sum,
            num_vars,
            MERGE_POLY_DEG,
            transcript,
        )?;

        let eq_tz_r = EqPolynomial::new(opening_point_lifted).evaluate(&rs);
        let combined_f_r = combined_f_eq / eq_tz_r;

        if batch_proof.iopp_last_oracle_message.len() != 1 << setup.rate_bits {
            return Err(ProofVerifyError::InternalError);
        }

        if batch_proof
            .iopp_last_oracle_message
            .iter()
            .any(|&x| x != combined_f_r)
        {
            return Err(ProofVerifyError::InternalError);
        }

        let virtual_oracles: Vec<_> = commitments.iter().map(|c| c.merkle.root()).collect();
        let points = setup.iopp_challenges(num_vars, transcript);

        if !batch_proof
            .iopp_queries
            .par_iter()
            .enumerate()
            .all(|(i, query)| {
                query.verify(
                    setup,
                    points[i],
                    virtual_oracles.par_iter(),
                    batch_proof.iopp_oracles.par_iter(),
                    &t_vec,
                    &rs,
                )
            })
        {
            return Err(ProofVerifyError::InternalError);
        }

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Jolty2 BasefoldOpeningProof"
    }
}
