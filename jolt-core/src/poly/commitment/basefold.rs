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
use std::{cmp, iter, marker::PhantomData};

use crate::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, field::JoltField},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{
        errors::ProofVerifyError,
        math::Math,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};

use super::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};

#[derive(Clone)]
pub struct BasefoldCommitmentScheme<F: JoltField + TwoAdicField, H: OctoposHasherTrait + Sync> {
    _marker_f: PhantomData<F>,
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
    fn verify_iopp_query<'a, F: JoltField + TwoAdicField, H: OctoposHasherTrait + Sync + Send>(
        iopp_round_query: &[BasefoldIOPPQuerySingleRound],
        setup: &BasefoldPP<F, H>,
        challenge_point: usize,
        oracles: impl IndexedParallelIterator<Item = &'a OctoposDigest>,
        folding_rs: &[F],
    ) -> bool {
        let hasher = H::new_instance();
        let num_vars = folding_rs.len();

        let mt_verify = iopp_round_query
            .par_iter()
            .zip(oracles)
            .all(|(round_i_query, root_i)| {
                let left = round_i_query.left.verify::<F, _>(root_i, &hasher);
                let right = round_i_query.right.verify::<F, _>(root_i, &hasher);

                left && right
            });

        if !mt_verify {
            return false;
        }

        let mut point = challenge_point;
        iopp_round_query.iter().tuple_windows().enumerate().all(
            |(round_i, (current_query, next_query))| {
                let oracle_rhs_start = 1 << (setup.codeword_bits(num_vars) - round_i - 1);
                let sibling_point = point ^ oracle_rhs_start;
                let left_index = cmp::min(point, sibling_point);

                let g1 = setup.t_term(num_vars, round_i, left_index);
                let g2 = -g1;

                let c1 = current_query.left.leaf_value::<F>();
                let c2 = current_query.right.leaf_value::<F>();

                // interpolate y = b + kx form
                let k = (c2 - c1) / (g2 - g1);
                let b = c1 - k * g1;
                let expected_codeword = b + k * folding_rs[round_i];

                point = left_index;

                next_query.check_expected_codeword(left_index, oracle_rhs_start, &expected_codeword)
            },
        )
    }

    pub fn verify<'a, F: JoltField + TwoAdicField, H: OctoposHasherTrait + Send + Sync>(
        &self,
        setup: &BasefoldPP<F, H>,
        challenge_point: usize,
        oracles: impl IndexedParallelIterator<Item = &'a OctoposDigest>,
        folding_rs: &[F],
    ) -> bool {
        Self::verify_iopp_query(
            &self.iopp_round_query,
            setup,
            challenge_point,
            oracles,
            folding_rs,
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

    pub fn verify<'a, F: JoltField + TwoAdicField, H: OctoposHasherTrait + Send + Sync>(
        &self,
        setup: &BasefoldPP<F, H>,
        challenge_point: usize,
        virtual_oracles: impl IndexedParallelIterator<Item = &'a OctoposDigest>,
        query_oracles: impl IndexedParallelIterator<Item = &'a OctoposDigest>,
        folding_rs: &[F],
    ) -> bool {
        if virtual_oracles.len() == 1 {
            // directly clone for there is only one query and one oracle
            let vanilla_iopp_query = self.deteriorate_to_basefold_iopp_query();
            let vanilla_oracles = virtual_oracles.chain(query_oracles);
            return vanilla_iopp_query.verify(setup, challenge_point, vanilla_oracles, folding_rs);
        }

        let hasher = H::new_instance();
        let num_vars = query_oracles.len();
        let leading_random_vars = folding_rs.len() - num_vars;

        let virtual_mt_verify =
            self.virtual_queries
                .par_iter()
                .zip(virtual_oracles)
                .all(|(query, root)| {
                    let left = query.left.verify::<F, _>(&root, &hasher);
                    let right = query.right.verify::<F, _>(&root, &hasher);
                    left && right
                });

        if !virtual_mt_verify {
            return false;
        }

        let oracle_rhs_start = 1 << (setup.codeword_bits(num_vars) - 1);
        let sibling_point = challenge_point ^ oracle_rhs_start;
        let left_index = cmp::min(challenge_point, sibling_point);

        let g1 = setup.t_term(num_vars, 0, left_index);
        let g2 = -g1;

        let (c1s, c2s): (Vec<_>, Vec<_>) = self
            .virtual_queries
            .iter()
            .map(|q| (q.left.leaf_value::<F>(), q.right.leaf_value::<F>()))
            .unzip();

        let c1 = DensePolynomial::new_padded(c1s).evaluate(&folding_rs[..leading_random_vars]);
        let c2 = DensePolynomial::new_padded(c2s).evaluate(&folding_rs[..leading_random_vars]);

        let k = (c2 - c1) / (g2 - g1);
        let b = c1 - k * g1;
        let expected_codeword = b + k * folding_rs[leading_random_vars];

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
            &folding_rs[leading_random_vars + 1..],
        )
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct BasefoldProof<F: JoltField + TwoAdicField> {
    sumcheck_transcript: SumcheckInstanceProof<F>,
    iopp_oracles: Vec<OctoposDigest>,
    iopp_last_oracle_message: Vec<F>,
    iopp_queries: Vec<BasefoldIOPPQuery>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct BasefoldBatchedProof<F: JoltField + TwoAdicField> {
    sumcheck_transcript: SumcheckInstanceProof<F>,
    iopp_oracles: Vec<OctoposDigest>,
    iopp_last_oracle_message: Vec<F>,
    iopp_queries: Vec<BasefoldVirtualIOPPQuery>,
}

pub const BASEFOLD_ADDITIONAL_RATE_BITS: usize = 3;

pub const SECURITY_BITS: usize = 100;

pub const MERGE_POLY_DEG: usize = 2;

pub const BASEFOLD_IOPP_CHALLENGE_TAG: &[u8] = b"basefold IOPP query challenge";

pub const BASEFOLD_BATCH_OPENING_CHALLENGE_TAG: &[u8] = b"basefold batch opening challenge";

#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct BasefoldPP<F: JoltField + TwoAdicField, H: OctoposHasherTrait + Sync + Send> {
    pub rate_bits: usize,
    pub verifier_queries: usize,
    _marker_f: PhantomData<F>,
    _marker_h: PhantomData<H>,
}

impl<F: JoltField + TwoAdicField, H: OctoposHasherTrait + Sync + Send> BasefoldPP<F, H> {
    pub fn new(rate_bits: usize) -> Self {
        // TODO: compute soundness - for verifier_queries
        let verifier_queries = 30;

        Self {
            rate_bits,
            verifier_queries,
            _marker_f: PhantomData,
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
    pub fn t_term(&self, num_vars: usize, round: usize, index: usize) -> F {
        let round_gen = F::two_adic_generator(self.codeword_bits(num_vars) - round);
        round_gen.exp_u64(index as u64)
    }

    pub fn reed_solomon_from_coeffs(&self, mut coeffs: Vec<F>) -> Vec<F> {
        plonky2_util::reverse_index_bits_in_place(&mut coeffs);
        let extended_length = coeffs.len() << self.rate_bits;
        coeffs.resize(extended_length, <F as JoltField>::zero());
        p3_dft::Radix2DitParallel.dft(coeffs)
    }

    /// Performs dft in batch. returns a vector that is concatenated from all the dft results.
    fn batch_reed_solomon_from_coeff_vecs(&self, mut coeff_vecs: Vec<Vec<F>>) -> Vec<Vec<F>> {
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
        let mut buf = vec![<F as JoltField>::zero(); coeff_vecs.len() * extended_length];
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

    pub fn basefold_oracle_from_evals(&self, evals: &[F]) -> OctoposTree<F> {
        let timer = start_timer!(|| "interpolate over hypercube");
        let coeffs = DensePolynomial::interpolate_over_hypercube_impl(evals);
        end_timer!(timer);

        let timer = start_timer!(|| "reed solomon from coeffs");
        let codeword = self.reed_solomon_from_coeffs(coeffs);
        end_timer!(timer);

        let timer = start_timer!(|| "new from leaves");
        let hasher = H::new_instance();
        let tree = OctoposTree::new_from_leaves(codeword, &hasher);
        end_timer!(timer);
        tree
    }
}

impl<F: JoltField + TwoAdicField, H: OctoposHasherTrait + Sync + Send + Clone + 'static>
    CommitmentScheme for BasefoldCommitmentScheme<F, H>
{
    type Field = F;
    type Setup = BasefoldPP<F, H>;
    type Commitment = BasefoldCommitment<F>;
    type Proof = BasefoldProof<F>;
    type BatchedProof = BasefoldBatchedProof<F>;

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
        let shift_z = EqPolynomial::new(opening_point.to_vec()).to_dense();
        let mut sumcheck_poly_vec = vec![poly.clone(), shift_z];
        let merge_function = |x: &[F]| x.iter().product::<F>();

        let num_vars = poly.get_num_vars();

        let mut sumcheck_polys: Vec<_> = Vec::with_capacity(num_vars);
        let mut iopp_oracles: Vec<OctoposTree<F>> = Vec::with_capacity(num_vars);

        (0..num_vars).for_each(|_| {
            // NOTE: sumcheck a single step, r_i start from x_0 towards x_n
            let (sc_univariate_poly_i, _, _) = SumcheckInstanceProof::prove_arbitrary(
                &<F as JoltField>::zero(),
                1,
                &mut sumcheck_poly_vec,
                merge_function,
                MERGE_POLY_DEG,
                transcript,
            );
            sumcheck_polys.push(sc_univariate_poly_i.compressed_polys[0].clone());
            drop(sc_univariate_poly_i);

            let oracle_i = setup.basefold_oracle_from_poly(&sumcheck_poly_vec[0]);
            iopp_oracles.push(oracle_i);
        });

        let iopp_last_oracle_message = iopp_oracles[iopp_oracles.len() - 1].leaves.clone();
        let iopp_challenges = setup.iopp_challenges(num_vars, transcript);

        let iopp_queries = (0..setup.verifier_queries)
            .zip(iopp_challenges)
            .map(|(_, mut point)| {
                let iopp_round_query = iter::once(&commitment.merkle)
                    .chain(iopp_oracles.iter())
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
        let t_len: usize = polynomials.len().next_power_of_two().log_2();

        // NOTE: compose equality polynomial of joining opening point and t_vec
        let num_vars = opening_point.len();

        let joined_point = {
            let mut t_vec =
                transcript.challenge_vector::<F>(BASEFOLD_BATCH_OPENING_CHALLENGE_TAG, t_len);
            t_vec.extend(opening_point);
            t_vec
        };

        let joined_num_vars = num_vars + t_len;
        let timer2 = start_timer!(|| "compose eq polynomial");
        let eq_z_t = EqPolynomial::new(joined_point).to_dense();
        end_timer!(timer2);

        // NOTE: batch polynomials into one, and group them for sumcheck inputs
        let timer2 = start_timer!(|| "compose lde polynomials");
        let lde_polynomials = DensePolynomial::low_degree_extension(polynomials);
        end_timer!(timer2);

        let mut sumcheck_poly_vec = vec![lde_polynomials, eq_z_t];
        let merge_function = |x: &[F]| x.iter().product::<F>();

        // NOTE: declare sumcheck related variables
        let mut sumcheck_polys: Vec<_> = Vec::with_capacity(joined_num_vars);
        let mut iopp_oracles: Vec<OctoposTree<F>> = Vec::with_capacity(num_vars);
        let virtual_oracle = BasefoldVirtualOracle::new(commitments);

        (0..joined_num_vars).for_each(|ith_folded_var| {
            let (sc_univariate_poly_i, _, _) = SumcheckInstanceProof::prove_arbitrary(
                &<F as JoltField>::zero(),
                1,
                &mut sumcheck_poly_vec,
                merge_function,
                MERGE_POLY_DEG,
                transcript,
            );
            sumcheck_polys.push(sc_univariate_poly_i.compressed_polys[0].clone());

            if ith_folded_var >= t_len {
                let oracle_i = setup.basefold_oracle_from_poly(&sumcheck_poly_vec[0]);
                iopp_oracles.push(oracle_i);
            }
        });

        let iopp_last_oracle_message = iopp_oracles[iopp_oracles.len() - 1].leaves.clone();
        let iopp_challenges = setup.iopp_challenges(num_vars, transcript);

        let iopp_queries = (0..setup.verifier_queries)
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

        // NOTE: check sumcheck statement:
        // f(z) = \sum_{r \in {0, 1}^n} (f(r) \eq(r, z)) can be reduced to
        // f_r_eq_zr = f(rs) \eq(rs, z)
        let (f_r_eq_zr, rs) =
            proof
                .sumcheck_transcript
                .verify(*opening, num_vars, MERGE_POLY_DEG, transcript)?;

        let eq_zr = EqPolynomial::new(opening_point.to_vec()).evaluate(&rs);
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
        let t_len = commitments.len().next_power_of_two().log_2();
        let t_vec = transcript.challenge_vector::<F>(BASEFOLD_BATCH_OPENING_CHALLENGE_TAG, t_len);
        let sumcheck_claimed_sum = DensePolynomial::new_padded(openings.to_vec()).evaluate(&t_vec);

        let num_vars = opening_point.len();
        let joined_point = [t_vec.clone(), opening_point.to_vec()].concat();
        let joined_num_vars = num_vars + t_len;

        // NOTE: run sumcheck and retrieve all sumcheck challenges
        let (combined_f_eq, rs) = batch_proof.sumcheck_transcript.verify(
            sumcheck_claimed_sum,
            joined_num_vars,
            MERGE_POLY_DEG,
            transcript,
        )?;

        let eq_tz_r = EqPolynomial::new(joined_point).evaluate(&rs);
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
