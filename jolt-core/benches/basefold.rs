use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use goldilocks::{Field, Goldilocks, GoldilocksExt2};
use jolt_core::{
    poly::{
        commitment::{
            basefold::BasefoldCommitmentScheme,
            commitment_scheme::{BatchType, CommitShape, CommitmentScheme},
        },
        dense_mlpoly::DensePolynomial,
    },
    utils::transcript::ProofTranscript,
};
use sha2::Sha256;
use std::{hint::black_box, time::Duration};

fn basefold_poly_setup(num_vars: usize) -> Vec<Goldilocks> {
    let mut rng = test_rng();
    let num_evals = 1 << num_vars;

    (0..num_evals)
        .map(|_| Goldilocks::random(&mut rng))
        .collect()
}

fn criterion_basefold_goldilocks(c: &mut Criterion) {
    let largest_num_vars = 20;
    let poly_evals = basefold_poly_setup(largest_num_vars);

    let mut group = c.benchmark_group("basefold single commit and prove");
    for i in 10..=largest_num_vars {
        let seq_len = 1 << i;
        let pp = BasefoldCommitmentScheme::<Goldilocks, GoldilocksExt2, Sha256>::setup(&[
            CommitShape::new(seq_len, BatchType::Big),
        ]);

        group
            .bench_function(BenchmarkId::new("basefold commit", i), |b| {
                let evals_seg = poly_evals[..seq_len].to_vec();
                let dense_poly = DensePolynomial::new(evals_seg);
                b.iter(|| {
                    let _ = black_box(BasefoldCommitmentScheme::commit(&dense_poly, &pp));
                })
            })
            .sample_size(10)
            .measurement_time(Duration::from_secs(50));
    }

    let mut rng = test_rng();
    for i in 10..=largest_num_vars {
        let seq_len = 1 << i;
        let pp = BasefoldCommitmentScheme::<Goldilocks, GoldilocksExt2, Sha256>::setup(&[
            CommitShape::new(seq_len, BatchType::Big),
        ]);

        group
            .bench_function(BenchmarkId::new("basefold prove", i), |b| {
                let evals_seg = poly_evals[..seq_len].to_vec();
                let dense_poly = DensePolynomial::new(evals_seg);
                let opening_point: Vec<_> = (0..i).map(|_| Goldilocks::random(&mut rng)).collect();
                let basefold_commit = BasefoldCommitmentScheme::commit(&dense_poly, &pp);
                let mut proof_transcript = ProofTranscript::new(b"basefold benchmark");

                b.iter(|| {
                    let _ = black_box(BasefoldCommitmentScheme::prove(
                        &dense_poly,
                        &pp,
                        &basefold_commit,
                        &opening_point,
                        &mut proof_transcript,
                    ));
                })
            })
            .sample_size(10)
            .measurement_time(Duration::from_secs(50));
    }
}

criterion_group!(benches, criterion_basefold_goldilocks);
criterion_main!(benches);
