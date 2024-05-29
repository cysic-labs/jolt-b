use ark_std::test_rng;
use goldilocks::Goldilocks;
use rand_core::RngCore;
use std::time::Instant;

use crate::{
    poly::{
        commitment::{
            basefold::{BasefoldCommitmentScheme, BasefoldPP, BASEFOLD_ADDITIONAL_RATE_BITS},
            commitment_scheme::{BatchType, CommitmentScheme},
        },
        dense_mlpoly::DensePolynomial,
        field::JoltField,
    },
    utils::transcript::ProofTranscript,
};

fn test_basefold_helper<F: JoltField>(num_vars: usize, rng: &mut impl RngCore) {
    let pp = BasefoldPP::<Goldilocks>::new(BASEFOLD_ADDITIONAL_RATE_BITS);

    let poly_evals = (0..(1 << num_vars))
        .map(|_| Goldilocks::random(rng))
        .collect();
    let poly = DensePolynomial::new(poly_evals);

    let opening_point: Vec<_> = (0..num_vars).map(|_| Goldilocks::random(rng)).collect();
    let eval = poly.evaluate(&opening_point);

    let now = Instant::now();
    let commitment = BasefoldCommitmentScheme::commit(&poly, &pp);
    println!("committing elapsed {}", now.elapsed().as_millis());

    let mut prover_transcript = ProofTranscript::new(b"example");
    let mut verifier_transcript = ProofTranscript::new(b"example");

    let now = Instant::now();
    let eval_proof = BasefoldCommitmentScheme::prove(
        &poly,
        &pp,
        &commitment,
        &opening_point,
        &mut prover_transcript,
    );
    println!("proving elapsed {}", now.elapsed().as_millis());

    let now = Instant::now();
    BasefoldCommitmentScheme::verify(
        &eval_proof,
        &pp,
        &mut verifier_transcript,
        &opening_point,
        &eval,
        &commitment,
    )
    .unwrap();
    println!("verifying elapsed {}", now.elapsed().as_millis());
}

#[test]
fn test_basefold_vanilla() {
    let mut rng = test_rng();

    for i in 5..=18 {
        for _ in 0..10 {
            test_basefold_helper::<Goldilocks>(i, &mut rng);
        }
    }
}

fn test_basefold_batch_helper<F: JoltField>(
    num_vars: usize,
    batch_size: usize,
    rng: &mut impl RngCore,
) {
    let pp = BasefoldPP::<Goldilocks>::new(BASEFOLD_ADDITIONAL_RATE_BITS);

    let poly_evals: Vec<Vec<_>> = (0..batch_size)
        .map(|_| {
            (0..(1 << num_vars))
                .map(|_| Goldilocks::random(rng))
                .collect()
        })
        .collect();

    let polys: Vec<_> = (0..batch_size)
        .map(|i| DensePolynomial::new(poly_evals[i].clone()))
        .collect();

    let opening_point: Vec<_> = (0..num_vars).map(|_| Goldilocks::random(rng)).collect();

    let evals: Vec<_> = polys
        .iter()
        .map(|poly| poly.evaluate(&opening_point))
        .collect();

    let now = Instant::now();
    let commitments = BasefoldCommitmentScheme::batch_commit_polys(&polys, &pp, BatchType::Big);
    println!("committing elapsed {}", now.elapsed().as_millis());

    let mut prover_transcript = ProofTranscript::new(b"example");
    let mut verifier_transcript = ProofTranscript::new(b"example");

    let now = Instant::now();
    let batch_proof = BasefoldCommitmentScheme::batch_prove(
        &polys.iter().collect::<Vec<_>>(),
        &pp,
        &commitments.iter().collect::<Vec<_>>(),
        &opening_point,
        &evals,
        BatchType::Big,
        &mut prover_transcript,
    );
    println!("proving elapsed {}", now.elapsed().as_millis());

    let now = Instant::now();
    BasefoldCommitmentScheme::batch_verify(
        &batch_proof,
        &pp,
        &opening_point,
        &evals,
        &commitments.iter().collect::<Vec<_>>(),
        &mut verifier_transcript,
    )
    .unwrap();
    println!("verifying elapsed {}", now.elapsed().as_millis());
}

#[test]
fn test_basefold_batch() {
    let mut rng = test_rng();

    for num_vars in 5..=18 {
        for batch_size in 1..=10 {
            test_basefold_batch_helper::<Goldilocks>(num_vars, batch_size, &mut rng);
        }
    }
}
