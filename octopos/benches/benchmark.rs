use ark_std::test_rng;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use goldilocks::{Field, Goldilocks};
use octopos::{
    hash::{new_octopos_hasher, Digest},
    tree::{hash_internals_vanilla, hash_leaves_vanilla, OctoposTree},
};
use std::time::Duration;

fn leaves_setup(num_leaves: usize) -> Vec<Goldilocks> {
    let mut rng = test_rng();

    (0..num_leaves)
        .map(|_| Goldilocks::random(&mut rng))
        .collect()
}

fn criterion_poseidon_leaves(c: &mut Criterion) {
    let final_single_threaded_leaves_po2 = 20;
    let final_single_threaded_leaves = 1 << final_single_threaded_leaves_po2;
    let leaves = leaves_setup(final_single_threaded_leaves);

    let mut digest_buffer = vec![Digest::default(); final_single_threaded_leaves];

    let mut group = c.benchmark_group("poseidion leaves");

    for i in 10..=final_single_threaded_leaves_po2 {
        let seq_len = 1 << i;

        group
            .bench_function(BenchmarkId::new("poseidon leaves", i), |b| {
                b.iter(|| {
                    let hasher = new_octopos_hasher();
                    hash_leaves_vanilla(&leaves[..seq_len], &mut digest_buffer[0..], &hasher);
                })
            })
            .sample_size(10)
            .measurement_time(Duration::from_secs(50));
    }
}

fn internal_setup(num_children: usize) -> Vec<Digest> {
    let mut rng = test_rng();

    (0..num_children)
        .map(|_| {
            Digest([
                Goldilocks::random(&mut rng),
                Goldilocks::random(&mut rng),
                Goldilocks::random(&mut rng),
                Goldilocks::random(&mut rng),
            ])
        })
        .collect()
}

fn criterion_poseidon_internals(c: &mut Criterion) {
    let final_single_threaded_internals_po2 = 20;
    let final_single_threaded_internals = 1 << final_single_threaded_internals_po2;
    let leaves = internal_setup(final_single_threaded_internals);

    let mut digest_buffer = vec![Digest::default(); final_single_threaded_internals];

    let mut group = c.benchmark_group("poseidion internals");

    for i in 10..=final_single_threaded_internals_po2 {
        let seq_len = 1 << i;

        group
            .bench_function(BenchmarkId::new("poseidon internals", i), |b| {
                b.iter(|| {
                    let hasher = new_octopos_hasher();
                    hash_internals_vanilla(&leaves[..seq_len], &mut digest_buffer[0..], &hasher);
                })
            })
            .sample_size(10)
            .measurement_time(Duration::from_secs(50));
    }
}

fn criterion_octopos_tree(c: &mut Criterion) {
    let final_mt_po2 = 27;
    let final_mt_size = 1 << final_mt_po2;
    let leaves = leaves_setup(final_mt_size);

    let mut group = c.benchmark_group("octopos tree");

    for i in 10..=final_mt_po2 {
        let leaves_size = 1 << i;

        group
            .bench_function(BenchmarkId::new("octopos MT", i), |b| {
                let hasher = new_octopos_hasher();

                b.iter(|| {
                    let bench_leaves = leaves[..leaves_size].to_vec();
                    let _ = black_box(OctoposTree::new_from_leaves(bench_leaves, &hasher));
                })
            })
            .sample_size(10)
            .measurement_time(Duration::from_secs(20));
    }
}

criterion_group!(
    benches,
    criterion_poseidon_leaves,
    criterion_poseidon_internals,
    criterion_octopos_tree
);
criterion_main!(benches);
