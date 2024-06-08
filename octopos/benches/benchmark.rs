use ark_std::test_rng;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use goldilocks::{Field, Goldilocks};
use octopos::{
    hash::{Digest, OctoposHasherTrait, PoseidonHasher, OCTOPOS_OUTPUT_BYTES},
    tree::{hash_internals_vanilla, hash_leaves_vanilla, OctoposTree},
};
use rand_core::RngCore;
use sha2::{Sha256, Sha512_256};
use std::time::Duration;

fn leaves_setup(num_leaves: usize) -> Vec<Goldilocks> {
    let mut rng = test_rng();

    (0..num_leaves)
        .map(|_| Goldilocks::random(&mut rng))
        .collect()
}

fn criterion_octopos_leaves<H: OctoposHasherTrait + Sync>(c: &mut Criterion) {
    let final_single_threaded_leaves_po2 = 20;
    let final_single_threaded_leaves = 1 << final_single_threaded_leaves_po2;
    let leaves = leaves_setup(final_single_threaded_leaves);

    let mut digest_buffer = vec![Digest::default(); final_single_threaded_leaves];

    let hasher = H::new_instance();
    let mut group = c.benchmark_group(format!("{} leaves", hasher.name()));

    for i in 10..=final_single_threaded_leaves_po2 {
        let seq_len = 1 << i;

        group
            .bench_function(
                BenchmarkId::new(format!("{} leaves", hasher.name()), i),
                |b| {
                    b.iter(|| {
                        hash_leaves_vanilla(&leaves[..seq_len], &mut digest_buffer[..], &hasher);
                    })
                },
            )
            .sample_size(10)
            .measurement_time(Duration::from_secs(50));
    }
}

fn internal_setup(num_children: usize) -> Vec<Digest> {
    let mut rng = test_rng();

    (0..num_children)
        .map(|_| {
            let mut data = [0u8; OCTOPOS_OUTPUT_BYTES];
            rng.fill_bytes(&mut data);
            Digest(data)
        })
        .collect()
}

fn criterion_octopos_internals<H: OctoposHasherTrait + Sync>(c: &mut Criterion) {
    let final_single_threaded_internals_po2 = 20;
    let final_single_threaded_internals = 1 << final_single_threaded_internals_po2;
    let leaves = internal_setup(final_single_threaded_internals);

    let mut digest_buffer = vec![Digest::default(); final_single_threaded_internals];

    let hasher = H::new_instance();
    let mut group = c.benchmark_group(format!("{} internals", hasher.name()));

    for i in 10..=final_single_threaded_internals_po2 {
        let seq_len = 1 << i;

        group
            .bench_function(
                BenchmarkId::new(format!("{} internals", hasher.name()), i),
                |b| {
                    b.iter(|| {
                        hash_internals_vanilla(&leaves[..seq_len], &mut digest_buffer[..], &hasher);
                    })
                },
            )
            .sample_size(10)
            .measurement_time(Duration::from_secs(50));
    }
}

fn criterion_octopos_tree<H: OctoposHasherTrait + Sync>(c: &mut Criterion) {
    let final_mt_po2 = 27;
    let final_mt_size = 1 << final_mt_po2;
    let leaves = leaves_setup(final_mt_size);

    let mut group = c.benchmark_group("octopos tree");
    let hasher = H::new_instance();

    for i in 10..=final_mt_po2 {
        let leaves_size = 1 << i;

        group
            .bench_function(
                BenchmarkId::new(format!("octopos {} MT", hasher.name()), i),
                |b| {
                    b.iter(|| {
                        let bench_leaves = leaves[..leaves_size].to_vec();
                        let _ = black_box(OctoposTree::new_from_leaves(bench_leaves, &hasher));
                    })
                },
            )
            .sample_size(10)
            .measurement_time(Duration::from_secs(20));
    }
}

criterion_group!(
    benches,
    criterion_octopos_leaves<PoseidonHasher>,
    criterion_octopos_leaves<Sha256>,
    criterion_octopos_leaves<Sha512_256>,
    criterion_octopos_internals<PoseidonHasher>,
    criterion_octopos_internals<Sha256>,
    criterion_octopos_internals<Sha512_256>,
    criterion_octopos_tree<PoseidonHasher>,
    criterion_octopos_tree<Sha256>,
    criterion_octopos_tree<Sha512_256>,
);
criterion_main!(benches);
