use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::test_rng;
use goldilocks::{Field, Goldilocks, GoldilocksExt2};
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sha2::{Sha256, Sha512_256};
use std::time::Instant;

use crate::{
    hash::{OctoposHasherTrait, PoseidonHasher},
    tree::OctoposTree,
};

#[test]
fn test_octopos_correctness() {
    test_octopos_correctness_helper::<Goldilocks, Sha256>();
    test_octopos_correctness_helper::<GoldilocksExt2, Sha256>();
    test_octopos_correctness_helper::<Goldilocks, Sha512_256>();
    test_octopos_correctness_helper::<GoldilocksExt2, Sha512_256>();
    test_octopos_correctness_helper::<Goldilocks, PoseidonHasher>();
    test_octopos_correctness_helper::<GoldilocksExt2, PoseidonHasher>()
}

fn test_octopos_correctness_helper<
    F: Field + CanonicalSerialize + CanonicalDeserialize,
    H: OctoposHasherTrait + Sync,
>() {
    let mut rng = test_rng();

    let leaves_num = 1 << 20;
    let leaves: Vec<_> = (0..leaves_num).map(|_| F::random(&mut rng)).collect();

    let hasher = H::new_instance();

    let now = Instant::now();
    let tree = OctoposTree::new_from_leaves(leaves.clone(), &hasher);
    println!("total elapsed {}", now.elapsed().as_millis());

    let partitions = rayon::current_num_threads().next_power_of_two();
    let task_partition_length = leaves_num / partitions;

    (0..partitions).into_par_iter().for_each(|partition_index| {
        for i in (partition_index * task_partition_length)
            ..((partition_index + 1) * task_partition_length)
        {
            let path = tree.index_opening(i as usize);
            assert!(path.verify::<F, _>(&tree.root(), &hasher));
            assert_eq!(path.leaf_value::<F>(), leaves[i])
        }
    });
}

#[test]
fn test_octopos_batch_correctness() {
    test_octopos_batch_correctness_helper::<Goldilocks, Sha256>();
    test_octopos_batch_correctness_helper::<GoldilocksExt2, Sha256>();
}

fn test_octopos_batch_correctness_helper<
    F: Field + CanonicalSerialize + CanonicalDeserialize,
    H: OctoposHasherTrait + Sync,
>() {
    let mut rng = test_rng();

    let leaves_num = 1 << 10;
    let leaves: Vec<_> = (0..leaves_num).map(|_| F::random(&mut rng)).collect();

    let hasher = H::new_instance();

    let leavess = vec![
        leaves[..512].to_vec(),
        leaves[512..768].to_vec(),
        leaves[768..896].to_vec(),
        leaves[896..960].to_vec(),
        leaves[960..992].to_vec(),
        leaves[992..1008].to_vec(),
        leaves[1008..1016].to_vec(),
    ];

    let trees = OctoposTree::batch_tree_for_recursive_oracles(leavess.clone(), &hasher);

    let expected_trees = leavess
        .into_iter()
        .map(|l| OctoposTree::new_from_leaves(l, &hasher))
        .collect_vec();

    for (t, e_t) in trees.iter().zip(expected_trees.iter()) {
        assert_eq!(t.root(), e_t.root())
    }
}
