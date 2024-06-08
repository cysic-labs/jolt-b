use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::test_rng;
use goldilocks::{Field, Goldilocks, GoldilocksExt2};
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
