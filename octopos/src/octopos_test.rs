use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::test_rng;
use goldilocks::{Field, Goldilocks, GoldilocksExt2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::time::Instant;

use crate::{hash::new_octopos_hasher, tree::OctoposTree};

#[test]
fn test_octopos_correctness() {
    test_octopos_correctness_helper::<Goldilocks>();
    test_octopos_correctness_helper::<GoldilocksExt2>()
}

fn test_octopos_correctness_helper<F: Field + CanonicalSerialize + CanonicalDeserialize>() {
    let mut rng = test_rng();

    let leaves_num = 1 << 20;
    let leaves: Vec<_> = (0..leaves_num).map(|_| F::random(&mut rng)).collect();

    let hasher = new_octopos_hasher();

    let now = Instant::now();
    let tree = OctoposTree::new_from_leaves(leaves, &hasher);
    println!("total elapsed {}", now.elapsed().as_millis());

    let partitions = rayon::current_num_threads().next_power_of_two();
    let task_partition_length = leaves_num / partitions;

    (0..partitions).into_par_iter().for_each(|partition_index| {
        for i in (partition_index * task_partition_length)
            ..((partition_index + 1) * task_partition_length)
        {
            let path = tree.index_opening(i as usize);
            assert!(path.verify::<F>(&tree.root(), &hasher))
        }
    });
}
