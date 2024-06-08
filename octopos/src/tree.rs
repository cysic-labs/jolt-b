use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use goldilocks::Goldilocks;
use rayon::prelude::*;
use std::{cmp::min, mem::size_of};

use crate::{
    hash::{Digest, OctoposHasherTrait, OCTOPOS_LEAF_BYTES, OCTOPOS_LEAF_GOLDILOCKS},
    path::{OctoposInternalNode, OctoposLeavesNode, OctoposPath},
};

#[derive(Clone, Debug, CanonicalDeserialize, CanonicalSerialize)]
pub struct OctoposTree<F: Sized + Clone + CanonicalDeserialize + CanonicalSerialize> {
    pub(crate) internals: Vec<Digest>,
    // todo: change to Arc<Vec<F>> so we can avoid cloning it everywhere?
    pub leaves: Vec<F>,
}

#[inline(always)]
pub(crate) const fn leaves_adic<F: Sized>() -> usize {
    OCTOPOS_LEAF_BYTES / size_of::<F>()
}

pub fn hash_leaves_vanilla<
    F: Sized + Clone + CanonicalSerialize + CanonicalDeserialize,
    H: OctoposHasherTrait + Sync,
>(
    leaves: &[F],
    internals: &mut [Digest],
    hasher: &H,
) {
    let chunk_size = leaves_adic::<F>();

    leaves
        .par_chunks(chunk_size)
        .zip(internals.par_iter_mut())
        .for_each(|(chunk, internal)| {
            let mut raw_bytes = [0u8; OCTOPOS_LEAF_BYTES];
            chunk.iter().enumerate().for_each(|(i, elem)| {
                elem.serialize_uncompressed(&mut raw_bytes[i * size_of::<F>()..])
                    .unwrap();
            });

            *internal = hasher.hash_leaves(&raw_bytes);
        });
}

fn multithreaded_hash_leaves<
    F: Sized + Clone + CanonicalDeserialize + CanonicalSerialize,
    H: OctoposHasherTrait + Sync,
>(
    leaves: &[F],
    internals: &mut [Digest],
    hasher: &H,
) {
    let partitions = rayon::current_num_threads().next_power_of_two();

    let task_partition_length = internals.len() / partitions;
    let leaves_partition_length = leaves.len() / partitions;

    if task_partition_length == 0 {
        hash_leaves_vanilla(leaves, internals, hasher);
        return;
    }

    let hashed: Vec<Digest> = (0..partitions)
        .into_par_iter()
        .flat_map_iter(|partition_index| {
            let left = partition_index * leaves_partition_length;
            let right = left + leaves_partition_length;

            let mut chunk = vec![Digest::default(); task_partition_length];

            let leaves_slice = &leaves[left..right];
            hash_leaves_vanilla(leaves_slice, &mut chunk, hasher);

            chunk
        })
        .collect();

    internals.copy_from_slice(hashed.as_slice());
}

pub fn hash_internals_vanilla<H: OctoposHasherTrait>(
    children: &[Digest],
    parents: &mut [Digest],
    hasher: &H,
) {
    children
        .chunks(2)
        .zip(parents.iter_mut())
        .for_each(|(children, parent)| *parent = hasher.hash_internals(&children[0], &children[1]));
}

fn multithreaded_hash_internals<H: OctoposHasherTrait + Sync>(
    children: &[Digest],
    parents: &mut [Digest],
    hasher: &H,
) {
    let partitions = rayon::current_num_threads().next_power_of_two();

    let task_partition_length = parents.len() / partitions;
    let children_partition_length = children.len() / partitions;

    if task_partition_length == 0 {
        hash_internals_vanilla(children, parents, hasher);
        return;
    }

    let hashed: Vec<Digest> = (0..partitions)
        .into_par_iter()
        .flat_map_iter(|partition_index| {
            let left = partition_index * children_partition_length;
            let right = left + children_partition_length;

            let mut chunk = vec![Digest::default(); task_partition_length];

            let children_slice = &children[left..right];
            hash_internals_vanilla(children_slice, &mut chunk, hasher);

            chunk
        })
        .collect();

    parents.copy_from_slice(hashed.as_slice());
}

impl<F: Sized + Clone + CanonicalDeserialize + CanonicalSerialize> OctoposTree<F> {
    // NOTE: we directly move the value here, as oracle should give point query result
    // together with MT path, so the leaves should appear only one copy in RAM.
    pub fn new_from_leaves<H: OctoposHasherTrait + Sync>(leaves: Vec<F>, hasher: &H) -> Self {
        // assert leaves size being a power of 2
        assert_eq!(leaves.len() & (leaves.len() - 1), 0);

        // assert leaves size being at least 8 Goldilocks
        assert!(leaves.len() * size_of::<F>() >= OCTOPOS_LEAF_GOLDILOCKS * size_of::<Goldilocks>());

        // assert size of fields should fit right in 8 Goldilocks
        assert_eq!(OCTOPOS_LEAF_BYTES % size_of::<F>(), 0);

        let mut internal_size = leaves.len() * 2 / leaves_adic::<F>() - 1;
        let mut internals: Vec<Digest> = vec![Digest::default(); internal_size];

        multithreaded_hash_leaves(&leaves, &mut internals[(internal_size >> 1)..], hasher);
        internal_size >>= 1;

        while internal_size > 0 {
            let (parent_leaves, children_leaves) =
                internals[..(1 + (internal_size << 1))].split_at_mut(internal_size);

            multithreaded_hash_internals(
                children_leaves,
                &mut parent_leaves[(internal_size >> 1)..],
                hasher,
            );
            internal_size >>= 1;
        }

        Self { internals, leaves }
    }

    #[inline]
    pub fn root(&self) -> Digest {
        self.internals[0]
    }

    pub fn index_opening(&self, index: usize) -> OctoposPath {
        let mut index_bits = self.internals.len().ilog2() as usize;

        let leaves_srt = index / leaves_adic::<F>() * leaves_adic::<F>();
        let mut leaves_chunk = [0u8; OCTOPOS_LEAF_BYTES];
        self.leaves[leaves_srt..leaves_srt + leaves_adic::<F>()]
            .iter()
            .enumerate()
            .for_each(|(i, elem)| {
                elem.serialize_compressed(&mut leaves_chunk[i * size_of::<F>()..])
                    .unwrap();
            });

        if index_bits == 0 {
            return OctoposPath {
                index,
                leaves: OctoposLeavesNode(leaves_chunk),
                internals: Vec::new(),
            };
        }

        let mut internals: Vec<OctoposInternalNode> = Vec::with_capacity(index_bits - 1);

        let mut index_start_from = 1;

        while index_start_from < self.internals.len() {
            let level_index = (index / leaves_adic::<F>()) >> (index_bits - 1);
            let sibling = level_index ^ 1;

            let left = index_start_from + min(level_index, sibling);
            internals.push(OctoposInternalNode(
                self.internals[left],
                self.internals[left + 1],
            ));

            index_start_from = (index_start_from << 1) + 1;
            index_bits -= 1
        }
        OctoposPath {
            index,
            leaves: OctoposLeavesNode(leaves_chunk),
            internals,
        }
    }
}

pub trait AbstractOracle {
    type QueryResult;

    fn index_query(&self, index: usize) -> Self::QueryResult;

    fn size(&self) -> usize;
}

impl<F: Sized + Clone + CanonicalDeserialize + CanonicalSerialize> AbstractOracle
    for OctoposTree<F>
{
    type QueryResult = OctoposPath;

    #[inline]
    fn index_query(&self, index: usize) -> Self::QueryResult {
        self.index_opening(index)
    }

    #[inline]
    fn size(&self) -> usize {
        self.leaves.len()
    }
}
