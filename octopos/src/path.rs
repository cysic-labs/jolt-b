use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use goldilocks::Goldilocks;
use std::mem::{size_of, transmute};

use crate::{
    hash::{
        Digest, OctoposHasher, OCTOPOS_HASH_OUTPUT_WIDTH, OCTOPOS_LEAF_BYTES,
        OCTOPOS_LEAF_GOLDILOCKS,
    },
    tree::leaves_adic,
};

trait OctoposTreeNode: Sized {
    fn verify(&self, root: &Digest, hasher: &OctoposHasher) -> bool;
}

#[derive(Clone, Debug, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub(crate) struct OctoposInternalNode(pub Digest, pub Digest);

#[derive(Clone, Debug, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub(crate) struct OctoposLeavesNode(pub [u8; OCTOPOS_LEAF_BYTES]);

pub(crate) fn hash_leaves(leaves: &[u8; OCTOPOS_LEAF_BYTES], hasher: &OctoposHasher) -> Digest {
    let mut hasher = hasher.clone();

    unsafe {
        let leaves_cast =
            transmute::<&[u8; OCTOPOS_LEAF_BYTES], &[Goldilocks; OCTOPOS_LEAF_GOLDILOCKS]>(leaves);

        hasher.update(leaves_cast);
        let res = hasher.squeeze_vec()[..OCTOPOS_HASH_OUTPUT_WIDTH]
            .try_into()
            .unwrap();

        Digest(res)
    }
}

pub(crate) fn hash_internals(left: &Digest, right: &Digest, hasher: &OctoposHasher) -> Digest {
    let mut hasher = hasher.clone();
    hasher.update(left.0.as_slice());
    hasher.update(right.0.as_slice());

    let res = hasher.squeeze_vec()[..OCTOPOS_HASH_OUTPUT_WIDTH]
        .try_into()
        .unwrap();
    Digest(res)
}

impl OctoposTreeNode for OctoposInternalNode {
    fn verify(&self, root: &Digest, hasher: &OctoposHasher) -> bool {
        hash_internals(&self.0, &self.1, hasher) == *root
    }
}

impl OctoposTreeNode for OctoposLeavesNode {
    fn verify(&self, root: &Digest, hasher: &OctoposHasher) -> bool {
        hash_leaves(&self.0, hasher) == *root
    }
}

#[derive(Clone, Debug, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub struct OctoposPath {
    pub(crate) internals: Vec<OctoposInternalNode>,
    pub(crate) leaves: OctoposLeavesNode,
    pub(crate) index: usize,
}

impl OctoposPath {
    pub fn leaf_value<F: Sized + CanonicalDeserialize + Clone>(&self) -> F {
        F::deserialize_compressed(
            &self.leaves.0[self.index * size_of::<F>() % OCTOPOS_LEAF_BYTES..],
        )
        .unwrap()
    }

    pub fn leaves_len<F: Sized>(&self) -> usize {
        (1 << self.internals.len()) * leaves_adic::<F>()
    }

    pub fn verify<F: Sized>(&self, root: &Digest, hasher: &OctoposHasher) -> bool {
        assert!(self.index < self.leaves_len::<F>());

        let mut current_root = root;
        let mut sub_tree_leaves = self.leaves_len::<F>();

        let internal_check = self.internals.iter().all(|node| {
            let res = node.verify(current_root, hasher);

            current_root = match self.index & (sub_tree_leaves >> 1) {
                0 => &node.0,
                _ => &node.1,
            };

            sub_tree_leaves >>= 1;
            res
        });

        if !internal_check {
            return false;
        }

        self.leaves.verify(current_root, hasher)
    }
}
