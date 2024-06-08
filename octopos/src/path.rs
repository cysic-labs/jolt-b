use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::mem::size_of;

use crate::{
    hash::{Digest, OctoposHasherTrait, OCTOPOS_LEAF_BYTES},
    tree::leaves_adic,
};

trait OctoposTreeNode: Sized {
    fn verify<H: OctoposHasherTrait>(&self, root: &Digest, hasher: &H) -> bool;
}

#[derive(Clone, Debug, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub(crate) struct OctoposInternalNode(pub Digest, pub Digest);

#[derive(Clone, Debug, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub(crate) struct OctoposLeavesNode(pub [u8; OCTOPOS_LEAF_BYTES]);

impl OctoposTreeNode for OctoposInternalNode {
    fn verify<H: OctoposHasherTrait>(&self, root: &Digest, hasher: &H) -> bool {
        hasher.hash_internals(&self.0, &self.1) == *root
    }
}

impl OctoposTreeNode for OctoposLeavesNode {
    fn verify<H: OctoposHasherTrait>(&self, root: &Digest, hasher: &H) -> bool {
        hasher.hash_leaves(&self.0) == *root
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

    pub fn verify<F: Sized, H: OctoposHasherTrait>(&self, root: &Digest, hasher: &H) -> bool {
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
