use super::random_field_tests;
use super::random_inversion_tests;
use super::random_prime_field_tests;
use super::random_small_field_tests;
use crate::fp::Goldilocks;
use crate::fp2::GoldilocksExt2;

#[test]
fn test_field() {
    random_field_tests::<GoldilocksExt2>("GoldilocksExt2".to_string());
    random_prime_field_tests::<GoldilocksExt2>("GoldilocksExt2".to_string());
    random_inversion_tests::<GoldilocksExt2>("GoldilocksExt2".to_string());
    random_small_field_tests::<GoldilocksExt2>("GoldilocksExt2".to_string());
}

#[test]
fn known_answer_tests() {
    let a = GoldilocksExt2([Goldilocks::from(1), Goldilocks::from(2)]);
    let b = GoldilocksExt2([Goldilocks::from(3), Goldilocks::from(4)]);
    let c = GoldilocksExt2([Goldilocks::from(59), Goldilocks::from(10)]);
    assert_eq!(a * b, c)
}
