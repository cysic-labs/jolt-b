use super::random_field_tests;
use super::random_prime_field_tests;
use super::random_small_field_tests;
use crate::fp::Babybear;
use crate::fp3::BabybearExt3;

#[test]
fn test_field() {
    random_field_tests::<BabybearExt3>("BabybearExt3".to_string());
    random_prime_field_tests::<BabybearExt3>("BabybearExt3".to_string());
    random_small_field_tests::<BabybearExt3>("BabybearExt3".to_string());
}

#[test]
fn known_answer_tests() {
    let a = BabybearExt3([Babybear::from(1), Babybear::from(2), Babybear::from(3)]);
    let b = BabybearExt3([Babybear::from(4), Babybear::from(5), Babybear::from(6)]);
    let c = BabybearExt3([-Babybear::from(50), -Babybear::from(23), Babybear::from(28)]);
    assert_eq!(a * b, c)
}
