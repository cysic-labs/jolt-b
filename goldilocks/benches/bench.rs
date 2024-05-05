use criterion::{criterion_group, criterion_main, Criterion};
use ff::Field;
use goldilocks::{Goldilocks, GoldilocksExt2, GoldilocksExt3, SmallField};
use halo2curves::bn256::Fr;
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;

const SIZE: usize = 1000;

criterion_main!(bench);

criterion_group!(bench, bench_fields);

fn bench_fields(c: &mut Criterion) {
    bench_field::<Goldilocks>(c, Goldilocks::NAME);
    bench_field::<GoldilocksExt2>(c, GoldilocksExt2::NAME);
    bench_field::<GoldilocksExt3>(c, GoldilocksExt3::NAME);
    bench_field::<Fr>(c, "Bn256 scalar")
}

fn bench_field<F: Field>(c: &mut Criterion, field_name: &str) {
    let mut bench_group = c.benchmark_group(format!("{}", field_name));

    let mut rng = XorShiftRng::from_seed([
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);
    let a = (0..SIZE).map(|_| F::random(&mut rng)).collect::<Vec<_>>();
    let b = (0..SIZE).map(|_| F::random(&mut rng)).collect::<Vec<_>>();

    let bench_str = format!("{} additions", SIZE);
    bench_group.bench_function(bench_str, |bencher| {
        bencher.iter(|| {
            a.iter()
                .zip(b.iter())
                .map(|(&ai, &bi)| ai + bi)
                .collect::<Vec<_>>()
        })
    });

    let bench_str = format!("{} multiplications", SIZE);
    bench_group.bench_function(bench_str, |bencher| {
        bencher.iter(|| {
            a.iter()
                .zip(b.iter())
                .map(|(&ai, &bi)| ai * bi)
                .collect::<Vec<_>>()
        })
    });
    bench_group.finish();
}
