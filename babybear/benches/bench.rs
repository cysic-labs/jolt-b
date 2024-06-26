use babybear::{Babybear, BabybearExt3};
use criterion::{criterion_group, criterion_main, Criterion};
use ff::Field;
use goldilocks::SmallField;
use halo2curves::bn256::Fr;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;

const SIZE: usize = 1000;

criterion_main!(bench);

criterion_group!(bench, bench_fields);

fn bench_fields(c: &mut Criterion) {
    bench_dft::<Babybear>(c, Babybear::NAME);
    bench_field::<Babybear>(c, Babybear::NAME);
    bench_field::<BabybearExt3>(c, BabybearExt3::NAME);
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
}

fn bench_dft<F: Field + TwoAdicField>(c: &mut Criterion, field_name: &str) {
    let mut bench_group = c.benchmark_group(format!("{}", field_name));

    let mut rng = XorShiftRng::from_seed([
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);
    for size in 10..20 {
        let coeffs = (0..(1 << size))
            .map(|_| F::random(&mut rng))
            .collect::<Vec<_>>();

        let bench_str = format!("dim {} dft", 1 << size);
        bench_group.bench_function(bench_str, |bencher| {
            bencher.iter(|| p3_dft::Radix2DitParallel.dft(coeffs.clone()))
        });
    }

    bench_group.finish();
}