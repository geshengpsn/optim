//! Criterion benchmarks for the LP / QP / SOCP / SDP problem suite, comparing
//! faer's single-threaded (`Par::Seq`) and multi-threaded (`Par::rayon`) modes.
//!
//! Run with `cargo bench`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::{Par, set_global_parallelism};
use optim::conic::ConicParams;
use optim::problems::suite;

fn bench_suite(c: &mut Criterion) {
    let cases = suite();
    for (label, par) in [("seq", Par::Seq), ("par", Par::rayon(0))] {
        set_global_parallelism(par);
        let mut group = c.benchmark_group(format!("conic_{label}"));
        group.sample_size(20);
        for inst in &cases {
            group.bench_with_input(BenchmarkId::from_parameter(&inst.name), inst, |b, inst| {
                b.iter(|| inst.solve(Some(ConicParams::default())));
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_suite);
criterion_main!(benches);
