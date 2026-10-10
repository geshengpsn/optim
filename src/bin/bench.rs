//! Benchmark runner comparing faer's single-threaded (`Par::Seq`) and
//! multi-threaded (`Par::rayon`) execution on the LP/QP/SOCP/SDP suite.
//!
//! Run with:
//! ```text
//! cargo run --release --bin bench                 # seq vs par comparison
//! cargo run --release --bin bench -- 3            # 3 repetitions
//! cargo run --release --bin bench -- 3 1 seq      # single-threaded only
//! cargo run --release --bin bench -- 3 0 par      # barrier + multi-threaded
//! ```
//! Arguments: `<reps> <predcorr(0|1)> <both|seq|par>`.

use std::time::Instant;

use faer::{Par, set_global_parallelism};
use optim::conic::{ConicParams, ConicStatus};
use optim::problems::{Instance, random_lp, random_qp, suite};

fn large_suite() -> Vec<Instance> {
    vec![
        random_lp(300, 150, 0x51),
        random_qp(300, 150, 0x52),
        random_lp(600, 300, 0x53),
        random_qp(600, 300, 0x54),
    ]
}

fn fmt_status(s: ConicStatus) -> &'static str {
    match s {
        ConicStatus::Optimal => "opt",
        ConicStatus::MaxIterations => "maxit",
        ConicStatus::Failed => "fail",
    }
}

fn bench_instance(inst: &Instance, reps: usize, predcorr: bool) -> (f64, optim::ConicSolution) {
    let params = ConicParams {
        predcorr,
        ..ConicParams::default()
    };
    let mut best = f64::INFINITY;
    let mut last = inst.solve(Some(params));
    for _ in 0..reps {
        let t0 = Instant::now();
        last = inst.solve(Some(params));
        best = best.min(t0.elapsed().as_secs_f64());
    }
    (best, last)
}

fn run(cases: &[Instance], reps: usize, predcorr: bool) -> Vec<(f64, optim::ConicSolution)> {
    cases
        .iter()
        .map(|inst| bench_instance(inst, reps, predcorr))
        .collect()
}

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let predcorr = std::env::args().nth(2).map(|s| s != "0").unwrap_or(true);
    let mode = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "both".to_string());

    let cases: Vec<Instance> = suite().into_iter().chain(large_suite()).collect();
    let nthreads = Par::rayon(0).degree();

    println!(
        "{} instances, best of {} runs, predictor-corrector = {predcorr}, rayon threads = {nthreads}",
        cases.len(),
        reps
    );

    match mode.as_str() {
        "seq" => {
            set_global_parallelism(Par::Seq);
            let seq = run(&cases, reps, predcorr);
            print_single(&cases, &seq, "single-thread");
        }
        "par" => {
            set_global_parallelism(Par::rayon(0));
            let par = run(&cases, reps, predcorr);
            print_single(&cases, &par, "multi-thread");
        }
        _ => {
            set_global_parallelism(Par::Seq);
            let seq = run(&cases, reps, predcorr);
            set_global_parallelism(Par::rayon(0));
            let par = run(&cases, reps, predcorr);
            print_comparison(&cases, &seq, &par);
        }
    }
}

fn print_single(cases: &[Instance], rows: &[(f64, optim::ConicSolution)], label: &str) {
    println!(
        "\n[{label}]\n{:<14} {:>5} {:>5} {:>6} {:>7} {:>12} {:>10}",
        "instance", "n", "p", "status", "iters", "objective", "time(ms)"
    );
    println!("{}", "-".repeat(72));
    let mut total = 0.0;
    for (inst, (secs, sol)) in cases.iter().zip(rows) {
        total += secs;
        println!(
            "{:<14} {:>5} {:>5} {:>6} {:>7} {:>12.6} {:>10.3}",
            inst.name,
            inst.program.n(),
            inst.program.p(),
            fmt_status(sol.status),
            sol.newton_iterations,
            sol.objective,
            secs * 1e3
        );
    }
    println!("{}", "-".repeat(72));
    println!("total: {:.3} ms", total * 1e3);
}

fn print_comparison(
    cases: &[Instance],
    seq: &[(f64, optim::ConicSolution)],
    par: &[(f64, optim::ConicSolution)],
) {
    println!(
        "\n{:<14} {:>5} {:>5} {:>7} {:>11} {:>11} {:>9}",
        "instance", "n", "p", "iters", "seq(ms)", "par(ms)", "speedup"
    );
    println!("{}", "-".repeat(70));
    let mut seq_total = 0.0;
    let mut par_total = 0.0;
    for ((inst, (s, sol)), (p, _)) in cases.iter().zip(seq).zip(par) {
        seq_total += s;
        par_total += p;
        println!(
            "{:<14} {:>5} {:>5} {:>7} {:>11.3} {:>11.3} {:>8.2}x",
            inst.name,
            inst.program.n(),
            inst.program.p(),
            sol.newton_iterations,
            s * 1e3,
            p * 1e3,
            s / p
        );
    }
    println!("{}", "-".repeat(70));
    println!(
        "{:<14} {:>5} {:>5} {:>7} {:>11.3} {:>11.3} {:>8.2}x",
        "TOTAL",
        "",
        "",
        "",
        seq_total * 1e3,
        par_total * 1e3,
        seq_total / par_total
    );
}
