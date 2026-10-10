//! Unconstrained minimization with `optimize_no_constraints`.
//!
//! Run with `cargo run --example unconstrained`.

use faer::{Col, Mat, col, mat};
use optim::{noc_general_solver, optimize_no_constraints};

fn main() {
    // min (x0 - 1)² + (x1 - 2)²  →  (1, 2).
    let f = |x: &Col<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
    let grad = |x: &Col<f64>| col![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)];
    let hess = |_: &Col<f64>| mat![[2.0, 0.0], [0.0, 2.0]];

    let x = optimize_no_constraints(f, grad, hess, noc_general_solver, col![0.0, 0.0], 1e-10);
    println!("minimum at x = [{:.8}, {:.8}]", x[0], x[1]);
}

// The default direction solver is `noc_general_solver`, which solves
// `H * dir = -grad` with faer's pivoted LU. You can pass your own closure with
// the signature `Fn(&Mat<f64>, &Col<f64>) -> Col<f64>` instead.
#[allow(dead_code)]
fn _signature_reference(_: fn(&Mat<f64>, &Col<f64>) -> Col<f64>) {}
