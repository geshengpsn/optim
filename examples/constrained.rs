//! Equality- and inequality-constrained minimization.
//!
//! Run with `cargo run --example constrained`.

use faer::{Col, Mat, col, mat};
use optim::{
    EqConstraints, EqParams, GeneralConstraintsParams, InequalityConstraints, InitMode,
    eqc_general_solver, gc_general_solver, optimize_eq_constraints, optimize_general_constraints,
};

// min ½ (x0² + 2 x1²) + x0
fn f(x: &Col<f64>) -> f64 {
    0.5 * (x[0] * (x[0] + 1.0) + 2.0 * x[1] * x[1])
}
fn df(x: &Col<f64>) -> Col<f64> {
    col![x[0] + 1.0, 2.0 * x[1]]
}
fn d2f(_: &Col<f64>) -> Mat<f64> {
    mat![[1.0, 0.0], [0.0, 2.0]]
}

fn main() {
    // Equality only: min f  s.t.  x0 + x1 = 1.
    let eq = EqConstraints {
        a: mat![[1.0, 1.0]],
        b: col![1.0],
    };
    let params = EqParams {
        e: 1e-10,
        ..Default::default()
    };
    let r = optimize_eq_constraints(
        f,
        df,
        d2f,
        &eq,
        eqc_general_solver,
        col![0.0, 0.0],
        Some(params),
    );
    println!(
        "eq : x = [{:.6}, {:.6}]  dual = [{:.6}]",
        r.x[0], r.x[1], r.dual[0]
    );

    // Equality + inequality: add 0.8 - x0 ≤ 0 and -x1 ≤ 0.
    let h = |x: &Col<f64>| col![0.8 - x[0], -x[1]];
    let h_jac = |_: &Col<f64>| mat![[-1.0, 0.0], [0.0, -1.0]];
    let h_hess = |_: &Col<f64>, _: &Col<f64>| Mat::<f64>::zeros(2, 2);
    let ieq = InequalityConstraints {
        f: h,
        jac: h_jac,
        hess_sum: h_hess,
    };
    let g = optimize_general_constraints(
        f,
        df,
        d2f,
        &eq,
        &ieq,
        gc_general_solver,
        col![0.0, 0.0],
        InitMode::Infeasible,
        true,
        Some(GeneralConstraintsParams::default()),
    );
    println!(
        "gen: x = [{:.6}, {:.6}]  ieq_dual = [{:.6}, {:.6}]  eq_dual = [{:.6}]",
        g.x[0], g.x[1], g.ieq_dual[0], g.ieq_dual[1], g.eq_dual[0]
    );
}
