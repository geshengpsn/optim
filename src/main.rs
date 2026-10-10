//! Demonstration of the `optim` crate: runs each algorithm on the same
//! problems as the original Zig test-suite.

use faer::{Col, Mat, col, mat};
use optim::{
    EqConstraints, EqParams, GeneralConstraintsParams, InequalityConstraints, InitMode, RootParams,
    eqc_general_solver, find_root, gc_general_solver, noc_general_solver, optimize_eq_constraints,
    optimize_general_constraints, optimize_no_constraints,
};

fn main() {
    demo_root_finding();
    demo_unconstrained();
    demo_eq_constrained();
    demo_general_constrained();
}

fn demo_root_finding() {
    fn f(x: &Col<f64>) -> Col<f64> {
        col![x[0] * x[0] - 2.0]
    }
    fn jac(x: &Col<f64>) -> Mat<f64> {
        mat![[2.0 * x[0]]]
    }
    let params = RootParams {
        e: 1e-12,
        ..Default::default()
    };
    let r = find_root(f, jac, col![1.0], Some(params)).unwrap();
    println!(
        "root find:      sqrt(2) = {:.12} ({} iterations, residual {:.2e})",
        r.x[0], r.iterations, r.residual_norm
    );
}

fn demo_unconstrained() {
    fn f(x: &Col<f64>) -> f64 {
        (x[0] + 3.0 * x[1] - 0.1).exp() + (x[0] - 3.0 * x[1] - 0.1).exp() + (-x[0] - 0.1).exp()
    }
    fn grad(x: &Col<f64>) -> Col<f64> {
        let a = (x[0] + 3.0 * x[1] - 0.1).exp();
        let b = (x[0] - 3.0 * x[1] - 0.1).exp();
        let c = (-x[0] - 0.1).exp();
        col![a + b - c, 3.0 * (a - b)]
    }
    fn hess(x: &Col<f64>) -> Mat<f64> {
        let a = (x[0] + 3.0 * x[1] - 0.1).exp();
        let b = (x[0] - 3.0 * x[1] - 0.1).exp();
        let c = (-x[0] - 0.1).exp();
        mat![[a + b + c, 3.0 * (a - b)], [3.0 * (a - b), 9.0 * (a + b)]]
    }
    let x = optimize_no_constraints(f, grad, hess, noc_general_solver, col![2.0, 4.0], 1e-10);
    let g = grad(&x);
    println!(
        "unconstrained:  x = [{:.8}, {:.8}], ‖∇f‖∞ = {:.2e}",
        x[0],
        x[1],
        g[0].abs().max(g[1].abs())
    );
}

fn f_qp(x: &Col<f64>) -> f64 {
    0.5 * (x[0] * (x[0] + 1.0) + 2.0 * x[1] * x[1])
}
fn df(x: &Col<f64>) -> Col<f64> {
    col![x[0] + 1.0, 2.0 * x[1]]
}
fn hessian(_: &Col<f64>) -> Mat<f64> {
    mat![[1.0, 0.0], [0.0, 2.0]]
}

fn demo_eq_constrained() {
    let eq = EqConstraints {
        a: mat![[1.0, 1.0]],
        b: col![1.0],
    };
    let params = EqParams {
        e: 1e-10,
        ..Default::default()
    };
    let r = optimize_eq_constraints(
        f_qp,
        df,
        hessian,
        &eq,
        eqc_general_solver,
        col![0.0, 0.0],
        Some(params),
    );
    println!(
        "eq-constrained: x = [{:.8}, {:.8}], dual = [{:.8}]",
        r.x[0], r.x[1], r.dual[0]
    );
}

fn demo_general_constrained() {
    fn h(x: &Col<f64>) -> Col<f64> {
        col![0.8 - x[0], -x[1]]
    }
    fn h_jac(_: &Col<f64>) -> Mat<f64> {
        mat![[-1.0, 0.0], [0.0, -1.0]]
    }
    fn h_hess_sum(_: &Col<f64>, _: &Col<f64>) -> Mat<f64> {
        Mat::<f64>::zeros(2, 2)
    }

    let eq = EqConstraints {
        a: mat![[1.0, 1.0]],
        b: col![1.0],
    };
    let ieq = InequalityConstraints {
        f: h,
        jac: h_jac,
        hess_sum: h_hess_sum,
    };
    let params = GeneralConstraintsParams {
        e_feasible: 1e-9,
        e_gap: 1e-9,
        ..Default::default()
    };
    let r = optimize_general_constraints(
        f_qp,
        df,
        hessian,
        &eq,
        &ieq,
        gc_general_solver,
        col![0.9, 0.2],
        InitMode::Feasible,
        false,
        Some(params),
    );
    println!(
        "gen-constrained: x = [{:.8}, {:.8}], ieq_dual = [{:.6}, {:.6}], eq_dual = [{:.6}]",
        r.x[0], r.x[1], r.ieq_dual[0], r.ieq_dual[1], r.eq_dual[0]
    );
}
