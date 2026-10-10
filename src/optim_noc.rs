//! Unconstrained minimization via Newton's method with Armijo line search.
//!
//! Rust port of `src/optim_noc.zig`.

use crate::linalg::{axpy, dot, negate, solve};
use faer::{Col, Mat};

/// Armijo backtracking factor.
const ARMIJO_C: f64 = 0.4;

/// Finds a step length `t` such that
/// `f(x + t*dir) <= f(x) + c * t * (grad · dir)` with `c = 0.4`.
fn armijo(f: &impl Fn(&Col<f64>) -> f64, x: &Col<f64>, grad: &Col<f64>, dir: &Col<f64>) -> f64 {
    let mut t = 1.0_f64;
    let f_x = f(x);
    while f(&axpy(x, t, dir)) > f_x + ARMIJO_C * t * dot(grad, dir) {
        t *= 0.5;
    }
    t
}

/// Minimizes an unconstrained objective using Newton steps.
///
/// The `solver` callback receives the Hessian and gradient and must return the
/// Newton direction, i.e. the solution of `H * dir = -grad`. The default
/// direction solver is [`noc_general_solver`]. Iteration stops when the
/// Newton decrement `-grad · dir / 2` drops below `e`.
pub fn optimize_no_constraints(
    f: impl Fn(&Col<f64>) -> f64,
    grad: impl Fn(&Col<f64>) -> Col<f64>,
    hess: impl Fn(&Col<f64>) -> Mat<f64>,
    solver: impl Fn(&Mat<f64>, &Col<f64>) -> Col<f64>,
    init_x: Col<f64>,
    e: f64,
) -> Col<f64> {
    let mut x = init_x;
    loop {
        let hess_val = hess(&x);
        let grad_val = grad(&x);
        let dir = solver(&hess_val, &grad_val);
        let t = armijo(&f, &x, &grad_val, &dir);
        let err = -dot(&grad_val, &dir) / 2.0;
        if err < e {
            break;
        }
        x = axpy(&x, t, &dir);
    }
    x
}

/// Default Newton direction solver: solves `H * dir = -grad` with pivoted LU.
///
/// # Panics
/// Panics if the Hessian is singular.
pub fn noc_general_solver(hess: &Mat<f64>, grad: &Col<f64>) -> Col<f64> {
    let rhs = negate(grad);
    solve(hess, &rhs).expect("noc_general_solver: singular Hessian")
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::col;

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
        Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => a + b + c,
            (1, 1) => 9.0 * (a + b),
            _ => 3.0 * (a - b),
        })
    }

    #[test]
    fn minimizes_exponential() {
        let result =
            optimize_no_constraints(f, grad, hess, noc_general_solver, col![2.0, 4.0], 1e-10);
        let g = grad(&result);
        assert!(g[0].abs() < 1e-4);
        assert!(g[1].abs() < 1e-4);
    }
}
