//! Equality-constrained minimization via a primal-dual Newton method.
//!
//! Rust port of `src/optim_eqc.zig`. Solves
//! `min f(x)  s.t.  A x = b` using Newton steps on the KKT conditions with a
//! residual-norm backtracking line search.

use crate::linalg::{add, axpy, mat_t_vec, mat_vec, norm_sq, solve, sub};
use faer::{Col, Mat};

/// Linear equality constraints `A x = b`, with `A` of shape `P × N`.
#[derive(Debug, Clone)]
pub struct EqConstraints {
    /// Constraint matrix `A`.
    pub a: Mat<f64>,
    /// Right-hand side `b`.
    pub b: Col<f64>,
}

/// Tuning parameters for [`optimize_eq_constraints`].
#[derive(Debug, Clone, Copy)]
pub struct EqParams {
    /// Residual-norm tolerance.
    pub e: f64,
    /// Required residual decrease, in `(0, 0.5)`.
    pub alpha: f64,
    /// Step reduction factor, in `(0, 1)`.
    pub beta: f64,
}

impl Default for EqParams {
    fn default() -> Self {
        Self {
            e: 1e-9,
            alpha: 0.4,
            beta: 0.5,
        }
    }
}

/// Outcome of [`optimize_eq_constraints`].
#[derive(Debug, Clone)]
pub struct EqConstraintsResult {
    /// Primal minimizer.
    pub x: Col<f64>,
    /// Equality multipliers.
    pub dual: Col<f64>,
}

/// Residual of the KKT conditions.
fn eq_residual(
    grad_val: &Col<f64>,
    a: &Mat<f64>,
    b: &Col<f64>,
    x: &Col<f64>,
    dual: &Col<f64>,
) -> (Col<f64>, Col<f64>) {
    let a_transpose_dual = mat_t_vec(a, dual);
    let r_dual = add(grad_val, &a_transpose_dual);
    let ax = mat_vec(a, x);
    let r_prim = sub(&ax, b);
    (r_dual, r_prim)
}

fn eq_residual_norm(r_dual: &Col<f64>, r_prim: &Col<f64>) -> f64 {
    (norm_sq(r_dual) + norm_sq(r_prim)).sqrt()
}

/// Minimizes `f` subject to `A x = b`.
///
/// The `solver` callback receives the Hessian, the constraint matrix, and the
/// dual/primal residuals, and returns the Newton step `(x_step, dual_step)`.
/// The default solver is [`eqc_general_solver`]. Pass `None` for default
/// parameters.
pub fn optimize_eq_constraints(
    f: impl Fn(&Col<f64>) -> f64,
    grad: impl Fn(&Col<f64>) -> Col<f64>,
    hess: impl Fn(&Col<f64>) -> Mat<f64>,
    eq_constraints: &EqConstraints,
    solver: impl Fn(&Mat<f64>, &Mat<f64>, &Col<f64>, &Col<f64>) -> (Col<f64>, Col<f64>),
    init_x: Col<f64>,
    params: Option<EqParams>,
) -> EqConstraintsResult {
    let param = params.unwrap_or_default();
    assert!(param.e > 0.0, "EqParams::e must be positive");
    assert!(
        param.alpha > 0.0 && param.alpha < 0.5,
        "EqParams::alpha must be in (0, 0.5)"
    );
    assert!(
        param.beta > 0.0 && param.beta < 1.0,
        "EqParams::beta must be in (0, 1)"
    );
    assert!(
        f(&init_x).is_finite(),
        "objective must be finite at the initial point"
    );

    let a = &eq_constraints.a;
    let b = &eq_constraints.b;
    let mut x = init_x;
    let mut dual = Col::zeros(a.nrows());

    loop {
        let hess_val = hess(&x);
        let grad_val = grad(&x);

        let (r_dual, r_prim) = eq_residual(&grad_val, a, b, &x, &dual);
        let norm = eq_residual_norm(&r_dual, &r_prim);
        if norm < param.e {
            break;
        }

        let (x_step, dual_step) = solver(&hess_val, a, &r_dual, &r_prim);

        let mut t = 1.0_f64;
        loop {
            let x_t = axpy(&x, t, &x_step);
            let dual_t = axpy(&dual, t, &dual_step);
            let grad_t = grad(&x_t);
            let (r_dual_t, r_prim_t) = eq_residual(&grad_t, a, b, &x_t, &dual_t);
            let norm_t = eq_residual_norm(&r_dual_t, &r_prim_t);
            if norm_t <= (1.0 - param.alpha * t) * norm {
                break;
            }
            t *= param.beta;
        }

        x = axpy(&x, t, &x_step);
        dual = axpy(&dual, t, &dual_step);
    }

    EqConstraintsResult { x, dual }
}

/// Default KKT solver for equality-constrained problems.
///
/// Builds and solves
/// ```text
/// [ H  Aᵀ ] [ x_step    ] = [ -r_dual ]
/// [ A  0  ] [ dual_step ]   [ -r_prim ]
/// ```
///
/// # Panics
/// Panics if the KKT system is singular.
pub fn eqc_general_solver(
    hess: &Mat<f64>,
    a: &Mat<f64>,
    r_dual: &Col<f64>,
    r_prim: &Col<f64>,
) -> (Col<f64>, Col<f64>) {
    let n = hess.nrows();
    let p = a.nrows();
    let size = n + p;
    let mut kkt = Mat::<f64>::zeros(size, size);
    for i in 0..n {
        for j in 0..n {
            kkt[(i, j)] = hess[(i, j)];
        }
    }
    // Aᵀ in the (0, N) block.
    for i in 0..n {
        for j in 0..p {
            kkt[(i, n + j)] = a[(j, i)];
        }
    }
    // A in the (N, 0) block.
    for i in 0..p {
        for j in 0..n {
            kkt[(n + i, j)] = a[(i, j)];
        }
    }
    let mut rhs = Col::zeros(size);
    for i in 0..n {
        rhs[i] = -r_dual[i];
    }
    for i in 0..p {
        rhs[n + i] = -r_prim[i];
    }
    let sol = solve(&kkt, &rhs).expect("eqc_general_solver: singular KKT system");
    let x_step = Col::from_fn(n, |i| sol[i]);
    let dual_step = Col::from_fn(p, |i| sol[n + i]);
    (x_step, dual_step)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{col, mat};

    fn f_qp(x: &Col<f64>) -> f64 {
        0.5 * (x[0] * (x[0] + 1.0) + 2.0 * x[1] * x[1])
    }

    fn df(x: &Col<f64>) -> Col<f64> {
        col![x[0] + 1.0, 2.0 * x[1]]
    }

    fn hessian(_: &Col<f64>) -> Mat<f64> {
        mat![[1.0, 0.0], [0.0, 2.0]]
    }

    #[test]
    fn solves_qp_with_equality() {
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
        assert!((r.x[0] - 1.0 / 3.0).abs() < 1e-9);
        assert!((r.x[1] - 2.0 / 3.0).abs() < 1e-9);
        assert!((r.x[0] + r.x[1] - 1.0).abs() < 1e-9);
        assert!((r.dual[0] + 4.0 / 3.0).abs() < 1e-9);
    }
}
