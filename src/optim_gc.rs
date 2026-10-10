//! General constrained minimization via a primal-dual interior-point method.
//!
//! Rust port of `src/optim_gc.zig`. Solves
//! ```text
//! min f(x)   s.t.  A x = b,  h(x) <= 0
//! ```
//! by driving the perturbed KKT conditions
//! ```text
//! r_dual = ∇f + Aᵀ y + ∇hᵀ z
//! r_prim = A x - b
//! r_cent = s * z - 1/t
//! r_ieq  = h + s
//! ```
//! to zero, where `s` are the slacks and `z` the inequality multipliers.

use crate::linalg::{
    add, axpy, dot, mat_add, mat_t_vec, mat_vec, mul, negate, norm_sq, solve, sub,
};
use crate::optim_eqc::EqConstraints;
use faer::{Col, Mat};

/// Nonlinear inequality constraints `h(x) <= 0`.
pub struct InequalityConstraints<F, J, Hs> {
    /// Constraint function `h(x)` returning the constraint values.
    pub f: F,
    /// Jacobian `∇h(x)`, shape `M × N`.
    pub jac: J,
    /// Weighted Hessian sum `Σ_i z_i ∇²h_i(x)`, shape `N × N`.
    pub hess_sum: Hs,
}

/// How the initial slack variables are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitMode {
    /// The initial point is strictly feasible (`h(x) < 0`).
    Feasible,
    /// The initial point may violate the inequalities; slacks are chosen
    /// positive and a correction step is available.
    Infeasible,
}

/// Tuning parameters for [`optimize_general_constraints`].
#[derive(Debug, Clone, Copy)]
pub struct GeneralConstraintsParams {
    /// Primal/dual feasibility tolerance.
    pub e_feasible: f64,
    /// Complementary slackness (gap) tolerance.
    pub e_gap: f64,
    /// Centering parameter in `(1, ∞)`.
    pub miu: f64,
    /// Required residual decrease, in `(0, 0.5)`.
    pub alpha: f64,
    /// Step reduction factor, in `(0, 1)`.
    pub beta: f64,
}

impl Default for GeneralConstraintsParams {
    fn default() -> Self {
        Self {
            e_feasible: 1e-9,
            e_gap: 1e-9,
            miu: 10.0,
            alpha: 0.1,
            beta: 0.5,
        }
    }
}

/// Outcome of [`optimize_general_constraints`].
#[derive(Debug, Clone)]
pub struct GeneralConstraintsResult {
    /// Primal minimizer.
    pub x: Col<f64>,
    /// Inequality multipliers.
    pub ieq_dual: Col<f64>,
    /// Equality multipliers.
    pub eq_dual: Col<f64>,
}

/// Builds `-∇h(x) · diag(z)`, shape `M × N` (entry `[i][j] = -h_jac[i][j] * z[i]`).
fn h_jac_diag_ieq_dual(h_jac: &Mat<f64>, ieq_dual: &Col<f64>) -> Mat<f64> {
    Mat::from_fn(h_jac.nrows(), h_jac.ncols(), |i, j| {
        -h_jac[(i, j)] * ieq_dual[i]
    })
}

/// Positive initial slacks: `-h_i` where `h_i < 0`, otherwise `1`.
fn positive_initial_slack(h: &Col<f64>) -> Col<f64> {
    Col::from_fn(h.nrows(), |i| if h[i] < 0.0 { -h[i] } else { 1.0 })
}

/// Computes the four KKT residuals.
#[allow(clippy::too_many_arguments)]
fn ie_r(
    x_grad: &Col<f64>,
    h: &Col<f64>,
    slack: &Col<f64>,
    ieq_dual: &Col<f64>,
    eq_dual: &Col<f64>,
    x: &Col<f64>,
    h_jac: &Mat<f64>,
    a: &Mat<f64>,
    b: &Col<f64>,
    t: f64,
) -> (Col<f64>, Col<f64>, Col<f64>, Col<f64>) {
    let a_transpose_dual = mat_t_vec(a, eq_dual);
    let h_jac_transpose_cent = mat_t_vec(h_jac, ieq_dual);
    let r_dual = add(&add(x_grad, &a_transpose_dual), &h_jac_transpose_cent);
    let inv_t = 1.0 / t;
    let r_cent = Col::from_fn(slack.nrows(), |i| slack[i] * ieq_dual[i] - inv_t);
    let ax = mat_vec(a, x);
    let r_prim = sub(&ax, b);
    let r_ieq = add(h, slack);
    (r_dual, r_cent, r_prim, r_ieq)
}

fn ie_r_norm(r_dual: &Col<f64>, r_cent: &Col<f64>, r_prim: &Col<f64>, r_ieq: &Col<f64>) -> f64 {
    (norm_sq(r_prim) + norm_sq(r_dual) + norm_sq(r_cent) + norm_sq(r_ieq)).sqrt()
}

/// Minimizes `f` subject to `A x = b` and `h(x) <= 0`.
///
/// The `solver` callback receives the assembled KKT blocks and residuals and
/// returns `(x_step, ieq_dual_step, eq_dual_step)`; the default is
/// [`gc_general_solver`].
///
/// `init_mode` selects how slacks are initialized, and `use_correction_step`
/// enables the Mehrotra-style correction when running in
/// [`InitMode::Infeasible`]. Pass `None` for default parameters.
#[allow(clippy::too_many_arguments)]
pub fn optimize_general_constraints<F, Gr, Hs, IF, IJ, IH, S>(
    f: F,
    grad: Gr,
    hess: Hs,
    eq_constraints: &EqConstraints,
    ieq_constraints: &InequalityConstraints<IF, IJ, IH>,
    solver: S,
    init_x: Col<f64>,
    init_mode: InitMode,
    use_correction_step: bool,
    params: Option<GeneralConstraintsParams>,
) -> GeneralConstraintsResult
where
    F: Fn(&Col<f64>) -> f64,
    Gr: Fn(&Col<f64>) -> Col<f64>,
    Hs: Fn(&Col<f64>) -> Mat<f64>,
    IF: Fn(&Col<f64>) -> Col<f64>,
    IJ: Fn(&Col<f64>) -> Mat<f64>,
    IH: Fn(&Col<f64>, &Col<f64>) -> Mat<f64>,
    S: Fn(
        &Mat<f64>,
        &Mat<f64>,
        &Mat<f64>,
        &Mat<f64>,
        &Col<f64>,
        &Col<f64>,
        &Col<f64>,
        &Col<f64>,
    ) -> (Col<f64>, Col<f64>, Col<f64>),
{
    let param = params.unwrap_or_default();
    let a = &eq_constraints.a;
    let b = &eq_constraints.b;

    let mut x = init_x;
    let init_h = (ieq_constraints.f)(&x);
    let m = init_h.nrows();
    let mut slack: Col<f64> = match init_mode {
        InitMode::Feasible => negate(&init_h),
        InitMode::Infeasible => positive_initial_slack(&init_h),
    };
    let mut ieq_dual = Col::from_fn(m, |_| 1.0);
    let mut eq_dual = Col::zeros(a.nrows());

    assert!(param.e_feasible > 0.0, "e_feasible must be positive");
    assert!(param.e_gap > 0.0, "e_gap must be positive");
    assert!(param.miu > 1.0, "miu must be greater than 1");
    assert!(
        param.alpha > 0.0 && param.alpha < 0.5,
        "alpha must be in (0, 0.5)"
    );
    assert!(
        param.beta > 0.0 && param.beta < 1.0,
        "beta must be in (0, 1)"
    );
    assert!(
        f(&x).is_finite(),
        "objective must be finite at the initial point"
    );

    loop {
        let grad_val = grad(&x);
        let hess_val = hess(&x);
        let h_val = (ieq_constraints.f)(&x);
        if init_mode == InitMode::Feasible {
            slack = negate(&h_val);
        }
        let h_jac_val = (ieq_constraints.jac)(&x);
        let hess_sum_val = (ieq_constraints.hess_sum)(&x, &ieq_dual);
        let h_jac_transpose = h_jac_val.transpose().to_owned();

        let gap = dot(&slack, &ieq_dual);
        debug_assert!(gap > 0.0, "complementary gap must be positive");
        for i in 0..m {
            if init_mode == InitMode::Feasible {
                debug_assert!(h_val[i] < 0.0, "feasible start requires h(x) < 0");
            }
            debug_assert!(slack[i] > 0.0, "slacks must stay positive");
            debug_assert!(
                ieq_dual[i] > 0.0,
                "inequality multipliers must stay positive"
            );
        }

        let centering_t = param.miu * (m as f64) / gap;
        let (r_dual, r_cent, r_prim, r_ieq) = ie_r(
            &grad_val,
            &h_val,
            &slack,
            &ieq_dual,
            &eq_dual,
            &x,
            &h_jac_val,
            a,
            b,
            centering_t,
        );
        let norm = ie_r_norm(&r_dual, &r_cent, &r_prim, &r_ieq);
        let e_feas_sq = param.e_feasible * param.e_feasible;
        if norm_sq(&r_prim) < e_feas_sq
            && norm_sq(&r_ieq) < e_feas_sq
            && norm_sq(&r_dual) < e_feas_sq
            && gap < param.e_gap
        {
            break;
        }

        let block_hess = mat_add(&hess_val, &hess_sum_val);
        let block_h_jac_diag_ieq_dual = h_jac_diag_ieq_dual(&h_jac_val, &ieq_dual);
        let solver_r_cent = sub(&r_cent, &mul(&ieq_dual, &r_ieq));

        let (mut x_step, mut ieq_dual_step, mut eq_dual_step) = solver(
            &block_hess,
            &h_jac_transpose,
            a,
            &block_h_jac_diag_ieq_dual,
            &slack,
            &r_dual,
            &solver_r_cent,
            &r_prim,
        );

        let h_x_step = mat_vec(&h_jac_val, &x_step);
        let mut slack_step = sub(&negate(&r_ieq), &h_x_step);

        if use_correction_step && init_mode == InitMode::Infeasible {
            let x_full_step = add(&x, &x_step);
            let slack_full_step = add(&slack, &slack_step);
            let h_full_step = (ieq_constraints.f)(&x_full_step);
            let correction_r_ieq = add(&h_full_step, &slack_full_step);

            let correction_r_dual = Col::zeros(x.nrows());
            let correction_r_prim = Col::zeros(a.nrows());
            let correction_r_cent = negate(&mul(&ieq_dual, &correction_r_ieq));

            let (correction_x_step, correction_ieq_dual_step, correction_eq_dual_step) = solver(
                &block_hess,
                &h_jac_transpose,
                a,
                &block_h_jac_diag_ieq_dual,
                &slack,
                &correction_r_dual,
                &correction_r_cent,
                &correction_r_prim,
            );

            let c_h_x_step = mat_vec(&h_jac_val, &correction_x_step);
            let correction_slack_step = sub(&negate(&correction_r_ieq), &c_h_x_step);

            x_step = add(&x_step, &correction_x_step);
            slack_step = add(&slack_step, &correction_slack_step);
            ieq_dual_step = add(&ieq_dual_step, &correction_ieq_dual_step);
            eq_dual_step = add(&eq_dual_step, &correction_eq_dual_step);
        }

        // Fraction-to-the-boundary step length.
        let mut t = 1.0_f64;
        for i in 0..m {
            if ieq_dual_step[i] < 0.0 {
                t = t.min(-0.99 * ieq_dual[i] / ieq_dual_step[i]);
            }
            if slack_step[i] < 0.0 {
                t = t.min(-0.99 * slack[i] / slack_step[i]);
            }
        }

        // Backtracking on the residual norm.
        loop {
            let x_t = axpy(&x, t, &x_step);
            let slack_t: Col<f64> = match init_mode {
                InitMode::Feasible => negate(&(ieq_constraints.f)(&x_t)),
                InitMode::Infeasible => axpy(&slack, t, &slack_step),
            };
            let ieq_dual_t = axpy(&ieq_dual, t, &ieq_dual_step);
            let eq_dual_t = axpy(&eq_dual, t, &eq_dual_step);
            let grad_t = grad(&x_t);
            let h_t = (ieq_constraints.f)(&x_t);

            let mut reject = false;
            for i in 0..m {
                if (init_mode == InitMode::Feasible && h_t[i] >= 0.0)
                    || slack_t[i] <= 0.0
                    || ieq_dual_t[i] <= 0.0
                {
                    reject = true;
                    break;
                }
            }
            if reject {
                t *= param.beta;
                continue;
            }

            let h_jac_t = (ieq_constraints.jac)(&x_t);
            let (r_dual_t, r_cent_t, r_prim_t, r_ieq_t) = ie_r(
                &grad_t,
                &h_t,
                &slack_t,
                &ieq_dual_t,
                &eq_dual_t,
                &x_t,
                &h_jac_t,
                a,
                b,
                centering_t,
            );
            let norm_t = ie_r_norm(&r_dual_t, &r_cent_t, &r_prim_t, &r_ieq_t);
            if norm_t <= (1.0 - param.alpha * t) * norm {
                break;
            }
            t *= param.beta;
        }

        x = axpy(&x, t, &x_step);
        if init_mode == InitMode::Infeasible {
            slack = axpy(&slack, t, &slack_step);
        }
        ieq_dual = axpy(&ieq_dual, t, &ieq_dual_step);
        eq_dual = axpy(&eq_dual, t, &eq_dual_step);
    }

    GeneralConstraintsResult {
        x,
        ieq_dual,
        eq_dual,
    }
}

/// Default KKT solver for general constrained problems.
///
/// Assembles and solves the `(N + M + P)` system
/// ```text
/// [ H      ∇hᵀ        Aᵀ ] [ x_step       ]   [ -r_dual ]
/// [ -D_h   diag(s)    0  ] [ ieq_dual_step] = [ -r_cent ]
/// [ A      0          0  ] [ eq_dual_step ]   [ -r_prim ]
/// ```
/// where `-D_h = -∇h · diag(z)`.
///
/// # Panics
/// Panics if the KKT system is singular.
#[allow(clippy::too_many_arguments)]
pub fn gc_general_solver(
    block_hess: &Mat<f64>,
    block_h_jac_transpose: &Mat<f64>,
    block_a: &Mat<f64>,
    block_h_jac_diag_iedual: &Mat<f64>,
    block_diag_h: &Col<f64>,
    r_dual: &Col<f64>,
    r_cent: &Col<f64>,
    r_prim: &Col<f64>,
) -> (Col<f64>, Col<f64>, Col<f64>) {
    let n = block_hess.nrows();
    let m = block_diag_h.nrows();
    let p = block_a.nrows();
    let size = n + m + p;
    let mut kkt = Mat::<f64>::zeros(size, size);

    // (0, 0) = H
    for i in 0..n {
        for j in 0..n {
            kkt[(i, j)] = block_hess[(i, j)];
        }
    }
    // (0, N) = ∇hᵀ  (N × M)
    for i in 0..n {
        for j in 0..m {
            kkt[(i, n + j)] = block_h_jac_transpose[(i, j)];
        }
    }
    // (0, N + M) = Aᵀ  (N × P)
    for i in 0..n {
        for j in 0..p {
            kkt[(i, n + m + j)] = block_a[(j, i)];
        }
    }
    // (N, 0) = -∇h · diag(z)  (M × N)
    for i in 0..m {
        for j in 0..n {
            kkt[(n + i, j)] = block_h_jac_diag_iedual[(i, j)];
        }
    }
    // (N, N) = diag(s)
    for i in 0..m {
        kkt[(n + i, n + i)] = block_diag_h[i];
    }
    // (N + M, 0) = A  (P × N)
    for i in 0..p {
        for j in 0..n {
            kkt[(n + m + i, j)] = block_a[(i, j)];
        }
    }

    let mut rhs = Col::zeros(size);
    for i in 0..n {
        rhs[i] = -r_dual[i];
    }
    for i in 0..m {
        rhs[n + i] = -r_cent[i];
    }
    for i in 0..p {
        rhs[n + m + i] = -r_prim[i];
    }

    let sol = solve(&kkt, &rhs).expect("gc_general_solver: singular KKT system");
    let x_step = Col::from_fn(n, |i| sol[i]);
    let ieq_dual_step = Col::from_fn(m, |i| sol[n + i]);
    let eq_dual_step = Col::from_fn(p, |i| sol[n + m + i]);
    (x_step, ieq_dual_step, eq_dual_step)
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

    fn h(x: &Col<f64>) -> Col<f64> {
        col![0.8 - x[0], -x[1]]
    }

    fn h_jac(_: &Col<f64>) -> Mat<f64> {
        mat![[-1.0, 0.0], [0.0, -1.0]]
    }

    fn h_hess_sum(_: &Col<f64>, _: &Col<f64>) -> Mat<f64> {
        Mat::<f64>::zeros(2, 2)
    }

    fn make_eq() -> EqConstraints {
        EqConstraints {
            a: mat![[1.0, 1.0]],
            b: col![1.0],
        }
    }

    type TestIeq = InequalityConstraints<
        fn(&Col<f64>) -> Col<f64>,
        fn(&Col<f64>) -> Mat<f64>,
        fn(&Col<f64>, &Col<f64>) -> Mat<f64>,
    >;

    fn make_ieq() -> TestIeq {
        InequalityConstraints {
            f: h,
            jac: h_jac,
            hess_sum: h_hess_sum,
        }
    }

    fn check(result: GeneralConstraintsResult) {
        assert!((result.x[0] - 0.8).abs() < 1e-6, "x0 = {}", result.x[0]);
        assert!((result.x[1] - 0.2).abs() < 1e-6, "x1 = {}", result.x[1]);
        assert!((result.x[0] + result.x[1] - 1.0).abs() < 1e-9);
        assert!(result.x[0] >= 0.8 - 1e-9);
        assert!(result.x[1] >= -1e-9);
        assert!(
            (result.ieq_dual[0] - 1.4).abs() < 1e-5,
            "z0 = {}",
            result.ieq_dual[0]
        );
        assert!(
            result.ieq_dual[1].abs() < 1e-5,
            "z1 = {}",
            result.ieq_dual[1]
        );
        assert!(
            (result.eq_dual[0] + 0.4).abs() < 1e-5,
            "y = {}",
            result.eq_dual[0]
        );
    }

    #[test]
    fn solves_qp_feasible() {
        let eq = make_eq();
        let ieq = make_ieq();
        let params = GeneralConstraintsParams {
            e_feasible: 1e-9,
            e_gap: 1e-9,
            ..Default::default()
        };
        let result = optimize_general_constraints(
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
        check(result);
    }

    #[test]
    fn solves_qp_infeasible_with_correction() {
        let eq = make_eq();
        let ieq = make_ieq();
        let params = GeneralConstraintsParams {
            e_feasible: 1e-9,
            e_gap: 1e-9,
            ..Default::default()
        };
        let result = optimize_general_constraints(
            f_qp,
            df,
            hessian,
            &eq,
            &ieq,
            gc_general_solver,
            col![0.0, 0.0],
            InitMode::Infeasible,
            true,
            Some(params),
        );
        check(result);
    }

    #[test]
    fn helpers_are_consistent() {
        assert!(crate::linalg::all_finite(&col![1.0, 2.0]));
        assert!(!crate::linalg::all_finite(&col![1.0, f64::NAN]));
        let _ = crate::linalg::scale(2.0, &col![1.0]);
    }
}
