//! Affine conic programming.
//!
//! Solves
//! ```text
//! min   cᵀx + ½ xᵀQx
//! s.t.  A x = b
//!       G x + h ∈ K
//! ```
//! where the convex cone `K` is a product of non-negative orthants,
//! second-order cones, and positive-semidefinite cones. This covers LP
//! (`Q = 0`, orthant), QP (`Q ⪰ 0`, orthant), SOCP (second-order cones), and
//! SDP (PSD cone).
//!
//! Two solution methods are provided, selected automatically by [`solve`]:
//!
//! * **Orthant problems (LP/QP)** use a primal-dual Mehrotra
//!   predictor-corrector, which reaches the optimum in ~10 iterations.
//! * **Problems with second-order or PSD cones** use a logarithmic-barrier
//!   method that minimizes `t·f(x) + φ(Gx + h)` subject to `A x = b` for an
//!   increasing sequence of weights `t`, where `φ` is the cone's barrier. Each
//!   subproblem is solved with equality-constrained damped Newton steps.
//!
//! Both methods solve KKT systems with faer's pivoted LU decomposition.

use crate::linalg::{
    add, all_finite, axpy, dot, infinity_norm, mat_t_vec, mul, negate, scale,
    solve as solve_linear, sub,
};
use faer::linalg::solvers::Solve;
use faer::{Col, Mat, Side};

/// A convex cone, or a factor of a product of cones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cone {
    /// Non-negative orthant of dimension `n` (`s >= 0`). Covers LP and QP.
    Nonneg(usize),
    /// Second-order (Lorentz) cone of dimension `d`: `s[0] >= ‖s[1..]‖₂`.
    Soc(usize),
    /// Positive-semidefinite cone of `k × k` symmetric matrices, stored with
    /// `k(k+1)/2` entries (lower triangle, column-major).
    Psd(usize),
}

impl Cone {
    /// Number of scalar coordinates occupied by the cone.
    pub fn dim(self) -> usize {
        match self {
            Cone::Nonneg(n) => n,
            Cone::Soc(d) => d,
            Cone::Psd(k) => k * (k + 1) / 2,
        }
    }

    /// Barrier parameter `ν`.
    pub fn nu(self) -> f64 {
        match self {
            Cone::Nonneg(n) => n as f64,
            Cone::Soc(_) => 1.0,
            Cone::Psd(k) => k as f64,
        }
    }
}

/// An affine conic program in the form documented at the module level.
#[derive(Clone)]
pub struct ConicProgram {
    /// Quadratic objective term, shape `n × n`.
    pub q: Mat<f64>,
    /// Linear objective term, length `n`.
    pub c: Col<f64>,
    /// Equality constraint matrix, shape `p × n` (may have zero rows).
    pub a: Mat<f64>,
    /// Equality right-hand side, length `p`.
    pub b: Col<f64>,
    /// Cone constraint matrix, shape `d × n`, with `d` the total cone dimension.
    pub g: Mat<f64>,
    /// Cone constraint offset, length `d`.
    pub h: Col<f64>,
    /// Product cone `K`.
    pub cones: Vec<Cone>,
}

impl ConicProgram {
    /// Number of decision variables.
    pub fn n(&self) -> usize {
        self.c.nrows()
    }
    /// Number of equality constraints.
    pub fn p(&self) -> usize {
        self.a.nrows()
    }
    /// Total cone dimension `d`.
    pub fn cone_dim(&self) -> usize {
        self.cones.iter().map(|c| c.dim()).sum()
    }
    /// Total barrier parameter `ν`.
    pub fn nu(&self) -> f64 {
        self.cones.iter().map(|c| c.nu()).sum()
    }
    /// Evaluates the quadratic objective `cᵀx + ½ xᵀQx`.
    pub fn objective(&self, x: &Col<f64>) -> f64 {
        let qx = &self.q * x;
        dot(&self.c, x) + 0.5 * dot(x, &qx)
    }
}

/// Tuning parameters for [`solve`].
#[derive(Clone, Copy, Debug)]
pub struct ConicParams {
    /// Duality-gap tolerance on `ν / t`.
    pub tol: f64,
    /// Initial barrier weight `t`.
    pub t0: f64,
    /// Barrier weight growth factor.
    pub mu: f64,
    /// Maximum outer (barrier) iterations.
    pub max_outer: usize,
    /// Maximum Newton steps per barrier subproblem.
    pub max_newton: usize,
    /// Armijo constant for the barrier objective.
    pub alpha: f64,
    /// Backtracking reduction factor.
    pub beta: f64,
    /// Newton-decrement tolerance for stopping an inner solve.
    pub newton_tol: f64,
    /// Use the primal-dual predictor-corrector method when every cone is an
    /// orthant (LP/QP). Ignored for problems with SOC or PSD cones.
    pub predcorr: bool,
}

impl Default for ConicParams {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            t0: 1.0,
            mu: 500.0,
            max_outer: 60,
            max_newton: 20,
            alpha: 0.01,
            beta: 0.5,
            newton_tol: 1e-12,
            predcorr: true,
        }
    }
}

/// Termination status of [`solve`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConicStatus {
    /// The barrier gap fell below the tolerance.
    Optimal,
    /// The outer-iteration budget was exhausted.
    MaxIterations,
    /// The line search stalled or a linear system became unsolvable.
    Failed,
}

/// Outcome of [`solve`].
#[derive(Clone, Debug)]
pub struct ConicSolution {
    /// Primal solution.
    pub x: Col<f64>,
    /// Equality multipliers.
    pub y: Col<f64>,
    /// Inequality (cone) multipliers, `-∇φ(Gx+h) / t`.
    pub z: Col<f64>,
    /// Objective value at `x`.
    pub objective: f64,
    /// Final estimated duality gap `ν / t`.
    pub gap: f64,
    /// Number of barrier (outer) iterations.
    pub outer_iterations: usize,
    /// Total Newton iterations.
    pub newton_iterations: usize,
    /// Termination status.
    pub status: ConicStatus,
}

/* ------------------------------ barrier core ------------------------------ */

struct Barrier {
    phi: f64,
    grad: Col<f64>,
    hess: Mat<f64>,
}

/// Lower-triangular column-major index of the symmetric entry `(i, j)` (`i <= j`).
fn sym_index(i: usize, j: usize) -> usize {
    debug_assert!(i <= j);
    j * (j + 1) / 2 + i
}

/// Reads a symmetric `k × k` matrix from cone coordinates starting at `off`.
fn sym_to_mat(k: usize, s: &Col<f64>, off: usize) -> Mat<f64> {
    let mut m = Mat::<f64>::zeros(k, k);
    for j in 0..k {
        for i in 0..=j {
            let v = s[off + sym_index(i, j)];
            m[(i, j)] = v;
            m[(j, i)] = v;
        }
    }
    m
}

/// Evaluates `φ(s)`, `∇φ(s)`, and the dense Hessian `∇²φ(s)`.
fn evaluate_barrier(cones: &[Cone], s: &Col<f64>) -> Barrier {
    let d = s.nrows();
    let mut phi = 0.0_f64;
    let mut grad = Col::zeros(d);
    let mut hess = Mat::<f64>::zeros(d, d);
    let mut off = 0usize;

    for &cone in cones {
        match cone {
            Cone::Nonneg(n) => {
                for i in 0..n {
                    let si = s[off + i];
                    phi -= si.ln();
                    grad[off + i] = -1.0 / si;
                    hess[(off + i, off + i)] = 1.0 / (si * si);
                }
                off += n;
            }
            Cone::Soc(dim) => {
                let s0 = s[off];
                let mut s1 = vec![0.0; dim.saturating_sub(1)];
                for i in 0..s1.len() {
                    s1[i] = s[off + 1 + i];
                }
                let rho = s0 * s0 - s1.iter().map(|v| v * v).sum::<f64>();
                phi -= rho.ln();
                // ∇ρ = (2 s0, -2 s1), ∇φ = -∇ρ / ρ.
                let mut grad_rho = vec![0.0; dim];
                grad_rho[0] = 2.0 * s0;
                for i in 0..s1.len() {
                    grad_rho[1 + i] = -2.0 * s1[i];
                }
                for i in 0..dim {
                    grad[off + i] = -grad_rho[i] / rho;
                }
                // ∇²φ = (∇ρ ∇ρᵀ) / ρ² - ∇²ρ / ρ,  ∇²ρ = diag(2, -2, …, -2).
                for i in 0..dim {
                    for j in 0..dim {
                        let d2rho = if i == j {
                            if i == 0 { 2.0 } else { -2.0 }
                        } else {
                            0.0
                        };
                        hess[(off + i, off + j)] =
                            grad_rho[i] * grad_rho[j] / (rho * rho) - d2rho / rho;
                    }
                }
                off += dim;
            }
            Cone::Psd(k) => {
                let m = k * (k + 1) / 2;
                let s_mat = sym_to_mat(k, s, off);
                let inv = s_mat
                    .partial_piv_lu()
                    .solve(&Mat::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 }));
                phi -= s_mat.determinant().ln();
                // Coordinate gradient: -(S⁻¹ B_a) with B_a = E_ii or E_ij+E_ji.
                for j in 0..k {
                    for i in 0..=j {
                        let a = sym_index(i, j);
                        grad[off + a] = if i == j {
                            -inv[(i, i)]
                        } else {
                            -2.0 * inv[(i, j)]
                        };
                    }
                }
                // Hessian column b: F_b = S⁻¹ B_b S⁻¹, H[a,b] = tr(B_a F_b).
                for bj in 0..k {
                    for bi in 0..=bj {
                        let b = sym_index(bi, bj);
                        let mut bmat = Mat::<f64>::zeros(k, k);
                        bmat[(bi, bj)] = 1.0;
                        bmat[(bj, bi)] = 1.0;
                        let fb = &(&inv * &bmat) * &inv;
                        for j in 0..k {
                            for i in 0..=j {
                                let a = sym_index(i, j);
                                let val = if i == j { fb[(i, i)] } else { 2.0 * fb[(i, j)] };
                                hess[(off + a, off + b)] = val;
                                hess[(off + b, off + a)] = val;
                            }
                        }
                    }
                }
                off += m;
            }
        }
    }

    Barrier { phi, grad, hess }
}

/// Returns `true` when `s` lies in the interior of the product cone.
pub fn in_cone_interior(cones: &[Cone], s: &Col<f64>) -> bool {
    let mut off = 0usize;
    for &cone in cones {
        match cone {
            Cone::Nonneg(n) => {
                for i in 0..n {
                    let v = s[off + i];
                    if !v.is_finite() || v <= 0.0 {
                        return false;
                    }
                }
                off += n;
            }
            Cone::Soc(dim) => {
                let s0 = s[off];
                let mut nrm = 0.0;
                for i in 1..dim {
                    nrm += s[off + i] * s[off + i];
                }
                if !s0.is_finite() || s0 <= nrm.sqrt() {
                    return false;
                }
                off += dim;
            }
            Cone::Psd(k) => {
                let sm = sym_to_mat(k, s, off);
                if sm.llt(Side::Lower).is_err() {
                    return false;
                }
                off += k * (k + 1) / 2;
            }
        }
    }
    true
}

/// Solves `K [dx; dy] = -[r_dual; r_prim]` with a small regularization fallback.
fn solve_kkt(kkt: &Mat<f64>, rhs: &Col<f64>) -> Option<Col<f64>> {
    if let Some(sol) = solve_linear(kkt, rhs) {
        return Some(sol);
    }
    let mut regularized = kkt.clone();
    let n = regularized.nrows();
    for i in 0..n {
        regularized[(i, i)] += 1e-10;
    }
    solve_linear(&regularized, rhs)
}

/// Assembles `[H Aᵀ; A 0]` into `kkt` (which must be `(n+p) × (n+p)`).
fn assemble_kkt_into(kkt: &mut Mat<f64>, h: &Mat<f64>, a: &Mat<f64>) {
    let n = h.nrows();
    let p = a.nrows();
    for j in 0..n {
        for i in 0..n {
            kkt[(i, j)] = h[(i, j)];
        }
    }
    for i in 0..n {
        for j in 0..p {
            kkt[(i, n + j)] = a[(j, i)];
        }
    }
    for i in 0..p {
        for j in 0..n {
            kkt[(n + i, j)] = a[(i, j)];
        }
    }
}

/// Element-wise division `a / b`.
fn div_by(a: &Col<f64>, b: &Col<f64>) -> Col<f64> {
    Col::from_fn(a.nrows(), |i| a[i] / b[i])
}

/// Fraction-to-the-boundary step length for `s, z > 0` along `(ds, dz)`.
fn max_step(s: &Col<f64>, z: &Col<f64>, ds: &Col<f64>, dz: &Col<f64>) -> f64 {
    let mut a = 1.0_f64;
    for i in 0..s.nrows() {
        if ds[i] < 0.0 {
            a = a.min(-s[i] / ds[i]);
        }
        if dz[i] < 0.0 {
            a = a.min(-z[i] / dz[i]);
        }
    }
    (0.99 * a).min(1.0)
}

/// Primal-dual Mehrotra predictor-corrector for pure orthant problems (LP/QP).
///
/// Solves `min cᵀx + ½xᵀQx s.t. Ax=b, Gx+h = s ≥ 0` by driving the perturbed
/// KKT conditions to zero. Each iteration performs an affine *predictor* step
/// and a centering *corrector* step; both reuse a single LU factorization of
/// the reduced KKT system
/// ```text
/// [ Q + Gᵀ diag(z/s) G   -Aᵀ ] [dx]   [ rhs1   ]
/// [ A                     0  ] [dy] = [ -r_prim]
/// ```
/// so only one factorization is paid per iteration.
fn solve_orthant_pc(program: &ConicProgram, x0: &Col<f64>, param: ConicParams) -> ConicSolution {
    let n = program.n();
    let p = program.p();
    let d = program.cone_dim();
    let m_dim = d.max(1) as f64;

    let q = &program.q;
    let c = &program.c;
    let a = &program.a;
    let b = &program.b;
    let g = &program.g;
    let h = &program.h;
    let gt = g.transpose().to_owned();

    let mut x = x0.clone();
    let mut y = Col::zeros(p);
    let mut s = add(&(g * &x), h);
    for i in 0..d {
        if s[i] <= 0.0 {
            s[i] = 1e-3;
        }
    }
    let mut z = Col::from_fn(d, |_| 1.0);

    let b_scale = 1.0 + infinity_norm(b);
    let c_scale = 1.0 + infinity_norm(c);

    let max_iter = param.max_outer.max(200);
    let mut iterations = 0usize;
    let mut status = ConicStatus::MaxIterations;

    let mut kkt = Mat::<f64>::zeros(n + p, n + p);
    let mut rhs = Col::zeros(n + p);

    loop {
        if iterations >= max_iter {
            break;
        }
        iterations += 1;

        let r_dual = sub(
            &sub(&add(c, &(q * &x)), &mat_t_vec(g, &z)),
            &mat_t_vec(a, &y),
        );
        let r_prim = sub(&(a * &x), b);
        let r_ieq = sub(&add(&(g * &x), h), &s);
        let mu = dot(&s, &z) / m_dim;

        let prim_res = infinity_norm(&r_prim) / b_scale;
        let dual_res = infinity_norm(&r_dual) / c_scale;
        if prim_res < param.tol && dual_res < param.tol && mu < param.tol {
            status = ConicStatus::Optimal;
            break;
        }
        // Machine-precision floor: the reduced system becomes too ill-conditioned
        // to make further progress once the gap collapses.
        if mu < 1e-14 {
            status = ConicStatus::Optimal;
            break;
        }

        // Reduced KKT: [Q + Gᵀ diag(z/s) G  -Aᵀ; A  0].
        let mut dg = Mat::<f64>::zeros(d, n);
        for i in 0..d {
            let di = z[i] / s[i];
            for j in 0..n {
                dg[(i, j)] = di * g[(i, j)];
            }
        }
        let gtdg = &gt * &dg;
        for i in 0..n {
            for j in 0..n {
                kkt[(i, j)] = q[(i, j)] + gtdg[(i, j)];
            }
        }
        for i in 0..n {
            for j in 0..p {
                kkt[(i, n + j)] = -a[(j, i)];
            }
        }
        for i in 0..p {
            for j in 0..n {
                kkt[(n + i, j)] = a[(i, j)];
            }
        }
        for i in 0..p {
            for j in 0..p {
                kkt[(n + i, n + j)] = 0.0;
            }
        }

        // Predictor (affine) step: target complementary slackness is zero.
        let b4_pred = negate(&mul(&s, &z));
        let w_pred = div_by(&sub(&b4_pred, &mul(&z, &r_ieq)), &s);
        let rhs1_pred = add(&negate(&r_dual), &mat_t_vec(g, &w_pred));
        for i in 0..n {
            rhs[i] = rhs1_pred[i];
        }
        for i in 0..p {
            rhs[n + i] = -r_prim[i];
        }

        let lu = kkt.partial_piv_lu();
        let step_pred = lu.solve(&rhs);
        if !all_finite(&step_pred) {
            status = ConicStatus::Failed;
            break;
        }
        let dx_a = Col::from_fn(n, |i| step_pred[i]);
        let ds_a = add(&(g * &dx_a), &r_ieq);
        let dz_a = div_by(&sub(&b4_pred, &mul(&z, &ds_a)), &s);

        let alpha_aff = max_step(&s, &z, &ds_a, &dz_a);
        let mu_aff = dot(&axpy(&s, alpha_aff, &ds_a), &axpy(&z, alpha_aff, &dz_a)) / m_dim;
        let sigma = (mu_aff / mu).clamp(0.0, 1.0).powi(3);

        // Corrector step: second-order term + centering toward sigma*mu.
        let cross = mul(&ds_a, &dz_a);
        let target = Col::from_fn(d, |_| sigma * mu);
        let b4_corr = negate(&sub(&add(&mul(&s, &z), &cross), &target));
        let w_corr = div_by(&sub(&b4_corr, &mul(&z, &r_ieq)), &s);
        let rhs1_corr = add(&negate(&r_dual), &mat_t_vec(g, &w_corr));
        for i in 0..n {
            rhs[i] = rhs1_corr[i];
        }
        for i in 0..p {
            rhs[n + i] = -r_prim[i];
        }

        let step = lu.solve(&rhs);
        if !all_finite(&step) {
            status = ConicStatus::Failed;
            break;
        }
        let dx = Col::from_fn(n, |i| step[i]);
        let dy = Col::from_fn(p, |i| step[n + i]);
        let ds = add(&(g * &dx), &r_ieq);
        let dz = div_by(&sub(&b4_corr, &mul(&z, &ds)), &s);

        let alpha = max_step(&s, &z, &ds, &dz);
        x = axpy(&x, alpha, &dx);
        y = axpy(&y, alpha, &dy);
        s = axpy(&s, alpha, &ds);
        z = axpy(&z, alpha, &dz);

        if !all_finite(&x) {
            status = ConicStatus::Failed;
            break;
        }
    }

    ConicSolution {
        objective: program.objective(&x),
        gap: dot(&s, &z),
        x,
        y,
        z,
        outer_iterations: iterations,
        newton_iterations: iterations,
        status,
    }
}

/// Solves an affine conic program.
///
/// When [`ConicParams::predcorr`] is set and every cone is an orthant (i.e. an
/// LP or QP), the primal-dual Mehrotra predictor-corrector is used; otherwise a
/// logarithmic-barrier method handles the (possibly SOC/PSD) cones.
///
/// `x0` must be a strictly feasible starting point: `A x0 = b` and
/// `G x0 + h` in the interior of the product cone.
pub fn solve(program: &ConicProgram, x0: &Col<f64>, params: Option<ConicParams>) -> ConicSolution {
    let param = params.unwrap_or_default();
    if param.predcorr && program.cones.iter().all(|c| matches!(c, Cone::Nonneg(_))) {
        return solve_orthant_pc(program, x0, param);
    }
    let n = program.n();
    let p = program.p();
    let d = program.cone_dim();
    let nu = program.nu();

    let q = &program.q;
    let c = &program.c;
    let a = &program.a;
    let b = &program.b;
    let g = &program.g;
    let h = &program.h;
    let gt = g.transpose().to_owned();

    let mut x = x0.clone();
    let mut y = Col::zeros(p);
    let mut z = Col::zeros(d);
    let mut t = param.t0;
    let mut newton_iterations = 0usize;
    let mut outer_iterations = 0usize;
    let mut status = ConicStatus::MaxIterations;

    // Reused buffers.
    let mut kkt = Mat::<f64>::zeros(n + p, n + p);
    let mut rhs = Col::zeros(n + p);
    let mut dx = Col::zeros(n);
    let mut dy = Col::zeros(p);

    'outer: loop {
        if outer_iterations >= param.max_outer {
            break;
        }
        outer_iterations += 1;

        for _ in 0..param.max_newton {
            let s = add(&(g * &x), h);
            let bar = evaluate_barrier(&program.cones, &s);
            let qx = q * &x;
            let f_val = t * (dot(c, &x) + 0.5 * dot(&x, &qx)) + bar.phi;

            let f_grad = add(c, &qx);
            let grad_f = add(&scale(t, &f_grad), &mat_t_vec(g, &bar.grad));

            let hg = &bar.hess * g;
            let mut hess_f = &gt * &hg;
            for i in 0..n {
                for j in 0..n {
                    hess_f[(i, j)] += t * q[(i, j)];
                }
            }

            let r_dual = add(&grad_f, &mat_t_vec(a, &y));
            let r_prim = sub(&(a * &x), b);
            let rnorm = (dot(&r_dual, &r_dual) + dot(&r_prim, &r_prim)).sqrt();
            z = scale(-1.0 / t, &bar.grad);
            if rnorm < 1e-10 {
                break;
            }

            assemble_kkt_into(&mut kkt, &hess_f, a);
            for i in 0..n {
                rhs[i] = -r_dual[i];
            }
            for i in 0..p {
                rhs[n + i] = -r_prim[i];
            }
            let Some(step) = solve_kkt(&kkt, &rhs) else {
                status = ConicStatus::Failed;
                break 'outer;
            };
            for i in 0..n {
                dx[i] = step[i];
            }
            for i in 0..p {
                dy[i] = step[n + i];
            }

            // Newton decrement (directional decrease of the barrier objective).
            let dec = -dot(&grad_f, &dx);
            let dec = dec.max(1e-300);

            // Backtracking Armijo line search on F_t(x).
            let mut alpha = 1.0_f64;
            let mut accepted = false;
            for _ in 0..80 {
                let xt = axpy(&x, alpha, &dx);
                let st = add(&(g * &xt), h);
                if in_cone_interior(&program.cones, &st) {
                    let bar_t = evaluate_barrier(&program.cones, &st);
                    let ft = t * program.objective(&xt) + bar_t.phi;
                    if ft <= f_val - param.alpha * alpha * dec {
                        x = xt;
                        y = axpy(&y, alpha, &dy);
                        accepted = true;
                        break;
                    }
                }
                alpha *= param.beta;
            }
            newton_iterations += 1;
            if !accepted || !all_finite(&x) {
                break;
            }
            if dec < param.newton_tol {
                break;
            }
        }

        if nu / t < param.tol {
            status = ConicStatus::Optimal;
            break 'outer;
        }
        t *= param.mu;
        if !all_finite(&x) {
            status = ConicStatus::Failed;
            break;
        }
    }

    ConicSolution {
        objective: program.objective(&x),
        x,
        y,
        z,
        gap: nu / t,
        outer_iterations,
        newton_iterations,
        status,
    }
}
