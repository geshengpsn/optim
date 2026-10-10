//! Benchmark problem generators for LP, QP, SOCP, and SDP.
//!
//! Every instance is returned in the affine conic form consumed by
//! [`crate::conic`], together with a strictly feasible starting point:
//!
//! ```text
//! min   cᵀx + ½ xᵀQx
//! s.t.  A x = b
//!       G x + h ∈ K
//! ```
//!
//! * **LP** — box-constrained linear program (`Q = 0`, orthant cone).
//! * **QP** — strongly convex box-constrained quadratic program (orthant cone).
//! * **SOCP** — linear objective with a second-order cone plus a trace bound
//!   (product of a second-order cone and orthants).
//! * **SDP** — standard-form semidefinite program with a trace bound (PSD cone
//!   plus an orthant).
//!
//! The instances are constructed so their feasible sets are bounded and a
//! strictly feasible point is known, which guarantees the barrier method starts
//! well and the optimum is attained.

use crate::conic::{Cone, ConicParams, ConicProgram, ConicSolution, solve};
use faer::{Col, Mat};

/// Which family a benchmark instance belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProblemKind {
    /// Linear program.
    Lp,
    /// Convex quadratic program.
    Qp,
    /// Second-order cone program.
    Socp,
    /// Semidefinite program.
    Sdp,
}

/// A benchmark instance: a conic program plus a strictly feasible start.
#[derive(Clone)]
pub struct Instance {
    /// Human-readable name (includes family and size).
    pub name: String,
    /// Problem family.
    pub kind: ProblemKind,
    /// The conic program.
    pub program: ConicProgram,
    /// Strictly feasible starting point.
    pub x0: Col<f64>,
}

impl Instance {
    /// Solves the instance with the barrier method.
    pub fn solve(&self, params: Option<ConicParams>) -> ConicSolution {
        solve(&self.program, &self.x0, params)
    }
}

/// Deterministic SplitMix64-based generator (no external RNG dependency).
pub struct Rng(u64);

impl Rng {
    /// Creates a generator from a seed.
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform sample in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform sample in `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard normal sample (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Box-constrained QP/LP:
/// `min cᵀx + ½ xᵀQx  s.t.  A x = b, 0 ≤ x ≤ u`.
fn boxed_qp(
    name: String,
    kind: ProblemKind,
    n: usize,
    p: usize,
    seed: u64,
    quad: bool,
) -> Instance {
    let mut rng = Rng::new(seed);
    let u: Col<f64> = Col::from_fn(n, |_| rng.range(1.0, 2.0));
    let x0: Col<f64> = Col::from_fn(n, |i| 0.5 * u[i]);
    let a = Mat::from_fn(p, n, |_, _| rng.normal());
    let b = &a * &x0;
    let c = Col::from_fn(n, |_| rng.normal());
    let q = if quad {
        let m = Mat::from_fn(n, n, |_, _| rng.normal());
        let mt = m.transpose().to_owned();
        let mut q = &mt * &m;
        for i in 0..n {
            q[(i, i)] += 1.0;
        }
        q
    } else {
        Mat::zeros(n, n)
    };

    // s = [x; u - x] ∈ Nonneg(2n).
    let mut g = Mat::<f64>::zeros(2 * n, n);
    for i in 0..n {
        g[(i, i)] = 1.0;
        g[(n + i, i)] = -1.0;
    }
    let mut h = Col::zeros(2 * n);
    for i in 0..n {
        h[n + i] = u[i];
    }

    Instance {
        name,
        kind,
        program: ConicProgram {
            q,
            c,
            a,
            b,
            g,
            h,
            cones: vec![Cone::Nonneg(2 * n)],
        },
        x0,
    }
}

/// Linear program `min cᵀx  s.t.  A x = b, 0 ≤ x ≤ u`.
pub fn random_lp(n: usize, p: usize, seed: u64) -> Instance {
    boxed_qp(format!("lp-{n}x{p}"), ProblemKind::Lp, n, p, seed, false)
}

/// Convex QP `min cᵀx + ½ xᵀQx  s.t.  A x = b, 0 ≤ x ≤ u` with `Q ⪰ I`.
pub fn random_qp(n: usize, p: usize, seed: u64) -> Instance {
    boxed_qp(format!("qp-{n}x{p}"), ProblemKind::Qp, n, p, seed, true)
}

/// SOCP with one second-order cone and a box on the tail variables.
///
/// The decision vector is `x = [t, w, y]` with `(t, w) ∈ SOC(d)` (so
/// `t ≥ ‖w‖`), `t ≤ trace_bound`, and `0 ≤ y ≤ u`.
pub fn random_socp(n: usize, d: usize, p: usize, seed: u64) -> Instance {
    assert!(d >= 2 && d <= n);
    let mut rng = Rng::new(seed);
    let trace_bound = 5.0;
    let box_dim = n - d;
    let u: Col<f64> = Col::from_fn(box_dim, |_| rng.range(1.0, 2.0));

    // Strictly feasible start.
    let t0 = 0.5 * trace_bound;
    let mut w0 = Col::from_fn(d - 1, |_| rng.range(-0.2, 0.2));
    // Shrink w so that t0 > ‖w0‖ by a margin.
    let wn = (0..d - 1).map(|i| w0[i] * w0[i]).sum::<f64>().sqrt();
    if wn > 0.0 {
        let factor = 0.5 * t0 / wn;
        for i in 0..d - 1 {
            w0[i] *= factor;
        }
    }
    let x0 = Col::from_fn(n, |i| {
        if i == 0 {
            t0
        } else if i < d {
            w0[i - 1]
        } else {
            0.5 * u[i - d]
        }
    });

    let a = Mat::from_fn(p, n, |_, _| rng.normal());
    let b = &a * &x0;
    let c = Col::from_fn(n, |_| rng.normal());

    // s = [ (t, w) ; trace_bound - t ; y ; u - y ].
    let d_total = d + 1 + 2 * box_dim;
    let mut g = Mat::<f64>::zeros(d_total, n);
    let mut h = Col::zeros(d_total);
    for i in 0..d {
        g[(i, i)] = 1.0;
    }
    // trace bound: trace_bound - t >= 0.
    g[(d, 0)] = -1.0;
    h[d] = trace_bound;
    for i in 0..box_dim {
        g[(d + 1 + i, d + i)] = 1.0;
        g[(d + 1 + box_dim + i, d + i)] = -1.0;
        h[d + 1 + box_dim + i] = u[i];
    }

    let mut cones = vec![Cone::Soc(d), Cone::Nonneg(1)];
    if box_dim > 0 {
        cones.push(Cone::Nonneg(box_dim));
        cones.push(Cone::Nonneg(box_dim));
    }

    Instance {
        name: format!("socp-{n}d{d}"),
        kind: ProblemKind::Socp,
        program: ConicProgram {
            q: Mat::zeros(n, n),
            c,
            a,
            b,
            g,
            h,
            cones,
        },
        x0,
    }
}

/// Standard-form SDP `min ⟨C, X⟩  s.t.  ⟨Aᵢ, X⟩ = bᵢ, X ⪰ 0, tr(X) ≤ bound`.
///
/// `X` is symmetric `k × k`, stored with `k(k+1)/2` coordinates (lower triangle,
/// column-major).
pub fn random_sdp(k: usize, p: usize, seed: u64) -> Instance {
    let mut rng = Rng::new(seed);
    let m = k * (k + 1) / 2;
    let trace_bound = 5.0 * k as f64;

    // Strictly feasible start: X0 = L Lᵀ + I, then rescaled below the trace bound.
    let l = Mat::from_fn(k, k, |_, _| rng.normal());
    let lt = l.transpose().to_owned();
    let mut x_mat = &l * &lt;
    let tr0: f64 = (0..k).map(|i| x_mat[(i, i)]).sum();
    let shrink = (0.5 * trace_bound) / tr0.max(1e-12);
    for i in 0..k {
        for j in 0..k {
            x_mat[(i, j)] *= shrink;
        }
    }
    for i in 0..k {
        x_mat[(i, i)] += 0.1;
    }

    let x0 = sym_coords(k, &x_mat);

    // Equality rows: aᵢ · x = tr(Aᵢ X).
    let mut a = Mat::<f64>::zeros(p, m);
    for r in 0..p {
        let am = Mat::from_fn(k, k, |_, _| rng.normal());
        let coords = sym_coords_trace(k, &am);
        for j in 0..m {
            a[(r, j)] = coords[j];
        }
    }
    let b = &a * &x0;

    // Objective c · x = tr(C X).
    let cm = Mat::from_fn(k, k, |_, _| rng.range(-1.0, 1.0));
    let c = sym_coords_trace(k, &cm);

    // s = [X ; trace_bound - tr(X)] ∈ Psd(k) × Nonneg(1).
    let mut g = Mat::<f64>::zeros(m + 1, m);
    for i in 0..m {
        g[(i, i)] = 1.0;
    }
    for i in 0..k {
        g[(m, sym_index(i, i))] = -1.0;
    }
    let mut h = Col::zeros(m + 1);
    h[m] = trace_bound;

    Instance {
        name: format!("sdp-{k}x{p}"),
        kind: ProblemKind::Sdp,
        program: ConicProgram {
            q: Mat::zeros(m, m),
            c,
            a,
            b,
            g,
            h,
            cones: vec![Cone::Psd(k), Cone::Nonneg(1)],
        },
        x0,
    }
}

/// Lower-triangular column-major index of symmetric entry `(i, j)`, `i <= j`.
fn sym_index(i: usize, j: usize) -> usize {
    j * (j + 1) / 2 + i
}

/// Coordinates of a symmetric matrix without trace scaling.
fn sym_coords(k: usize, m: &Mat<f64>) -> Col<f64> {
    Col::from_fn(k * (k + 1) / 2, |idx| {
        // Decode the coordinate index.
        let mut j = 0;
        while sym_index(0, j + 1) <= idx {
            j += 1;
        }
        let i = idx - sym_index(0, j);
        m[(i, j)]
    })
}

/// Coordinates such that `coords · sym_coords(X) = tr(M X)`.
fn sym_coords_trace(k: usize, m: &Mat<f64>) -> Col<f64> {
    Col::from_fn(k * (k + 1) / 2, |idx| {
        let mut j = 0;
        while sym_index(0, j + 1) <= idx {
            j += 1;
        }
        let i = idx - sym_index(0, j);
        if i == j { m[(i, j)] } else { 2.0 * m[(i, j)] }
    })
}

/// The full benchmark suite: LP, QP, SOCP, and SDP instances of varying size.
pub fn suite() -> Vec<Instance> {
    let mut out = Vec::new();

    for &n in &[20usize, 50, 100] {
        out.push(random_lp(n, n / 2, 0x1000 + n as u64));
        out.push(random_qp(n, n / 2, 0x2000 + n as u64));
    }
    for &n in &[20usize, 50, 100] {
        let d = (n / 2).max(2);
        out.push(random_socp(n, d, n / 4, 0x3000 + n as u64));
    }
    for &k in &[4usize, 8, 12] {
        out.push(random_sdp(k, k, 0x4000 + k as u64));
    }

    out
}

/// A compact suite used by the correctness tests.
pub fn tiny_suite() -> Vec<Instance> {
    vec![
        random_lp(8, 3, 1),
        random_qp(8, 3, 2),
        random_socp(8, 4, 2, 3),
        random_sdp(3, 3, 4),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::{ConicStatus, in_cone_interior};

    #[test]
    fn starts_are_strictly_feasible() {
        for inst in tiny_suite() {
            let a = &inst.program.a;
            let b = &inst.program.b;
            let ax = a * &inst.x0;
            for i in 0..b.nrows() {
                assert!(
                    (ax[i] - b[i]).abs() < 1e-9,
                    "{}: equality violated",
                    inst.name
                );
            }
            let s = &(&inst.program.g * &inst.x0) + &inst.program.h;
            assert!(
                in_cone_interior(&inst.program.cones, &s),
                "{}: start not in cone interior",
                inst.name
            );
        }
    }

    #[test]
    fn solves_all_families() {
        for inst in tiny_suite() {
            let sol = inst.solve(None);
            assert_eq!(
                sol.status,
                ConicStatus::Optimal,
                "{} did not converge",
                inst.name
            );
            assert!(sol.objective.is_finite());
        }
    }
}
