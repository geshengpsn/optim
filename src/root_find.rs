//! Root finding for square systems of nonlinear equations.
//!
//! Rust port of `src/root_find.zig`. `find_root` solves `f(x) = 0` using damped
//! Newton steps with residual backtracking; `find_root_partial` solves `f = 0`
//! for a single field of a struct of vectors while the remaining fields stay
//! fixed.

use crate::linalg::{all_finite, axpy, col_eq, infinity_norm, negate, solve};
use faer::{Col, Mat};

/// Errors returned by the root finders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootError {
    /// A tolerance or damping parameter was outside its valid range.
    InvalidParameters,
    /// An input, residual, Jacobian, or step contained a non-finite value.
    NonFiniteValue,
    /// The Jacobian was singular (a zero row after scaling, or a failed solve).
    SingularJacobian,
    /// Backtracking failed to reduce the residual.
    LineSearchFailed,
    /// The iteration budget was exhausted before convergence.
    MaxIterationsExceeded,
}

impl core::fmt::Display for RootError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            RootError::InvalidParameters => "invalid parameters",
            RootError::NonFiniteValue => "non-finite value encountered",
            RootError::SingularJacobian => "singular Jacobian",
            RootError::LineSearchFailed => "line search failed",
            RootError::MaxIterationsExceeded => "maximum iterations exceeded",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for RootError {}

/// Tuning parameters for [`find_root`] and [`find_root_partial`].
#[derive(Debug, Clone, Copy)]
pub struct RootParams {
    /// Absolute tolerance for `max_i |f(x)[i]|`.
    pub e: f64,
    /// Required residual decrease, in `(0, 0.5)`.
    pub alpha: f64,
    /// Step reduction factor, in `(0, 1)`.
    pub beta: f64,
    /// Maximum Newton steps; zero only accepts an initially converged point.
    pub max_iterations: usize,
    /// Maximum step reductions per iteration; zero tries only the full step.
    pub max_backtracks: usize,
}

impl Default for RootParams {
    fn default() -> Self {
        Self {
            e: 1e-9,
            alpha: 0.4,
            beta: 0.5,
            max_iterations: 100,
            max_backtracks: 50,
        }
    }
}

/// Outcome of [`find_root`].
#[derive(Debug, Clone)]
pub struct RootResult {
    /// The approximate root.
    pub x: Col<f64>,
    /// Infinity norm of `f(x)`, at most the requested tolerance.
    pub residual_norm: f64,
    /// Number of accepted Newton steps.
    pub iterations: usize,
}

/// Outcome of [`find_root_partial`].
#[derive(Debug, Clone)]
pub struct PartialRootResult<Args> {
    /// The struct with the free field moved to the solution.
    pub args: Args,
    /// Infinity norm of the free residual at the solution.
    pub residual_norm: f64,
    /// Number of accepted Newton steps.
    pub iterations: usize,
}

fn validate(params: &RootParams) -> Result<(), RootError> {
    if !params.e.is_finite()
        || params.e <= 0.0
        || !(params.alpha > 0.0 && params.alpha < 0.5)
        || !(params.beta > 0.0 && params.beta < 1.0)
    {
        return Err(RootError::InvalidParameters);
    }
    Ok(())
}

/// Scales each row of `jac` (in place) by the infinity norm of that row and
/// applies the same scaling to `rhs`. This makes LU's absolute pivot threshold
/// relative to each equation's scale.
fn scale_rows(jac: &mut Mat<f64>, rhs: &mut Col<f64>) -> Result<(), RootError> {
    let n = jac.nrows();
    for i in 0..n {
        let mut scale = 0.0_f64;
        for j in 0..n {
            let v = jac[(i, j)];
            if !v.is_finite() {
                return Err(RootError::NonFiniteValue);
            }
            scale = scale.max(v.abs());
        }
        if scale == 0.0 {
            return Err(RootError::SingularJacobian);
        }
        for j in 0..n {
            jac[(i, j)] /= scale;
        }
        rhs[i] /= scale;
    }
    Ok(())
}

/// Solves `f(x) = 0` using damped Newton steps and residual backtracking.
///
/// `jac` must return `J[i, j] = d f[i] / d x[j]`. Each step solves
/// `J * step = -f(x)` with row scaling and faer's pivoted LU solver. Pass `None`
/// for default parameters. Convergence is local and depends on the initial
/// guess and Jacobian.
pub fn find_root(
    f: impl Fn(&Col<f64>) -> Col<f64>,
    jac: impl Fn(&Col<f64>) -> Mat<f64>,
    init_x: Col<f64>,
    params: Option<RootParams>,
) -> Result<RootResult, RootError> {
    let param = params.unwrap_or_default();
    validate(&param)?;

    let mut x = init_x;
    if !infinity_norm(&x).is_finite() {
        return Err(RootError::NonFiniteValue);
    }
    let mut residual = f(&x);
    let mut norm = infinity_norm(&residual);
    if !norm.is_finite() {
        return Err(RootError::NonFiniteValue);
    }

    let mut iterations = 0usize;
    while norm > param.e {
        if iterations == param.max_iterations {
            return Err(RootError::MaxIterationsExceeded);
        }
        let mut jacobian = jac(&x);
        let mut rhs = negate(&residual);
        scale_rows(&mut jacobian, &mut rhs)?;
        if !all_finite(&rhs) {
            return Err(RootError::NonFiniteValue);
        }
        let step = solve(&jacobian, &rhs).ok_or(RootError::SingularJacobian)?;
        if !all_finite(&step) {
            return Err(RootError::NonFiniteValue);
        }

        let mut t = 1.0_f64;
        let mut backtracks = 0usize;
        loop {
            let trial_x = axpy(&x, t, &step);
            if col_eq(&trial_x, &x) {
                return Err(RootError::LineSearchFailed);
            }
            if all_finite(&trial_x) {
                let trial_residual = f(&trial_x);
                let trial_norm = infinity_norm(&trial_residual);
                if trial_norm <= param.e
                    || (trial_norm < norm && trial_norm <= (1.0 - param.alpha * t) * norm)
                {
                    x = trial_x;
                    residual = trial_residual;
                    norm = trial_norm;
                    break;
                }
            }
            if backtracks == param.max_backtracks {
                return Err(RootError::LineSearchFailed);
            }
            t *= param.beta;
            backtracks += 1;
        }
        iterations += 1;
    }

    Ok(RootResult {
        x,
        residual_norm: norm,
        iterations,
    })
}

/// Solves `f(args) = 0` for one field of a struct of vectors while the remaining
/// fields stay fixed.
///
/// `f` takes the struct and returns the free vector, and `jac` returns its
/// square Jacobian. Because Rust has no reflection, the caller supplies
/// `get_free` to read the free field and `set_free` to write it. This mirrors
/// `findRootPartial` from the original Zig code.
///
/// Pass `None` for default parameters.
pub fn find_root_partial<Args: Clone>(
    f: impl Fn(&Args) -> Col<f64>,
    jac: impl Fn(&Args) -> Mat<f64>,
    init: Args,
    get_free: impl Fn(&Args) -> Col<f64>,
    set_free: impl Fn(&mut Args, Col<f64>),
    params: Option<RootParams>,
) -> Result<PartialRootResult<Args>, RootError> {
    let param = params.unwrap_or_default();
    validate(&param)?;

    let mut args = init;
    let mut free = get_free(&args);
    if !infinity_norm(&free).is_finite() {
        return Err(RootError::NonFiniteValue);
    }
    let mut residual = f(&args);
    let mut norm = infinity_norm(&residual);
    if !norm.is_finite() {
        return Err(RootError::NonFiniteValue);
    }

    let mut iterations = 0usize;
    while norm > param.e {
        if iterations == param.max_iterations {
            return Err(RootError::MaxIterationsExceeded);
        }
        let mut jacobian = jac(&args);
        let mut rhs = negate(&residual);
        scale_rows(&mut jacobian, &mut rhs)?;
        if !all_finite(&rhs) {
            return Err(RootError::NonFiniteValue);
        }
        let step = solve(&jacobian, &rhs).ok_or(RootError::SingularJacobian)?;
        if !all_finite(&step) {
            return Err(RootError::NonFiniteValue);
        }

        let mut t = 1.0_f64;
        let mut backtracks = 0usize;
        loop {
            let trial_free = axpy(&free, t, &step);
            if col_eq(&trial_free, &free) {
                return Err(RootError::LineSearchFailed);
            }
            let mut trial = args.clone();
            set_free(&mut trial, trial_free.clone());
            if all_finite(&trial_free) {
                let trial_residual = f(&trial);
                let trial_norm = infinity_norm(&trial_residual);
                if trial_norm <= param.e
                    || (trial_norm < norm && trial_norm <= (1.0 - param.alpha * t) * norm)
                {
                    args = trial;
                    free = trial_free;
                    residual = trial_residual;
                    norm = trial_norm;
                    break;
                }
            }
            if backtracks == param.max_backtracks {
                return Err(RootError::LineSearchFailed);
            }
            t *= param.beta;
            backtracks += 1;
        }
        iterations += 1;
    }

    Ok(PartialRootResult {
        args,
        residual_norm: norm,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::col;

    #[test]
    fn solves_a_nonlinear_system() {
        fn f(x: &Col<f64>) -> Col<f64> {
            col![x[0] * x[0] + x[1] - 5.0, 2.0 * x[0] - x[1] - 3.0]
        }
        fn jac(x: &Col<f64>) -> Mat<f64> {
            Mat::from_fn(2, 2, |i, j| match (i, j) {
                (0, 0) => 2.0 * x[0],
                (0, 1) => 1.0,
                (1, 0) => 2.0,
                _ => -1.0,
            })
        }
        let r = find_root(f, jac, col![1.0, 0.0], None).unwrap();
        assert!((r.x[0] - 2.0).abs() < 1e-9);
        assert!((r.x[1] - 1.0).abs() < 1e-9);
        assert!(r.residual_norm <= 1e-9);
        assert!(r.iterations > 0);
    }

    #[test]
    fn accepts_initial_root_without_jacobian() {
        fn f(x: &Col<f64>) -> Col<f64> {
            col![x[0] - 2.0]
        }
        fn jac(_: &Col<f64>) -> Mat<f64> {
            unreachable!()
        }
        let params = RootParams {
            max_iterations: 0,
            ..Default::default()
        };
        let r = find_root(f, jac, col![2.0], Some(params)).unwrap();
        assert_eq!(r.x[0], 2.0);
        assert_eq!(r.residual_norm, 0.0);
        assert_eq!(r.iterations, 0);
    }

    #[test]
    fn scales_equations_and_pivots() {
        let scale = 2.0_f64.powi(-400);
        let f = move |x: &Col<f64>| col![scale * (x[1] - 20.0), (x[0] - 10.0) / scale];
        let jac = move |_: &Col<f64>| {
            Mat::from_fn(2, 2, |i, j| match (i, j) {
                (0, 1) => scale,
                (1, 0) => 1.0 / scale,
                _ => 0.0,
            })
        };
        let r = find_root(f, jac, col![0.0, 0.0], None).unwrap();
        assert_eq!(r.x[0], 10.0);
        assert_eq!(r.x[1], 20.0);
        assert_eq!(r.residual_norm, 0.0);
        assert_eq!(r.iterations, 1);
    }

    #[test]
    fn backtracks_through_non_finite_trial_residuals() {
        fn f(x: &Col<f64>) -> Col<f64> {
            col![x[0].ln()]
        }
        fn jac(x: &Col<f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| 1.0 / x[0])
        }
        let p = RootParams {
            max_backtracks: 0,
            ..Default::default()
        };
        assert!(matches!(
            find_root(f, jac, col![10.0], Some(p)),
            Err(RootError::LineSearchFailed)
        ));
        let params = RootParams {
            e: 1e-12,
            ..Default::default()
        };
        let r = find_root(f, jac, col![10.0], Some(params)).unwrap();
        assert!((r.x[0] - 1.0).abs() < 1e-12);
        assert!(r.residual_norm <= 1e-12);
    }

    #[test]
    fn enforces_iteration_limit() {
        fn f(x: &Col<f64>) -> Col<f64> {
            col![x[0] - 2.0]
        }
        fn jac(_: &Col<f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| 1.0)
        }
        fn slow_jac(_: &Col<f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| 2.0)
        }
        let p0 = RootParams {
            max_iterations: 0,
            ..Default::default()
        };
        assert!(matches!(
            find_root(f, jac, col![0.0], Some(p0)),
            Err(RootError::MaxIterationsExceeded)
        ));
        let p1 = RootParams {
            max_iterations: 1,
            ..Default::default()
        };
        assert!(matches!(
            find_root(f, slow_jac, col![0.0], Some(p1)),
            Err(RootError::MaxIterationsExceeded)
        ));
        let p1b = RootParams {
            max_iterations: 1,
            max_backtracks: 0,
            ..Default::default()
        };
        let r = find_root(f, jac, col![0.0], Some(p1b)).unwrap();
        assert_eq!(r.x[0], 2.0);
        assert_eq!(r.iterations, 1);
    }

    #[test]
    fn reports_singular_jacobian() {
        fn f(x: &Col<f64>) -> Col<f64> {
            col![x[0] * x[0] - 2.0]
        }
        fn jac(x: &Col<f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| 2.0 * x[0])
        }
        assert!(matches!(
            find_root(f, jac, col![0.0], None),
            Err(RootError::SingularJacobian)
        ));
    }

    #[test]
    fn rejects_invalid_parameters() {
        fn f(_: &Col<f64>) -> Col<f64> {
            unreachable!()
        }
        fn jac(_: &Col<f64>) -> Mat<f64> {
            unreachable!()
        }
        let invalid = [
            RootParams {
                e: 0.0,
                ..Default::default()
            },
            RootParams {
                e: -1.0,
                ..Default::default()
            },
            RootParams {
                e: f64::INFINITY,
                ..Default::default()
            },
            RootParams {
                e: f64::NAN,
                ..Default::default()
            },
            RootParams {
                alpha: 0.0,
                ..Default::default()
            },
            RootParams {
                alpha: 0.5,
                ..Default::default()
            },
            RootParams {
                alpha: f64::NAN,
                ..Default::default()
            },
            RootParams {
                beta: 0.0,
                ..Default::default()
            },
            RootParams {
                beta: 1.0,
                ..Default::default()
            },
            RootParams {
                beta: f64::INFINITY,
                ..Default::default()
            },
            RootParams {
                beta: f64::NAN,
                ..Default::default()
            },
        ];
        for p in invalid {
            assert!(matches!(
                find_root(f, jac, col![0.0], Some(p)),
                Err(RootError::InvalidParameters)
            ));
        }
    }

    #[test]
    fn reports_failed_backtracking_and_stagnation() {
        fn f(_: &Col<f64>) -> Col<f64> {
            col![1.0]
        }
        fn jac(_: &Col<f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| 1.0)
        }
        let p = RootParams {
            max_backtracks: 3,
            ..Default::default()
        };
        assert!(matches!(
            find_root(f, jac, col![0.0], Some(p)),
            Err(RootError::LineSearchFailed)
        ));
        assert!(matches!(
            find_root(f, jac, col![1e30], None),
            Err(RootError::LineSearchFailed)
        ));
    }

    #[test]
    fn handles_large_and_small_residuals() {
        fn f(x: &Col<f64>) -> Col<f64> {
            col![x[0]]
        }
        fn jac(_: &Col<f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| 1.0)
        }
        for initial in [1e200_f64, 1e-200] {
            let params = RootParams {
                e: 1e-210,
                ..Default::default()
            };
            let r = find_root(f, jac, col![initial], Some(params)).unwrap();
            assert_eq!(r.x[0], 0.0);
            assert_eq!(r.iterations, 1);
        }
    }

    #[test]
    fn partial_holds_other_fields_fixed() {
        #[derive(Clone)]
        struct Args {
            x1: Col<f64>,
            x2: Col<f64>,
        }
        fn f(a: &Args) -> Col<f64> {
            col![a.x1[0] + a.x2[0] - 3.0]
        }
        fn jac(_: &Args) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| 1.0)
        }
        let init = Args {
            x1: col![1.0],
            x2: col![0.0],
        };
        let r = find_root_partial(f, jac, init, |a| a.x2.clone(), |a, v| a.x2 = v, None).unwrap();
        assert_eq!(r.args.x1[0], 1.0);
        assert!((r.args.x2[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn partial_solves_free_vector() {
        #[derive(Clone)]
        struct Args {
            x: Col<f64>,
            p: Col<f64>,
        }
        fn f(a: &Args) -> Col<f64> {
            col![a.x[0] + a.x[1] - a.p[0], a.x[0] - a.x[1]]
        }
        fn jac(_: &Args) -> Mat<f64> {
            Mat::from_fn(2, 2, |i, j| match (i, j) {
                (0, 0) | (1, 1) => 1.0,
                (0, 1) => 1.0,
                _ => -1.0,
            })
        }
        let init = Args {
            x: col![0.0, 0.0],
            p: col![4.0],
        };
        let r = find_root_partial(f, jac, init, |a| a.x.clone(), |a, v| a.x = v, None).unwrap();
        assert!((r.args.x[0] - 2.0).abs() < 1e-12);
        assert!((r.args.x[1] - 2.0).abs() < 1e-12);
        assert_eq!(r.args.p[0], 4.0);
    }

    #[test]
    fn partial_rejects_non_finite_fixed_field() {
        #[derive(Clone)]
        struct Args {
            x1: Col<f64>,
            x2: Col<f64>,
        }
        fn f(a: &Args) -> Col<f64> {
            col![a.x1[0] + a.x2[0] - 3.0]
        }
        fn jac(_: &Args) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| 1.0)
        }
        let init = Args {
            x1: col![f64::INFINITY],
            x2: col![0.0],
        };
        let result = find_root_partial(f, jac, init, |a| a.x2.clone(), |a, v| a.x2 = v, None);
        assert!(matches!(result, Err(RootError::NonFiniteValue)));
    }
}
