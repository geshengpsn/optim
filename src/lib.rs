//! A Rust port of the [`zoptim`](../zoptim) optimization library, built on the
//! [`faer`] linear algebra library.
//!
//! The crate mirrors the four original Zig modules:
//!
//! * [`root_find`] — damped Newton root finding for systems and for a single
//!   free field of a struct of vectors.
//! * [`optim_noc`] — unconstrained Newton minimization with Armijo line search.
//! * [`optim_eqc`] — equality-constrained minimization (`A x = b`).
//! * [`optim_gc`] — general constrained minimization with equality and
//!   inequality constraints, using a primal-dual interior-point method.
//!
//! On top of the port it adds:
//!
//! * [`conic`] — a logarithmic-barrier solver for affine conic programs
//!   covering LP, QP, SOCP, and SDP through orthant, second-order, and PSD
//!   cones.
//! * [`problems`] — deterministic generators for LP / QP / SOCP / SDP
//!   benchmark instances with strictly feasible starting points.
//!
//! Both vectors and matrices use faer's vocabulary types: [`faer::Col<f64>`]
//! for vectors and [`faer::Mat<f64>`] for matrices. Every linear solve goes
//! through faer's pivoted LU decomposition.

pub mod conic;
pub mod linalg;
pub mod optim_eqc;
pub mod optim_gc;
pub mod optim_noc;
pub mod problems;
pub mod root_find;

pub use conic::{Cone, ConicParams, ConicProgram, ConicSolution, ConicStatus, in_cone_interior};
pub use linalg::{DMat, VecF, col_from_slice, mat_add, mat_t_vec, mat_vec, solve};
pub use optim_eqc::{
    EqConstraints, EqConstraintsResult, EqParams, eqc_general_solver, optimize_eq_constraints,
};
pub use optim_gc::{
    GeneralConstraintsParams, GeneralConstraintsResult, InequalityConstraints, InitMode,
    gc_general_solver, optimize_general_constraints,
};
pub use optim_noc::{noc_general_solver, optimize_no_constraints};
pub use problems::{
    Instance, ProblemKind, Rng, random_lp, random_qp, random_sdp, random_socp, suite,
};
pub use root_find::{
    PartialRootResult, RootError, RootParams, RootResult, find_root, find_root_partial,
};

#[cfg(test)]
mod tests {
    use super::*;
    use faer::col;

    #[test]
    fn public_root_finding_interface() {
        fn f(x: &faer::Col<f64>) -> faer::Col<f64> {
            col![x[0] * x[0] - 2.0]
        }
        fn jac(x: &faer::Col<f64>) -> faer::Mat<f64> {
            faer::Mat::from_fn(1, 1, |_, _| 2.0 * x[0])
        }
        let params = RootParams {
            e: 1e-12,
            ..Default::default()
        };
        let result = find_root(f, jac, col![1.0], Some(params)).unwrap();
        assert!((result.x[0] - 2.0_f64.sqrt()).abs() < params.e);
    }
}
