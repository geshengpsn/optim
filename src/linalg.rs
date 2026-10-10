//! Linear-algebra helpers built on top of [`faer`].
//!
//! Both vectors and matrices use faer's own vocabulary types: column vectors are
//! [`faer::Col<f64>`] and matrices are [`faer::Mat<f64>`]. Every linear solve
//! goes through faer's pivoted LU decomposition.

use faer::linalg::solvers::Solve;
use faer::{Col, Mat};

/// A dynamically sized column vector of `f64`.
pub type VecF = Col<f64>;

/// A dynamically sized dense matrix of `f64`.
pub type DMat = Mat<f64>;

/// Builds a column vector from a slice.
pub fn col_from_slice(s: &[f64]) -> Col<f64> {
    Col::from_fn(s.len(), |i| s[i])
}

/// Dot product `a · b`.
pub fn dot(a: &Col<f64>, b: &Col<f64>) -> f64 {
    debug_assert_eq!(a.nrows(), b.nrows());
    let mut s = 0.0;
    for i in 0..a.nrows() {
        s += a[i] * b[i];
    }
    s
}

/// Squared Euclidean norm `a · a`.
pub fn norm_sq(a: &Col<f64>) -> f64 {
    dot(a, a)
}

/// Euclidean norm `‖a‖₂`.
pub fn norm2(a: &Col<f64>) -> f64 {
    norm_sq(a).sqrt()
}

/// Element-wise sum `a + b`.
pub fn add(a: &Col<f64>, b: &Col<f64>) -> Col<f64> {
    a + b
}

/// Element-wise difference `a - b`.
pub fn sub(a: &Col<f64>, b: &Col<f64>) -> Col<f64> {
    a - b
}

/// Element-wise product `a * b`.
pub fn mul(a: &Col<f64>, b: &Col<f64>) -> Col<f64> {
    debug_assert_eq!(a.nrows(), b.nrows());
    Col::from_fn(a.nrows(), |i| a[i] * b[i])
}

/// Scalar product `s * a`.
pub fn scale(s: f64, a: &Col<f64>) -> Col<f64> {
    a * s
}

/// Negation `-a`.
pub fn negate(a: &Col<f64>) -> Col<f64> {
    -a
}

/// Returns `x + t * y`.
pub fn axpy(x: &Col<f64>, t: f64, y: &Col<f64>) -> Col<f64> {
    debug_assert_eq!(x.nrows(), y.nrows());
    Col::from_fn(x.nrows(), |i| x[i] + t * y[i])
}

/// Returns `true` when every component is finite.
pub fn all_finite(a: &Col<f64>) -> bool {
    a.iter().all(|v| v.is_finite())
}

/// Infinity norm `max_i |a_i|`, returning [`f64::INFINITY`] if any component is
/// not finite.
pub fn infinity_norm(a: &Col<f64>) -> f64 {
    let mut m = 0.0_f64;
    for &v in a.iter() {
        if !v.is_finite() {
            return f64::INFINITY;
        }
        m = m.max(v.abs());
    }
    m
}

/// Component-wise equality.
pub fn col_eq(a: &Col<f64>, b: &Col<f64>) -> bool {
    a.nrows() == b.nrows() && (0..a.nrows()).all(|i| a[i] == b[i])
}

/* ----------------------------- matrix helpers ----------------------------- */

/// Matrix-vector product `A * x`.
pub fn mat_vec(a: &Mat<f64>, x: &Col<f64>) -> Col<f64> {
    a * x
}

/// Matrix-vector product `Aᵀ * x`.
pub fn mat_t_vec(a: &Mat<f64>, x: &Col<f64>) -> Col<f64> {
    a.transpose() * x
}

/// Solves the square system `A * x = b` with pivoted LU.
///
/// Returns `None` when the shapes are inconsistent, the matrix is singular, or
/// the computed solution contains non-finite values.
pub fn solve(a: &Mat<f64>, b: &Col<f64>) -> Option<Col<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.nrows() {
        return None;
    }
    if n == 0 {
        return Some(Col::zeros(0));
    }
    let sol = a.partial_piv_lu().solve(b);
    if !all_finite(&sol) {
        return None;
    }
    Some(sol)
}

/// Matrix-matrix sum `A + B`.
pub fn mat_add(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::col;
    use faer::mat;

    #[test]
    fn vector_ops() {
        let a = col![1.0, 2.0, 3.0];
        let b = col![4.0, 5.0, 6.0];
        assert_eq!(dot(&a, &b), 32.0);
        assert_eq!(add(&a, &b), col![5.0, 7.0, 9.0]);
        assert_eq!(sub(&b, &a), col![3.0, 3.0, 3.0]);
        assert_eq!(axpy(&a, 2.0, &b), col![9.0, 12.0, 15.0]);
        assert_eq!(norm_sq(&col![3.0, 4.0]), 25.0);
        assert_eq!(infinity_norm(&col![-3.0, 4.0]), 4.0);
        assert!(all_finite(&col![1.0, 2.0]));
        assert!(!all_finite(&col![1.0, f64::NAN]));
        assert_eq!(negate(&a), col![-1.0, -2.0, -3.0]);
        assert_eq!(mul(&a, &b), col![4.0, 10.0, 18.0]);
        assert_eq!(scale(2.0, &a), col![2.0, 4.0, 6.0]);
    }

    #[test]
    fn col_equality() {
        assert!(col_eq(&col![1.0, 2.0], &col![1.0, 2.0]));
        assert!(!col_eq(&col![1.0, 2.0], &col![1.0, 3.0]));
        assert!(!col_eq(&col![1.0], &col![1.0, 2.0]));
    }

    #[test]
    fn matrix_products_and_solve() {
        let a = mat![[2.0, 1.0], [1.0, 3.0]];
        let x = col![1.0, 2.0];
        assert_eq!(mat_vec(&a, &x), col![4.0, 7.0]);
        let sol = solve(&a, &col![1.0, 2.0]).unwrap();
        assert!((sol[0] - 0.2).abs() < 1e-12);
        assert!((sol[1] - 0.6).abs() < 1e-12);
    }
}
