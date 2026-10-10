//! Root finding: `find_root` and `find_root_partial`.
//!
//! Run with `cargo run --example root_find`.

use faer::{Col, Mat, col, mat};
use optim::{RootError, find_root, find_root_partial};

fn main() -> Result<(), RootError> {
    // Solve x² - 2 = 0.
    fn f(x: &Col<f64>) -> Col<f64> {
        col![x[0] * x[0] - 2.0]
    }
    fn jac(x: &Col<f64>) -> Mat<f64> {
        mat![[2.0 * x[0]]]
    }

    let result = find_root(f, jac, col![1.0], None)?;
    println!(
        "sqrt(2) = {:.12}  ({} iterations, residual {:.2e})",
        result.x[0], result.iterations, result.residual_norm
    );

    // Solve for one field of a struct while the other stays fixed:
    //   x1 + x2 - 3 = 0, with x1 = 1  →  x2 = 2.
    #[derive(Clone)]
    struct Args {
        x1: Col<f64>,
        x2: Col<f64>,
    }
    fn residual(a: &Args) -> Col<f64> {
        col![a.x1[0] + a.x2[0] - 3.0]
    }
    fn dfree(_: &Args) -> Mat<f64> {
        mat![[1.0]]
    }

    let init = Args {
        x1: col![1.0],
        x2: col![0.0],
    };
    let solved = find_root_partial(
        residual,
        dfree,
        init,
        |a| a.x2.clone(),
        |a, v| a.x2 = v,
        None,
    )?;
    println!("x2 = {}", solved.args.x2[0]);

    Ok(())
}
