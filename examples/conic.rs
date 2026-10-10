//! LP / QP / SOCP / SDP via the conic solver.
//!
//! Run with `cargo run --example conic`.

use optim::ConicStatus;
use optim::conic::{Cone, ConicParams, ConicProgram};
use optim::problems::{random_lp, random_qp, random_sdp, random_socp};

fn manual_lp() {
    // min -x0 - 2 x1   s.t.  x0 + x1 = 1,  x0, x1 >= 0.
    let program = ConicProgram {
        q: faer::Mat::zeros(2, 2),
        c: faer::col![-1.0, -2.0],
        a: faer::mat![[1.0, 1.0]],
        b: faer::col![1.0],
        g: faer::mat![[1.0, 0.0], [0.0, 1.0]],
        h: faer::col![0.0, 0.0],
        cones: vec![Cone::Nonneg(2)],
    };
    let x0 = faer::col![0.5, 0.5]; // strictly feasible: Ax0 = b, x0 > 0
    let sol = optim::conic::solve(&program, &x0, None);
    println!(
        "manual LP:  x = [{:.4}, {:.4}]  objective = {:.4}",
        sol.x[0], sol.x[1], sol.objective
    );
}

fn main() {
    manual_lp();

    let instances = [
        random_lp(20, 10, 1),      // linear program
        random_qp(20, 10, 2),      // convex quadratic program
        random_socp(20, 10, 5, 3), // second-order cone program
        random_sdp(5, 5, 4),       // semidefinite program
    ];

    for inst in &instances {
        let sol = inst.solve(Some(ConicParams::default()));
        assert_eq!(sol.status, ConicStatus::Optimal, "{}", inst.name);
        println!(
            "{:<12} {:>3} iterations  objective = {:>12.6}  gap = {:.2e}",
            inst.name, sol.newton_iterations, sol.objective, sol.gap
        );
    }
}
