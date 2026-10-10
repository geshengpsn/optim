# optim

基于 [`faer`](https://crates.io/crates/faer) 线性代数库的 Rust 数值优化库。它把
[`zoptim`](../zoptim) 的四类优化算法移植到 Rust，并额外提供一个统一的锥规划求解器
（LP / QP / SOCP / SDP）。

- 向量用 `faer::Col<f64>`，矩阵用 `faer::Mat<f64>`。
- 所有线性方程组都用 faer 的选主元 LU（`partial_piv_lu`）求解。

## 安装

```toml
[dependencies]
faer = { version = "0.24", default-features = false, features = ["std", "rayon"] }
optim = { path = "../optim" }
```

`rayon` 开启 faer 的多线程分解；本库不自行并行。可在运行时通过
`faer::{Par, set_global_parallelism}` 在单线程（`Par::Seq`）与多线程
（`Par::rayon(0)`）之间切换（见 [多线程（rayon）对比](#多线程rayon对比)）。

## 模块总览

| 模块 | 用途 |
|------|------|
| `optim::root_find` | 非线性方程组求根：`find_root`、`find_root_partial` |
| `optim::optim_noc` | 无约束优化：牛顿法 + Armijo 线搜索 |
| `optim::optim_eqc` | 等式约束优化：`min f(x) s.t. Ax=b` |
| `optim::optim_gc` | 一般约束优化：`min f(x) s.t. Ax=b, h(x)≤0` |
| `optim::conic` | 锥规划：`min cᵀx+½xᵀQx s.t. Ax=b, Gx+h∈K`（LP/QP/SOCP/SDP） |
| `optim::problems` | LP/QP/SOCP/SDP 基准问题生成器 |

## 快速开始

```rust
use faer::{Col, Mat, col, mat};
use optim::{find_root, RootError};

fn main() -> Result<(), RootError> {
    // 求解 x² - 2 = 0
    fn f(x: &Col<f64>) -> Col<f64> { col![x[0] * x[0] - 2.0] }
    fn jac(x: &Col<f64>) -> Mat<f64> { mat![[2.0 * x[0]]] }

    let r = find_root(f, jac, col![1.0], None)?;
    println!("sqrt(2) = {}", r.x[0]);
    Ok(())
}
```

---

## 1. 求根 `optim::root_find`

求解方阵非线性方程组 `f(x) = 0`：阻尼牛顿步 + 残差回溯线搜索。`jac` 返回
`J[i,j] = ∂f[i]/∂x[j]`。

```rust
use faer::{Col, Mat, col, mat};
use optim::{find_root, RootParams};

fn f(x: &Col<f64>) -> Col<f64> {
    col![x[0] * x[0] + x[1] - 5.0, 2.0 * x[0] - x[1] - 3.0]
}
fn jac(x: &Col<f64>) -> Mat<f64> {
    mat![
        [2.0 * x[0], 1.0],
        [2.0,        -1.0],
    ]
}

let result = find_root(f, jac, col![1.0, 0.0], None).unwrap();
println!("x = {:?}, residual = {}", result.x, result.residual_norm);
```

- 参数 `None` 使用默认值，或传 `Some(RootParams { e, alpha, beta, max_iterations, max_backtracks })`。
- 返回 `Result<RootResult, RootError>`；`RootResult { x, residual_norm, iterations }`。
- 错误：`InvalidParameters` / `NonFiniteValue` / `SingularJacobian` / `LineSearchFailed` / `MaxIterationsExceeded`。

### 固定某个字段求根 `find_root_partial`

只对结构体中的一个向量字段求根，其余字段保持不变。由于 Rust 没有反射，需要传入
读取/写入该字段的闭包：

```rust
#[derive(Clone)]
struct Args { x1: Col<f64>, x2: Col<f64> }

fn residual(a: &Args) -> Col<f64> { col![a.x1[0] + a.x2[0] - 3.0] }
fn dfree(_: &Args) -> Mat<f64> { mat![[1.0]] }

let init = Args { x1: col![1.0], x2: col![0.0] };
let solved = optim::find_root_partial(
    residual, dfree, init,
    |a| a.x2.clone(),      // 读取自由字段
    |a, v| a.x2 = v,       // 写回自由字段
    None,
).unwrap();
// solved.args.x2 ≈ [2.0]
```

---

## 2. 无约束优化 `optim::optim_noc`

`optimize_no_constraints(f, grad, hess, solver, init_x, e)`，迭代到牛顿减量
`-grad·dir/2 < e` 为止。`solver(&H, &grad)` 需返回牛顿方向（解 `H·dir = -grad`），
默认用 `noc_general_solver`。

```rust
use faer::{Col, Mat, col, mat};
use optim::{noc_general_solver, optimize_no_constraints};

let f    = |x: &Col<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
let grad = |x: &Col<f64>| col![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)];
let hess = |_: &Col<f64>| mat![[2.0, 0.0], [0.0, 2.0]];

let x = optimize_no_constraints(f, grad, hess, noc_general_solver, col![0.0, 0.0], 1e-10);
// x ≈ [1, 2]
```

自定义方向求解器只需提供 `Fn(&Mat<f64>, &Col<f64>) -> Col<f64>`。

---

## 3. 等式约束优化 `optim::optim_eqc`

求解 `min f(x) s.t. A x = b`。约束用 `EqConstraints { a, b }` 描述（`a` 为 `p×n`）。

```rust
use faer::{Col, Mat, col, mat};
use optim::{EqConstraints, EqParams, eqc_general_solver, optimize_eq_constraints};

fn f(x: &Col<f64>) -> f64 { 0.5 * (x[0] * (x[0] + 1.0) + 2.0 * x[1] * x[1]) }
fn df(x: &Col<f64>) -> Col<f64> { col![x[0] + 1.0, 2.0 * x[1]] }
fn d2f(_: &Col<f64>) -> Mat<f64> { mat![[1.0, 0.0], [0.0, 2.0]] }

let eq = EqConstraints { a: mat![[1.0, 1.0]], b: col![1.0] };
let r = optimize_eq_constraints(
    f, df, d2f, &eq, eqc_general_solver, col![0.0, 0.0],
    Some(EqParams { e: 1e-10, ..Default::default() }),
);
// r.x ≈ [1/3, 2/3], r.dual ≈ [-4/3]
```

- solver 签名：`Fn(&H, &A, &r_dual, &r_prim) -> (x_step, dual_step)`，默认 `eqc_general_solver`。
- 返回 `EqConstraintsResult { x, dual }`。

---

## 4. 一般约束优化 `optim::optim_gc`

求解 `min f(x) s.t. A x = b, h(x) ≤ 0`，采用原始-对偶内点法。不等式约束用
`InequalityConstraints { f, jac, hess_sum }` 描述：

- `f(x)` —— 返回约束值向量 `h(x)`（长度 `m`）；
- `jac(x)` —— 返回 `∇h(x)`（`m×n`）；
- `hess_sum(x, z)` —— 返回加权 Hessian 和 `Σ_i z_i ∇²h_i(x)`（`n×n`）。

```rust
use faer::{Col, Mat, col, mat};
use optim::{
    EqConstraints, GeneralConstraintsParams, InequalityConstraints, InitMode,
    gc_general_solver, optimize_general_constraints,
};

fn f(x: &Col<f64>) -> f64 { 0.5 * (x[0] * (x[0] + 1.0) + 2.0 * x[1] * x[1]) }
fn df(x: &Col<f64>) -> Col<f64> { col![x[0] + 1.0, 2.0 * x[1]] }
fn d2f(_: &Col<f64>) -> Mat<f64> { mat![[1.0, 0.0], [0.0, 2.0]] }

let eq = EqConstraints { a: mat![[1.0, 1.0]], b: col![1.0] };

let h        = |x: &Col<f64>| col![0.8 - x[0], -x[1]];       // 两道不等式
let h_jac    = |_: &Col<f64>| mat![[-1.0, 0.0], [0.0, -1.0]];
let h_hess   = |_: &Col<f64>, _: &Col<f64>| Mat::<f64>::zeros(2, 2);
let ieq = InequalityConstraints { f: h, jac: h_jac, hess_sum: h_hess };

let g = optimize_general_constraints(
    f, df, d2f, &eq, &ieq, gc_general_solver,
    col![0.0, 0.0],
    InitMode::Infeasible,     // 起点可不可行
    true,                     // 是否使用二阶校正步
    Some(GeneralConstraintsParams::default()),
);
// g.x ≈ [0.8, 0.2], g.ieq_dual ≈ [1.4, 0], g.eq_dual ≈ [-0.4]
```

- `InitMode::Feasible`：起点严格可行（`h(x0) < 0`）；`InitMode::Infeasible`：起点可越界。
- solver 签名较长的闭包，默认用 `gc_general_solver`。
- 返回 `GeneralConstraintsResult { x, ieq_dual, eq_dual }`。

---

## 5. 锥规划 `optim::conic`（LP / QP / SOCP / SDP）

统一求解

```text
min  cᵀx + ½ xᵀQx
s.t. A x = b
     G x + h ∈ K
```

其中锥 `K` 是若干锥的乘积：

| 锥 | 含义 | 典型问题 |
|----|------|----------|
| `Cone::Nonneg(d)` | 分量 `≥ 0` | LP、QP |
| `Cone::Soc(d)` | `s[0] ≥ ‖s[1..]‖₂` | SOCP |
| `Cone::Psd(k)` | `k×k` 对称矩阵 `⪰ 0`（`k(k+1)/2` 个坐标） | SDP |

求解器自动分派：全部为正交锥（LP/QP）时用**原始-对偶预估-校正**（约 10 次迭代）；
含二阶锥或半正定锥时用**对数障碍法**。

### 5.1 用内置问题集（推荐入门）

`optim::problems` 提供确定了随机种子、且有严格可行起点的基准实例：

```rust
use optim::problems::{random_lp, random_qp, random_socp, random_sdp};

let inst = random_qp(50, 25, 1);   // n=50, p=25, seed=1
let sol = inst.solve(None);        // None 用默认参数
println!("objective = {}, iters = {}", sol.objective, sol.newton_iterations);
```

可用的生成器：

| 函数 | 问题 | 参数 |
|------|------|------|
| `random_lp(n, p, seed)` | `min cᵀx s.t. Ax=b, 0≤x≤u` | 变量数、等式数、种子 |
| `random_qp(n, p, seed)` | 同上的强凸二次目标 | 同上 |
| `random_socp(n, d, p, seed)` | 一个 `d` 维二阶锥 + 盒约束 | 变量数、锥维、等式数、种子 |
| `random_sdp(k, p, seed)` | 标准型 SDP，`k×k` 矩阵 `+` 迹上界 | 矩阵阶、等式数、种子 |
| `suite()` | 上面四类的混合集合 | — |

### 5.2 自定义问题

直接构造 `ConicProgram`：

```rust
use faer::{Col, Mat, col, mat};
use optim::conic::{Cone, ConicProgram};

// min -x0 - 2 x1   s.t.  x0 + x1 = 1,  x0, x1 >= 0
let program = ConicProgram {
    q: Mat::zeros(2, 2),                     // 二次项，n×n（LP 用零矩阵）
    c: col![-1.0, -2.0],                     // 线性项，长度 n
    a: mat![[1.0, 1.0]],                     // 等式矩阵，p×n
    b: col![1.0],                            // 等式右端，长度 p
    g: mat![[1.0, 0.0], [0.0, 1.0]],         // 锥约束矩阵，d×n
    h: col![0.0, 0.0],                       // 锥约束偏移，长度 d
    cones: vec![Cone::Nonneg(2)],            // 锥 K
};

// x0 必须严格可行：A x0 = b 且 G x0 + h 在锥内部
let x0 = col![0.5, 0.5];
let sol = optim::conic::solve(&program, &x0, None);
println!("x = {:?}, objective = {}", sol.x, sol.objective); // x ≈ [0, 1], obj = -2
```

**构造可行起点的要点**：先用随机/手工方式取一个锥内部点 `x0`（例如 LP 取
`x0 > 0`、SDP 取 `X0 ≻ 0`），再据此反推等式右端 `b = A x0` 或锥偏移 `h`。内置
生成器正是这样做的。

### 返回结果

```rust
pub struct ConicSolution {
    pub x: Col<f64>,            // 原始解
    pub y: Col<f64>,            // 等式乘子
    pub z: Col<f64>,            // 锥（不等式）乘子
    pub objective: f64,         // 目标值
    pub gap: f64,               // 对偶间隙估计
    pub outer_iterations: usize,
    pub newton_iterations: usize,
    pub status: ConicStatus,    // Optimal / MaxIterations / Failed
}
```

---

## 6. 参数默认值速查

| 类型 | 默认值 |
|------|--------|
| `RootParams` | `e=1e-9, alpha=0.4, beta=0.5, max_iterations=100, max_backtracks=50` |
| `EqParams` | `e=1e-9, alpha=0.4, beta=0.5` |
| `GeneralConstraintsParams` | `e_feasible=1e-9, e_gap=1e-9, miu=10, alpha=0.1, beta=0.5` |
| `ConicParams` | `tol=1e-8, t0=1, mu=500, max_outer=60, max_newton=20, alpha=0.01, beta=0.5, newton_tol=1e-12, predcorr=true` |

`ConicParams::predcorr` 控制正交锥问题是否使用预估-校正；设为 `false` 可强制走障碍法。

---

## 运行示例 / 测试 / 基准

```sh
# 可运行示例
cargo run --example root_find
cargo run --example unconstrained
cargo run --example constrained
cargo run --example conic

# 单元测试
cargo test

# 基准：默认给出单/多线程对比表
cargo run --release --bin bench
cargo run --release --bin bench -- 3 1 seq   # 仅单线程（参数：reps predcorr threads）
cargo run --release --bin bench -- 3 0 par   # 障碍法 + 多线程

# criterion 微基准（conic_seq / conic_par 两组）
cargo bench
```

用 `seq` 模式可得到单张明细表（节选）：

```
instance        n     p status   iters    objective   time(ms)
lp-100x50     100    50    opt      12   -51.384742      3.765
qp-100x50     100    50    opt      11   960.104794      2.984
socp-100d50   100    25    opt      58   -65.263816     18.457
sdp-12x12      78    12    opt      61  -153.932829     14.627
```

## 多线程（rayon）对比

`bench` 与 criterion 都会分别在 **单线程**（`Par::Seq`）与 **多线程**
（`Par::rayon(0)`，用满所有核心）下运行同一套问题并给出加速比：

```sh
cargo run --release --bin bench            # 默认：单线程 vs 多线程对比
cargo run --release --bin bench -- 3 1 seq # 仅单线程
cargo run --release --bin bench -- 3 1 par # 仅多线程
cargo bench                                # criterion：conic_seq / conic_par 两组
```

实测（8 线程，best-of-3）：

```
instance           n     p   iters     seq(ms)     par(ms)   speedup
lp-100x50        100    50      12       3.500       4.597     0.76x
socp-100d50      100    25      58      18.689      24.778     0.75x
sdp-12x12         78    12      61      14.777      14.720     1.00x
lp-300x150       300   150      13      61.762      52.456     1.18x
qp-300x150       300   150      14      66.735      54.346     1.23x
lp-600x300       600   300      14     483.032     257.407     1.88x
qp-600x300       600   300      17     595.472     317.183     1.88x
----------------------------------------------------------------------
TOTAL                                 1255.746     738.384     1.70x
```

**结论**：

- **中大规模问题**（`n ≥ 300`）rayon 有正收益，最大的 `n=600` 达到约 **1.9×**；
- **小规模问题**（`n ≤ 100`）以及 SDP 反而略慢（0.75–1.0×），因为线程调度与同步
  开销超过计算量，faer 对过小的矩阵也会退化为串行；
- 因此默认在**需要时**开启：对中小问题用单线程，对大规模问题再切到多线程。

切换代码：

```rust
use faer::{Par, set_global_parallelism};

set_global_parallelism(Par::Seq);        // 单线程
set_global_parallelism(Par::rayon(0));   // 多线程，用满全部核心
// set_global_parallelism(Par::rayon(4)); // 或指定线程数
```

## 备注

- **维度是运行期的**（faer 的动态矩阵），不再有编译期 `N/M/P` 泛型。因此
  `optimize_no_constraints` 需要显式初值 `init_x: Col<f64>`。
- **`find_root_partial`** 用 `get_free`/`set_free` 闭包代替 Zig 的编译期反射。
- 全程使用 `f64`；如需 `f32`/泛型浮点可再扩展。
- 默认启用 faer 的 `rayon` feature；是否真正多线程由 faer 的全局并行设置在运行时决定
  （`set_global_parallelism`），对大规模问题才建议开启。
