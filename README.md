# metaheuristics-nature

[![dependency status](https://deps.rs/repo/github/KmolYuan/metaheuristics-nature-rs/status.svg)](https://deps.rs/crate/metaheuristics-nature/)

A collection of nature-inspired metaheuristic algorithms. Provides objective function trait, well-known methods, and
tool functions let you implement your own searching method.

This crate implemented following algorithms:

+ Real-coded Genetic Algorithm (RGA)
+ Differential Evolution (DE)
+ Particle Swarm Optimization (PSO)
+ Firefly Algorithm (FA)
+ Teaching-Learning Based Optimization (TLBO)

Each algorithm gives same API and default parameters to help you test different implementation. For example, you can
test another algorithm by simply replacing `RGASettings` to `DESetting`.

```rust
use metaheuristics_nature::{Report, RgaSetting, Solver, Task};

let a = Solver::solve(
    MyFunc::new(),
    RgaSetting::default().task(Task::MinFit(1e-20)),
    | _ | true // Run without callback
);
let ans: f64 = a.result(); // Get the result from objective function
let (x, y): (Array1<f64>, f64) = a.parameters(); // Get the optimized XY value of your function
let history: Vec<Report> = a.history(); // Get the history reports
```
