#[cfg(feature = "std")]
extern crate std;

use crate::{utility::*, ObjFunc, Report, Task};
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::time::Instant;

/// A public API for using optimization methods.
///
/// Users can simply obtain their solution and see the result.
///
/// + The method is a type that implemented [`Algorithm`].
/// + The objective function is a type that implement [`ObjFunc`].
/// + A basic algorithm data is hold by [`Context`].
///
/// This type can infer the algorithm by [`Setting::Algorithm`].
///
/// ```
/// use metaheuristics_nature::{Rga, Solver, Task};
/// # use metaheuristics_nature::{ObjFunc, Report};
/// # struct MyFunc([f64; 3], [f64; 3]);
/// # impl MyFunc {
/// #     fn new() -> Self { Self([0.; 3], [50.; 3]) }
/// # }
/// # impl ObjFunc for MyFunc {
/// #     type Result = f64;
/// #     fn fitness(&self, v: &[f64], _: &Report) -> f64 {
/// #         v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
/// #     }
/// #     fn result(&self, v: &[f64]) -> Self::Result {
/// #         self.fitness(v, &Default::default())
/// #     }
/// #     fn ub(&self) -> &[f64] { &self.1 }
/// #     fn lb(&self) -> &[f64] { &self.0 }
/// # }
///
/// let a = Solver::solve(
///     MyFunc::new(),
///     Rga::default().task(Task::MinFit(1e-20)),
///     |_| true, // Run without callback
/// );
/// // Get the result from objective function
/// let ans = a.result();
/// // Get the optimized XY value of your function
/// let (x, y) = a.parameters();
/// // Get the history reports
/// let history = a.history();
/// ```
pub struct Solver<F: ObjFunc>(Context<F>);

impl<F: ObjFunc> Solver<F> {
    /// Create the task and run the algorithm, which may takes a lot of time.
    ///
    /// Argument `callback` is a progress feedback function,
    /// returns true to keep algorithm running, same as the behavior of the while-loop.
    pub fn solve<S, C>(func: F, setting: S, mut callback: C) -> Self
    where
        S: Setting,
        C: FnMut(&Report) -> bool,
    {
        let mut ctx = Context::new(func, setting.base());
        let mut method = setting.create();
        #[cfg(feature = "std")]
        let time_start = Instant::now();
        ctx.init_pop();
        #[cfg(feature = "std")]
        let _ = ctx.report.update_time(time_start);
        method.init(&mut ctx);
        if !callback(&ctx.report) {
            return Self(ctx);
        }
        ctx.report();
        loop {
            ctx.report.next_gen();
            #[cfg(feature = "std")]
            let _ = ctx.report.update_time(time_start);
            let best_f = ctx.report.best_f;
            let diff = ctx.report.diff;
            method.generation(&mut ctx);
            ctx.report.set_diff(best_f - ctx.report.best_f);
            if ctx.report.gen % ctx.rpt == 0 {
                if !callback(&ctx.report) {
                    break;
                }
                ctx.report();
            }
            match ctx.task {
                Task::MaxGen(v) => {
                    if ctx.report.gen >= v {
                        break;
                    }
                }
                Task::MinFit(v) => {
                    if ctx.report.best_f <= v {
                        break;
                    }
                }
                #[cfg(feature = "std")]
                Task::MaxTime(v) => {
                    if (Instant::now() - time_start).as_secs_f32() >= v {
                        break;
                    }
                }
                Task::SlowDown(v) => {
                    if ctx.report.diff / diff >= v {
                        break;
                    }
                }
            }
        }
        Self(ctx)
    }

    /// Get the history for plotting.
    #[inline(always)]
    pub fn history(&self) -> Vec<Report> {
        self.0.reports.clone()
    }

    /// Return the x and y of function.
    /// The algorithm must be executed once.
    #[inline(always)]
    pub fn parameters(&self) -> (&[f64], f64) {
        (self.0.best.as_slice().unwrap(), self.0.report.best_f)
    }

    /// Get the result of the objective function.
    #[inline(always)]
    pub fn result(&self) -> F::Result {
        self.0.func.result(self.0.best.as_slice().unwrap())
    }
}
