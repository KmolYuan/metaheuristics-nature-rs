#[cfg(feature = "std")]
extern crate std;

use crate::{utility::*, Array1, ObjFunc, Report, Task};
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
/// # use metaheuristics_nature::{ObjFunc, Array1, AsArray, Report};
/// # struct MyFunc([f64; 3], [f64; 3]);
/// # impl MyFunc {
/// #     fn new() -> Self { Self([0.; 3], [50.; 3]) }
/// # }
/// # impl ObjFunc for MyFunc {
/// #     type Result = f64;
/// #     fn fitness<'a, A>(&self, v: A, _: &Report) -> f64
/// #     where
/// #         A: AsArray<'a, f64>,
/// #     {
/// #         let v = v.into();
/// #         v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
/// #     }
/// #     fn result<'a, V>(&self, v: V) -> Self::Result
/// #     where
/// #         V: AsArray<'a, f64>
/// #     {
/// #         self.fitness(v, &Default::default())
/// #     }
/// #     fn ub(&self) -> &[f64] { &self.1 }
/// #     fn lb(&self) -> &[f64] { &self.0 }
/// # }
///
/// let a = Solver::solve(
///     MyFunc::new(),
///     Rga::default().task(Task::MinFit(1e-20)),
///     |_| true // Run without callback
/// );
/// // Get the result from objective function
/// let ans = a.result();
/// // Get the optimized XY value of your function
/// let (x, y) = a.parameters();
/// // Get the history reports
/// let history = a.history();
/// ```
pub struct Solver<M: Algorithm, F: ObjFunc> {
    ctx: Context<F>,
    method: M,
}

impl<M: Algorithm, F: ObjFunc> Solver<M, F> {
    /// Create the task and run the algorithm.
    ///
    /// Argument `callback` is a progress feedback function,
    /// returns true to keep algorithm running, same as the behavior of the while-loop.
    pub fn solve<S>(func: F, settings: S, callback: impl FnMut(Report) -> bool) -> Self
    where
        S: Setting<Algorithm = M>,
    {
        Self {
            ctx: Context::new(func, settings.base()),
            method: settings.create(),
        }
        .run(callback)
    }

    fn run(mut self, mut callback: impl FnMut(Report) -> bool) -> Self {
        #[cfg(feature = "std")]
        let time_start = Instant::now();
        self.ctx.init_pop();
        #[cfg(feature = "std")]
        {
            self.ctx.report.update_time(time_start);
        }
        self.method.init(&mut self.ctx);
        if !callback(self.ctx.report.clone()) {
            return self;
        }
        self.ctx.report();
        let mut last_diff = 0.;
        loop {
            let best_f = {
                self.ctx.report.next_gen();
                #[cfg(feature = "std")]
                {
                    self.ctx.report.update_time(time_start);
                }
                self.ctx.report.best_f
            };
            self.method.generation(&mut self.ctx);
            if self.ctx.report.gen % self.ctx.rpt == 0 {
                if !callback(self.ctx.report.clone()) {
                    break;
                }
                self.ctx.report();
            }
            match self.ctx.task {
                Task::MaxGen(v) => {
                    if self.ctx.report.gen >= v {
                        break;
                    }
                }
                Task::MinFit(v) => {
                    if self.ctx.report.best_f <= v {
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
                    let diff = best_f - self.ctx.report.best_f;
                    if last_diff > 0. && diff / last_diff >= v {
                        break;
                    }
                    last_diff = diff;
                }
            }
        }
        self
    }

    /// Get the history for plotting.
    #[inline(always)]
    pub fn history(&self) -> Vec<Report> {
        self.ctx.reports.clone()
    }

    /// Return the x and y of function.
    /// The algorithm must be executed once.
    #[inline(always)]
    pub fn parameters(&self) -> (Array1<f64>, f64) {
        (self.ctx.best.to_owned(), self.ctx.report.best_f)
    }

    /// Get the result of the objective function.
    #[inline(always)]
    pub fn result(&self) -> F::Result {
        self.ctx.func.result(&self.ctx.best)
    }
}
