use crate::utility::prelude::*;
use alloc::boxed::Box;

macro_rules! impl_fx {
    ($($(#[$meta:meta])* struct $ty:ident {
        box($($func:tt)+),
        fn($v:ident, $f:ident)($($expr:expr),+),
    })+) => {$(
        $(#[$meta])*
        pub struct $ty<'a, R: Fitness, const N: usize> {
            func: Box<dyn $($func)+ + Sync + Send + 'a>,
            bound: [[f64; 2]; N],
        }

        impl<'a, R: Fitness, const N: usize> $ty<'a, R, N> {
            /// Create objective function from a callable object.
            pub fn new<F>(f: F) -> Self
            where
                F: $($func)+ + Sync + Send + 'a,
            {
                Self {
                    func: Box::new(f),
                    bound: [[0.; 2]; N],
                }
            }

            impl_builders! {
                /// Set the bound.
                fn bound([[f64; 2]; N])
            }
        }

        impl<'a, R: Fitness, const N: usize> ObjFunc for $ty<'a, R, N> {
            type Result = R;
            type Fitness = R;

            fn fitness(&self, $v: &[f64], $f: f64) -> Self::Fitness {
                (self.func)($($expr),+)
            }

            fn result(&self, $v: &[f64]) -> Self::Result {
                self.fitness($v, 0.)
            }

            fn bound(&self) -> &[[f64; 2]] {
                &self.bound
            }
        }
    )+};
}

impl_fx! {
    /// A quick interface help to create objective function from a callable object.
    ///
    /// ```
    /// use metaheuristics_nature::{Fx, Rga, Solver};
    ///
    /// let f = Fx::new(|v| v[0] * v[0] + 8. * v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
    ///     .bound([[-50., 50.]; 4]);
    /// let s = Solver::build(Rga::default())
    ///     .task(|ctx| ctx.gen == 20)
    ///     .solve(f)
    /// .unwrap();
    /// ```
    ///
    /// Adaptive version is [`FxAdaptive`].
    struct Fx {
        box(Fn(&[f64]) -> R),
        fn(v, _f)(v),
    }

    /// A quick interface help to create adaptive objective function from a callable object.
    ///
    /// ```
    /// use metaheuristics_nature::{FxAdaptive, Rga, Solver};
    ///
    /// let f = FxAdaptive::new(|v, f| v[0] * v[0] + 8. * v[1] * v[1] + v[2] * v[2] + v[3] * v[3] * f)
    ///     .bound([[-50., 50.]; 4]);
    /// let s = Solver::build(Rga::default())
    ///     .task(|ctx| ctx.gen == 20)
    ///     .adaptive(|ctx| ctx.gen as f64)
    ///     .solve(f)
    /// .unwrap();
    /// ```
    ///
    /// Non-adaptive version is [`Fx`].
    struct FxAdaptive {
        box(Fn(&[f64], f64) -> R),
        fn(v, f)(v, f),
    }
}
