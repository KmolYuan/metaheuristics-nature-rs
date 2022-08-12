use crate::utility::prelude::*;
use alloc::boxed::Box;

macro_rules! impl_fx {
    ($($(#[$meta:meta])* struct $ty:ident {
        box($($func:tt)+),
        fn($v:ident, $f:ident)($($expr:expr),+),
    })+) => {$(
        $(#[$meta])*
        pub struct $ty<'f, 'b, R: Fitness, const N: usize> {
            func: Box<dyn $($func)+ + Sync + Send + 'f>,
            bound: &'b [[f64; 2]; N],
        }

        impl<'f, 'b, R: Fitness, const N: usize> $ty<'f, 'b, R, N> {
            /// Create objective function from a callable object.
            pub fn new<F>(bound: &'b [[f64; 2]; N], f: F) -> Self
            where
                F: $($func)+ + Sync + Send + 'f,
            {
                Self {
                    func: Box::new(f),
                    bound
                }
            }
        }

        impl<R: Fitness, const N: usize> ObjFunc for $ty<'_, '_, R, N> {
            type Result = R;
            type Fitness = R;

            fn fitness(&self, $v: &[f64], $f: f64) -> Self::Fitness {
                let $v = <[f64; N]>::try_from($v).unwrap();
                (self.func)($($expr),+)
            }

            fn result(&self, $v: &[f64]) -> Self::Result {
                self.fitness($v, 0.)
            }

            fn bound(&self) -> &[[f64; 2]] {
                self.bound
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
    /// let bound = [[-50., 50.]; 4];
    /// let f = Fx::new(&bound, |[a, b, c, d]| a * a + 8. * b * b + c * c + d * d);
    /// let s = Solver::build(Rga::default())
    ///     .task(|ctx| ctx.gen == 20)
    ///     .solve(f)
    ///     .unwrap();
    /// ```
    ///
    /// Adaptive version is [`FxAdaptive`].
    struct Fx {
        box(Fn([f64; N]) -> R),
        fn(v, _f)(v),
    }

    /// A quick interface help to create adaptive objective function from a callable object.
    ///
    /// ```
    /// use metaheuristics_nature::{FxAdaptive, Rga, Solver};
    ///
    /// let bound = [[-50., 50.]; 4];
    /// let f = FxAdaptive::new(&bound, |[a, b, c, d], f| a * a + 8. * b * b + c * c + d * d * f);
    /// let s = Solver::build(Rga::default())
    ///     .task(|ctx| ctx.gen == 20)
    ///     .adaptive(|ctx| ctx.gen as f64)
    ///     .solve(f)
    ///     .unwrap();
    /// ```
    ///
    /// Non-adaptive version is [`Fx`].
    struct FxAdaptive {
        box(Fn([f64; N], f64) -> R),
        fn(v, f)(v, f),
    }
}
