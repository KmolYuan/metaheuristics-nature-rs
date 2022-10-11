use crate::utility::prelude::*;
use alloc::boxed::Box;

macro_rules! impl_fx {
    ($($(#[$meta:meta])* struct $ty:ident {
        box($($func:tt)+),
        fn($v:ident)($($expr:expr),+),
    })+) => {$(
        $(#[$meta])*
        pub struct $ty<'f, 'b, R: Fitness, const DIM: usize> {
            func: Box<dyn $($func)+ + Sync + Send + 'f>,
            bound: &'b [[f64; 2]; DIM],
        }

        impl<'f, 'b, R: Fitness, const DIM: usize> $ty<'f, 'b, R, DIM> {
            /// Create objective function from a callable object.
            pub fn new<F>(bound: &'b [[f64; 2]; DIM], f: F) -> Self
            where
                F: $($func)+ + Sync + Send + 'f,
            {
                Self {
                    func: Box::new(f),
                    bound
                }
            }
        }

        impl<R: Fitness, const DIM: usize> Bounded for $ty<'_, '_, R, DIM> {
            fn bound(&self) -> &[[f64; 2]] {
                self.bound
            }
        }

        impl<R: Fitness, const DIM: usize> ObjFunc for $ty<'_, '_, R, DIM> {
            type Fitness = R;

            fn fitness(&self, $v: &[f64]) -> Self::Fitness {
                let $v = <[f64; DIM]>::try_from($v).unwrap();
                (self.func)($($expr),+)
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
    struct Fx {
        box(Fn([f64; DIM]) -> R),
        fn(v)(v),
    }
}
