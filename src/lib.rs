//! A collection of nature-inspired metaheuristic algorithms.
//! ```
//! use metaheuristics_nature::{Report, RGA, RGASetting, Setting, Solver, Task, ObjFunc};
//! # use ndarray::{Array1, AsArray, ArrayView1};
//! # struct MyFunc(Array1<f64>, Array1<f64>);
//! # impl MyFunc {
//! #     fn new() -> Self { Self(Array1::zeros(3), Array1::ones(3) * 50.) }
//! # }
//! # impl ObjFunc for MyFunc {
//! #     type Result = f64;
//! #     fn fitness<'a, A>(&self, gen: u32, v: A) -> f64
//! #     where
//! #         A: AsArray<'a, f64>,
//! #     {
//! #         let v = v.into();
//! #         v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
//! #     }
//! #     fn result<'a, A>(&self, v: A) -> Self::Result
//! #     where
//! #         A: AsArray<'a, f64>
//! #     {
//! #         self.fitness(0, v)
//! #     }
//! #     fn ub(&self) -> ArrayView1<f64> { self.1.view() }
//! #     fn lb(&self) -> ArrayView1<f64> { self.0.view() }
//! # }
//! let mut a = RGA::new(
//!     MyFunc::new(),
//!     RGASetting::default().task(Task::MinFit(1e-20)),
//! );
//! let ans = a.run(|| {});  // Run without callback, and get the final result
//! let (x, y): (Array1<f64>, f64) = a.result();  // Get the optimized XY value of your function
//! let reports: Vec<Report> = a.history();  // Get the history reports
//! ```
pub use crate::de::*;
pub use crate::fa::*;
pub use crate::obj_func::*;
pub use crate::pso::*;
pub use crate::rga::*;
pub use crate::tlbo::*;
pub use crate::utility::*;

/// Generate random values between [0., 1.) or by range.
#[macro_export]
macro_rules! rand {
    ($lb:expr, $ub:expr) => {{
        use rand::Rng;
        rand::thread_rng().gen_range($lb..$ub)
    }};
    () => {
        rand!(0., 1.)
    };
}

/// Generate random boolean by positive factor.
#[macro_export]
macro_rules! maybe {
    ($v:expr) => {{
        use rand::Rng;
        rand::thread_rng().gen_bool($v)
    }};
}

/// Define a data structure and its builder functions.
///
/// Use `@` to denote the base settings, such as population number, task category
/// or reporting interval.
/// ```
/// use metaheuristics_nature::setting_builder;
///
/// setting_builder! {
///     /// Real-coded Genetic Algorithm settings.
///     pub struct GASetting {
///         @base,
///         @pop_num = 500,
///         cross: f64 = 0.95,
///         mutate: f64 = 0.05,
///         win: f64 = 0.95,
///         delta: f64 = 5.,
///     }
/// }
/// let s = GASetting::default().pop_num(300).cross(0.9);
/// ```
#[macro_export]
macro_rules! setting_builder {
    (
        $(#[$attr:meta])*
        $v:vis struct $name:ident {
            $(@$base:ident, $(@$base_field:ident = $base_default:expr,)*)?
            $($(#[$field_attr:meta])* $field:ident: $field_type:ty = $field_default:expr,)+
        }
    ) => {
        $(#[$attr])*
        $v struct $name {
            $($base: $crate::Setting,)?
            $($field: $field_type,)+
        }
        impl $name {
            $(setting_builder! {
                @$base,
                task: $crate::Task,
                pop_num: usize,
                rpt: u32,
            })?
            $($(#[$field_attr])* pub fn $field(mut self, $field: $field_type) -> Self {
                self.$field = $field;
                self
            })+
        }
        impl Default for $name {
            fn default() -> Self {
                Self {
                    $($base: $crate::Setting::default()$(.$base_field($base_default))*,)?
                    $($field: $field_default,)+
                }
            }
        }
    };
    (@$base:ident, $($field:ident: $field_type:ty,)+) => {
        $(pub fn $field(mut self, $field: $field_type) -> Self {
            self.$base = self.$base.$field($field);
            self
        })+
    }
}

mod de;
mod fa;
mod obj_func;
mod pso;
mod rga;
#[cfg(test)]
mod tests;
mod tlbo;
mod utility;
