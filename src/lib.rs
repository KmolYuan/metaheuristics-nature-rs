//! A collection of nature-inspired meta-heuristic algorithms.
//!
//! # Algorithm
//!
//! There are two traits [`Algorithm`](crate::utility::Algorithm) and
//! [`Setting`](crate::utility::Setting) ([`setting!`]).
//! The previous is used to design the optimization method,
//! and the latter is the setting interface.
//!
//! [`Solver`] is a simple interface for obtaining the solution, or analyzing the result.
//! This type allows you to use the API without importing any traits.
//!
//! All provided methods are listed in the module [`methods`].
//!
//! # Objective Function
//!
//! You can define your question as a objective function through implementing [`ObjFunc`].
//!
//! First of all, the array types are [`ndarray::ArrayBase`].
//! And then you should define the upper bound, lower bound, and objective function [`ObjFunc::fitness`] by yourself.
//!
//! The final answer is [`ObjFunc::result`], which is generated from the design parameters.
//!
//! # Random Function
//!
//! This crate use 32bit PRNG algorithm to generate random value, before that,
//! a random seed is required.
//! The seed is generated by `getrandom`, please see its [support platform](getrandom#supported-targets).
//!
//! # Features
//!
//! + `std`: Default feature. Enable standard library function, such as timing and threading.
//! + `parallel`: Enable parallel function, let objective function running without ordered,
//!   uses [`std::thread::spawn`].
//!   Disable it for the platform that doesn't supported threading,
//!   or if your objective function is not complicate enough.
//!   This feature required `std`.
//! + `wasm`: Support for webassembly, especial for random seed generating.
#![cfg_attr(doc_cfg, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
extern crate alloc;
#[cfg(not(feature = "std"))]
extern crate core as std;

pub use crate::methods::*;
pub use crate::obj_func::ObjFunc;
pub use crate::report::*;
pub use crate::solver::Solver;
pub use crate::task::Task;

/// Define a data structure and its builder functions.
///
/// Use `@` to denote the base settings, such as population number, task category
/// or reporting interval.
///
/// ```
/// use metaheuristics_nature::setting;
///
/// setting! {
///     /// Genetic Algorithm settings.
///     pub struct Ga {
///         @base,
///         @pop_num = 500,
///         cross: f64 = 0.95,
///         mutate: f64 = 0.05,
///         win: f64 = 0.95,
///         delta: f64 = 5.,
///     }
/// }
/// let s = Ga::default().pop_num(300).cross(0.9);
/// ```
///
/// This macro is not necessary, you still can use literal syntax directly.
///
/// ```
/// use metaheuristics_nature::utility::BasicSetting;
///
/// #[derive(Default)]
/// pub struct MyAlgorithm {
///     base: BasicSetting,
///     field: f64,
/// }
///
/// let setting = MyAlgorithm {
///     field: 20.,
///     base: BasicSetting {
///         pop_num: 300,
///         ..Default::default()
///     }
/// };
/// ```
///
/// Please aware that [`Setting`](crate::utility::Setting) trait still needs to implement.
#[macro_export]
macro_rules! setting {
    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident {
            $(@$base:ident, $(@$base_field:ident = $base_default:literal,)*)?
            $($(#[$field_attr:meta])* $v:vis $field:ident: $field_ty:ty = $field_default:expr),* $(,)?
        }
    ) => {
        $(#[$attr])*
        $vis struct $name {
            $($base: $crate::utility::BasicSetting,)?
            $($(#[$field_attr])* $v $field: $field_ty,)*
        }
        impl $name {
            $($crate::setting! { @$base })?
            $($(#[$field_attr])* pub fn $field(mut self, $field: $field_ty) -> Self {
                self.$field = $field;
                self
            })*
        }
        impl Default for $name {
            fn default() -> Self {
                Self {
                    $($base: $crate::utility::BasicSetting::default()$(.$base_field($base_default))*,)?
                    $($field: $field_default,)*
                }
            }
        }
    };
    ($(#[$attr:meta])* $vis:vis struct $name:ident(@base);) => {
        $(#[$attr])*
        #[derive(Default)]
        $vis struct $name($crate::utility::BasicSetting);
        impl $name {
            $crate::setting! { @0 }
        }
    };
    (@$base:tt) => {
        $crate::setting! {
            @$base,
            /// Termination condition.
            task: $crate::Task,
            /// Population number.
            pop_num: usize,
            /// The report frequency. (per generation)
            rpt: u32,
        }
    };
    (@$base:tt, $($(#[$field_attr:meta])* $field:ident: $field_type:ty),+ $(,)?) => {
        $($(#[$field_attr])* pub fn $field(mut self, $field: $field_type) -> Self {
            self.$base = self.$base.$field($field);
            self
        })+
    };
}

pub mod methods;
mod obj_func;
pub mod random;
mod report;
mod solver;
mod task;
#[cfg(test)]
mod tests;
pub mod thread_pool;
pub mod utility;
