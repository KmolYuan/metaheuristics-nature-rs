//! A collection of nature-inspired metaheuristic algorithms.
//! ```rust
//! use metaheuristics_nature::{Report, RGA, RGASetting, Setting, Solver, Task};
//!
//! fn main() {
//!     let mut a = RGA::new(
//!         MyObj::new(),
//!         RGASetting {
//!             base: Setting {
//!                 task: Task::MinFit(1e-20),
//!                 ..Default::default()
//!             },
//!             ..Default::default()
//!         },
//!     );
//!     let ans = a.run();  // Run and get the final result
//!     let (x, y): (Vec<f64>, f64) = a.result();  // Get the optimized XY value of your function
//!     let reports: Vec<Report> = a.history();
//! }
//! ```

pub use crate::{de::*, fa::*, pso::*, rga::*, tlbo::*, utility::*};

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

/// Make a multi-dimension array of the floating point zeros.
#[macro_export]
macro_rules! zeros {
    () => { 0. };
    ($w:expr $(, $h:expr)* $(,)?) => { vec![zeros!($($h,)*); $w] };
}

mod de;
mod fa;
mod pso;
mod rga;
#[cfg(test)]
mod tests;
mod tlbo;
mod utility;
