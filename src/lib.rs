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
mod tlbo;
mod utility;

#[cfg(test)]
mod tests {
    use crate::{ObjFunc, Solver};

    pub(crate) struct TestObj(Vec<f64>, Vec<f64>);

    impl TestObj {
        pub(crate) fn new() -> Self {
            Self(vec![0.; 4], vec![50.; 4])
        }
    }

    impl ObjFunc for TestObj {
        type Result = f64;
        fn fitness(&self, _gen: u32, v: &Vec<f64>) -> f64 {
            v[0] * v[0] + 8. * v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
        }
        fn result(&self, v: &Vec<f64>) -> f64 {
            self.fitness(0, v)
        }
        fn ub(&self) -> &Vec<f64> {
            &self.1
        }
        fn lb(&self) -> &Vec<f64> {
            &self.0
        }
    }

    pub(crate) fn test<F, S>(mut a: S)
    where
        F: ObjFunc<Result = f64>,
        S: Solver<F>,
    {
        let ans = a.run();
        let (x, y) = a.result();
        assert!(a.history().len() > 0);
        assert!(ans.abs() < 1e-20, "{}", ans);
        for i in 0..4 {
            assert!(x[i].abs() < 1e-10, "x{} = {}", i, x[i]);
        }
        assert_eq!(y.abs(), ans);
    }
}
