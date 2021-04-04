mod rga;
mod de;
mod fa;
mod utility;

pub use crate::rga::{RGASetting, RGA};
pub use crate::de::{DESetting, DE, Strategy};
pub use crate::fa::{FASetting, FA};
pub use crate::utility::{ObjFunc, Algorithm, AlgorithmBase, Task, Report};

#[cfg(test)]
mod tests {
    use crate::ObjFunc;

    pub(crate) struct TestObj(u32, Vec<f64>, Vec<f64>);

    impl TestObj {
        pub(crate) fn new() -> Self {
            Self(0, vec![0., 0.], vec![50., 50.])
        }
    }

    impl ObjFunc for TestObj {
        type Result = f64;
        fn fitness(&self, _gen: u32, v: &Vec<f64>) -> f64 {
            v[0] * v[0] + 8. * v[1]
        }
        fn result(&self, v: &Vec<f64>) -> f64 { self.fitness(0, v) }
        fn ub(&self) -> &Vec<f64> { &self.2 }
        fn lb(&self) -> &Vec<f64> { &self.1 }
    }
}
