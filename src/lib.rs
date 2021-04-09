mod utility;
mod rga;
mod de;
mod pso;
mod fa;
mod tlbo;

pub use crate::{
    rga::{RGASetting, RGA},
    de::{DESetting, DE, Strategy},
    pso::{PSOSetting, PSO},
    fa::{FASetting, FA},
    tlbo::{TLBOSetting, TLBO},
    utility::{ObjFunc, Setting, Algorithm, AlgorithmBase, Task, Report},
};

#[cfg(test)]
mod tests {
    use crate::{ObjFunc, Algorithm};

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
        fn result(&self, v: &Vec<f64>) -> f64 { self.fitness(0, v) }
        fn ub(&self) -> &Vec<f64> { &self.1 }
        fn lb(&self) -> &Vec<f64> { &self.0 }
    }

    pub(crate) fn test<F, A>(mut a: A)
        where F: ObjFunc<Result=f64>,
              A: Algorithm<F> {
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
