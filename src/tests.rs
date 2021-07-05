use crate::*;
use ndarray::{Array1, ArrayView1, AsArray};

struct TestObj(Array1<f64>, Array1<f64>);

impl TestObj {
    fn new() -> Self {
        Self(Array1::zeros(4), Array1::ones(4) * 50.)
    }
}

impl ObjFunc for TestObj {
    type Result = f64;
    fn fitness<'a, A>(&self, _gen: u32, v: A) -> f64
    where
        A: AsArray<'a, f64>,
    {
        let v = v.into();
        v[0] * v[0] + 8. * v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    }
    fn result<'a, A>(&self, v: A) -> f64
    where
        A: AsArray<'a, f64>,
    {
        self.fitness(0, v)
    }
    fn ub(&self) -> ArrayView1<f64> {
        self.1.view()
    }
    fn lb(&self) -> ArrayView1<f64> {
        self.0.view()
    }
}

fn test<S>(obj: TestObj, setting: S::Setting)
where
    S: Solver<TestObj>,
{
    let mut a = S::new(obj, setting);
    let ans = a.run(|| {});
    let (x, y) = a.result();
    let history = a.history();
    assert!(history.len() > 0, "{}", history.len());
    assert!(ans.abs() < 1e-20, "{}", ans);
    for i in 0..4 {
        assert!(x[i].abs() < 1e-10, "x{} = {}", i, x[i]);
    }
    assert_eq!(y.abs(), ans);
}

#[test]
fn de() {
    test::<DE<_>>(
        TestObj::new(),
        DESetting::default().task(Task::MinFit(1e-20)),
    );
}

#[test]
fn pso() {
    test::<PSO<_>>(
        TestObj::new(),
        PSOSetting::default().task(Task::MinFit(1e-20)),
    );
}

#[test]
fn fa() {
    test::<FA<_>>(
        TestObj::new(),
        FASetting::default().task(Task::MinFit(1e-20)),
    );
}

#[test]
fn rga() {
    test::<RGA<_>>(
        TestObj::new(),
        RGASetting::default().task(Task::MinFit(1e-20)),
    );
}

#[test]
fn tlbo() {
    test::<TLBO<_>>(
        TestObj::new(),
        TLBOSetting::default().task(Task::MinFit(1e-20)),
    );
}
