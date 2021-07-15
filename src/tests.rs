use crate::*;
use ndarray::{Array1, ArrayView1, AsArray};

const OFFSET: f64 = 7.;

struct TestObj(Array1<f64>, Array1<f64>);

impl Default for TestObj {
    fn default() -> Self {
        // Self(Array1::ones(4) * -50., Array1::ones(4) * 50.)
        Self(Array1::zeros(4), Array1::ones(4) * 50.)
    }
}

impl ObjFunc for TestObj {
    type Result = f64;

    fn fitness<'a, A>(&self, v: A, _: &Report) -> f64
    where
        A: AsArray<'a, f64>,
    {
        let v = v.into();
        // std::thread::sleep(std::time::Duration::from_millis(10));
        OFFSET + v[0] * v[0] + 8. * v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    }

    fn result<'a, V>(&self, v: V) -> f64
    where
        V: AsArray<'a, f64>,
    {
        self.fitness(v, &Default::default())
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
    let a = S::solve(obj, setting, ());
    let ans = a.result();
    let (x, y) = a.parameters();
    let history = a.history();
    assert!(history.len() > 0, "{}", history.len());
    assert!((ans - OFFSET).abs() < 1e-20, "{}", ans);
    for i in 0..4 {
        assert!(x[i].abs() < 1e-6, "x{} = {}", i, x[i]);
    }
    assert_eq!(y.abs(), ans);
}

#[test]
fn de() {
    test::<DE<_>>(
        TestObj::default(),
        DESetting::default().task(Task::MinFit(OFFSET)),
    );
}

#[test]
fn pso() {
    test::<PSO<_>>(
        TestObj::default(),
        PSOSetting::default().task(Task::MinFit(OFFSET)),
    );
}

#[test]
fn fa() {
    test::<FA<_>>(
        TestObj::default(),
        FASetting::default().task(Task::MinFit(OFFSET)),
    );
}

#[test]
fn rga() {
    test::<RGA<_>>(
        TestObj::default(),
        RGASetting::default().task(Task::MinFit(OFFSET)),
    );
}

#[test]
fn tlbo() {
    test::<TLBO<_>>(
        TestObj::default(),
        TLBOSetting::default().task(Task::MinFit(OFFSET)),
    );
}
