use crate::*;
use ndarray::AsArray;

const OFFSET: f64 = 7.;

struct TestObj([f64; 4], [f64; 4]);

impl Default for TestObj {
    fn default() -> Self {
        Self([0.; 4], [50.; 4])
    }
}

impl ObjFunc for TestObj {
    type Result = f64;

    fn fitness<'a, A>(&self, v: A, _: &Report) -> f64
    where
        A: AsArray<'a, f64>,
    {
        let v = v.into();
        OFFSET + v[0] * v[0] + 8. * v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    }

    fn result<'a, V>(&self, v: V) -> f64
    where
        V: AsArray<'a, f64>,
    {
        self.fitness(v, &Default::default())
    }

    fn ub(&self) -> &[f64] {
        &self.1
    }
    fn lb(&self) -> &[f64] {
        &self.0
    }
}

fn test<S: Setting>(obj: TestObj, setting: S) {
    let a = Solver::solve(obj, setting, |_| true);
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
    test(
        TestObj::default(),
        DESetting::default().task(Task::MinFit(OFFSET)),
    );
}

#[test]
fn pso() {
    test(
        TestObj::default(),
        PSOSetting::default().task(Task::MinFit(OFFSET)),
    );
}

#[test]
fn fa() {
    test(
        TestObj::default(),
        FASetting::default().task(Task::MinFit(OFFSET)),
    );
}

#[test]
fn rga() {
    test(
        TestObj::default(),
        RGASetting::default().task(Task::MinFit(OFFSET)),
    );
}

#[test]
fn tlbo() {
    test(
        TestObj::default(),
        TLBOSetting::default().task(Task::MinFit(OFFSET)),
    );
}
