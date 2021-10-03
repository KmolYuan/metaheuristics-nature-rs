use crate::{utility::Setting, *};

const OFFSET: f64 = 7.;

struct TestObj([f64; 4], [f64; 4]);

impl Default for TestObj {
    fn default() -> Self {
        Self([0.; 4], [50.; 4])
    }
}

impl ObjFunc for TestObj {
    type Result = f64;
    type Respond = f64;

    fn fitness(&self, v: &[f64], _: &Report) -> f64 {
        OFFSET + v[0] * v[0] + 8. * v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    }

    fn result(&self, v: &[f64]) -> f64 {
        self.fitness(v, &Default::default())
    }

    fn ub(&self) -> &[f64] {
        &self.1
    }
    fn lb(&self) -> &[f64] {
        &self.0
    }
}

fn test(setting: impl Setting) {
    let a = Solver::solve(TestObj::default(), setting, |_| true);
    let ans = a.result();
    let x = a.best_parameters();
    let y = a.best_fitness();
    let reports = a.reports();
    assert!(reports.len() > 0, "{}", reports.len());
    assert!((ans - OFFSET).abs() < 1e-20, "{}", ans);
    for i in 0..4 {
        assert!(x[i].abs() < 1e-6, "x{} = {}", i, x[i]);
    }
    assert_eq!(y.abs(), ans);
}

#[test]
fn de() {
    test(setting!(De {
        +base: { task: Task::MinFit(OFFSET) }
    }));
}

#[test]
fn pso() {
    test(setting!(Pso {
        +base: { task: Task::MinFit(OFFSET) }
    }));
}

#[test]
fn fa() {
    test(setting!(Fa {
        +base: { task: Task::MinFit(OFFSET) }
    }));
}

#[test]
fn rga() {
    test(setting!(Rga {
        +base: { task: Task::MinFit(OFFSET) }
    }));
}

#[test]
fn tlbo() {
    test(setting!(Tlbo(task: Task::MinFit(OFFSET))));
}
