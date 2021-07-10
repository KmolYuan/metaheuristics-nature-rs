use crate::*;

/// A trait for fitting different callback functions.
///
/// + Empty callback `()`.
/// + None argument callback `Fn()`.
/// + One argument callback `Fn(&Report)`.
pub trait Callback<C> {
    #[must_use]
    fn call(&self, report: &Report) -> bool;
}

impl Callback<()> for () {
    #[inline(always)]
    fn call(&self, _: &Report) -> bool {
        false
    }
}

impl<T: Fn()> Callback<()> for T {
    #[inline(always)]
    fn call(&self, _: &Report) -> bool {
        self();
        false
    }
}

impl<T: Fn(&Report)> Callback<Report> for T {
    #[inline(always)]
    fn call(&self, report: &Report) -> bool {
        self(report);
        false
    }
}

impl<T: Fn() -> bool> Callback<bool> for T {
    #[inline(always)]
    fn call(&self, _: &Report) -> bool {
        self()
    }
}

impl<T: Fn(&Report) -> bool> Callback<(Report, bool)> for T {
    #[inline(always)]
    fn call(&self, report: &Report) -> bool {
        self(report)
    }
}
