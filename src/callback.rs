use crate::Report;

/// A trait for fitting different callback functions.
///
/// The function that returns boolean "true" can interrupt the algorithm manually.
///
/// + Empty callback `()`.
/// + None argument callback `FnMut()`.
/// + One argument callback `FnMut(&Report)`.
/// + None argument callback `FnMut() -> bool`.
/// + One argument callback `FnMut(&Report) -> bool`.
///
/// When using this trait, please use a generic parameter to keep the variability of
/// callback signature. For example:
///
/// ```
/// use metaheuristics_nature::Callback;
/// fn test<C>(_callback: impl Callback<C>) {}
/// ```
pub trait Callback<C> {
    #[must_use]
    fn call(&mut self, report: Report) -> bool;
}

impl Callback<()> for () {
    #[inline(always)]
    fn call(&mut self, _: Report) -> bool {
        false
    }
}

impl<T: FnMut()> Callback<()> for T {
    #[inline(always)]
    fn call(&mut self, _: Report) -> bool {
        self();
        false
    }
}

impl<T: FnMut(Report)> Callback<Report> for T {
    #[inline(always)]
    fn call(&mut self, report: Report) -> bool {
        self(report);
        false
    }
}

impl<T: FnMut() -> bool> Callback<bool> for T {
    #[inline(always)]
    fn call(&mut self, _: Report) -> bool {
        self()
    }
}

impl<T: FnMut(Report) -> bool> Callback<(Report, bool)> for T {
    #[inline(always)]
    fn call(&mut self, report: Report) -> bool {
        self(report)
    }
}
