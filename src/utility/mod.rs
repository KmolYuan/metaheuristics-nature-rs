//! The utility API used to create a new algorithm.
//!
//! When building a new method, just import this module as prelude.
//!
//! ```
//! use metaheuristics_nature::{utility::*, *};
//! ```
//!
//! In other hand, if you went to fork the task manually by using parallel structure,
//! import [`thread_pool::ThreadPool`](crate::thread_pool::ThreadPool) is required.
pub use self::{
    context::Context,
    respond::Respond,
    setting::{BasicSetting, Setting},
};
pub use crate::random::*;
use crate::ObjFunc;
pub use ndarray::{s, Array1, Array2, AsArray};

mod context;
mod respond;
pub(crate) mod setting;

/// The methods of the meta-heuristic algorithms.
///
/// + First, use [`setting!`] macro to build a "setting" type.
/// + Second, implement [`Setting`] trait then indicate to a "method" type.
/// + Last, implement `Algorithm` trait on the "method" type.
///
/// Usually, the "method" type that implements this trait will not leak from the API.
/// All most common dataset is store in the [`Context`] type.
/// So the "method" type is used to store the additional data if any.
///
/// ```
/// use metaheuristics_nature::{setting, utility::*, ObjFunc};
///
/// /// A setting with additional fields.
/// #[derive(Default)]
/// pub struct MySetting1 {
///    base: BasicSetting,
///    my_option: u32,
/// }
///
/// /// The implementation of the structure with fields.
/// impl Setting for MySetting1 {
///     type Algorithm = Method;
///     fn base(&self) -> &BasicSetting {
///         &self.base
///     }
///     fn create(self) -> Self::Algorithm {
///         Method
///     }
/// }
///
/// /// Tuple-like setting.
/// #[derive(Default)]
/// pub struct MySetting2(BasicSetting);
///
/// /// The implementation of a tuple-like structure.
/// impl Setting for MySetting2 {
///     type Algorithm = Method;
///     fn base(&self) -> &BasicSetting {
///         &self.0
///     }
///     fn create(self) -> Self::Algorithm {
///         Method
///     }
/// }
///
/// pub struct Method;
///
/// impl Algorithm for Method {
///     fn generation<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
///         unimplemented!()
///     }
/// }
/// ```
///
/// Your algorithm will be implemented by the [`Solver`](crate::Solver) type automatically.
/// All you have to do is implement the "initialization" method and
/// "generation" method, which are corresponded to the [`Algorithm::init`] and
/// [`Algorithm::generation`] respectively.
pub trait Algorithm<F: ObjFunc> {
    /// Initialization implementation.
    ///
    /// The information of the [`Context`] can be obtained or modified at this phase preliminarily.
    ///
    /// The default behavior is do nothing.
    #[inline(always)]
    #[allow(unused_variables)]
    fn init(&mut self, ctx: &mut Context<F>) {}

    /// Processing implementation of each generation.
    fn generation(&mut self, ctx: &mut Context<F>);
}

/// Product two iterators together.
///
/// For example, `[a, b, c]` and `[1, 2, 3]` will become `[a1, a2, a3, b1, b2, b3, c1, c2, c3]`.
pub fn product<A, I1, I2>(iter1: I1, iter2: I2) -> impl Iterator<Item = (A, A)>
where
    A: Clone,
    I1: IntoIterator<Item = A>,
    I2: IntoIterator<Item = A> + Clone,
{
    iter1
        .into_iter()
        .flat_map(move |e: A| core::iter::repeat(e).zip(iter2.clone()))
}
