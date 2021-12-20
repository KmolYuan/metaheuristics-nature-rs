use crate::utility::prelude::*;

/// The methods of the meta-heuristic algorithms.
///
/// 1. Implement [`Setting`](crate::Setting) trait then indicate to a "method" type.
/// 1. Implement `Algorithm` trait on the "method" type.
///
/// Usually, the "method" type that implements this trait will not leak from the API.
/// All most common dataset is store in the [`Context`] type.
/// So the "method" type is used to store the additional data if any.
///
/// ```
/// use metaheuristics_nature::{utility::*, ObjFunc, Setting};
///
/// /// A setting with additional fields.
/// #[derive(Default)]
/// pub struct MySetting1 {
///     my_option: u32,
/// }
///
/// /// The implementation of the structure with fields.
/// impl Setting for MySetting1 {
///     type Algorithm = Method;
///
///     fn algorithm(self) -> Self::Algorithm {
///         Method
///     }
/// }
///
/// /// Tuple-like setting.
/// #[derive(Default)]
/// pub struct MySetting2;
///
/// /// The implementation of a tuple-like structure.
/// impl Setting for MySetting2 {
///     type Algorithm = Method;
///
///     fn algorithm(self) -> Self::Algorithm {
///         Method
///     }
/// }
///
/// pub struct Method;
///
/// impl<F: ObjFunc> Algorithm<F> for Method {
///     fn generation(&mut self, ctx: &mut Context<F>) {
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
