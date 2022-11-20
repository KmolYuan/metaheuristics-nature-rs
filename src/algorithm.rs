use crate::utility::prelude::*;

/// The methods of the meta-heuristic algorithms.
///
/// 1. Implement [`Setting`] trait then indicate to a "method" type.
/// 1. Implement `Algorithm` trait on the "method" type.
///
/// Usually, the "method" type that implements this trait will not leak from the
/// API. All most common dataset is store in the [`Ctx`] type. So the "method"
/// type is used to store the additional data if any.
///
/// ```
/// use metaheuristics_nature::utility::prelude::*;
///
/// /// A setting with fields.
/// #[derive(Default)]
/// pub struct MySetting1 {
///     my_option: u32,
/// }
///
/// /// The implementation of the structure with fields.
/// impl Setting for MySetting1 {
///     type Algorithm<F: ObjFunc> = Method;
///
///     fn algorithm<F: ObjFunc>(self) -> Self::Algorithm<F> {
///         Method /* inherit setting */
///     }
/// }
///
/// /// No setting.
/// #[derive(Default)]
/// pub struct MySetting2;
///
/// /// The implementation of a tuple-like structure.
/// impl Setting for MySetting2 {
///     type Algorithm<F: ObjFunc> = Method;
///
///     fn algorithm<F: ObjFunc>(self) -> Self::Algorithm<F> {
///         Method
///     }
/// }
///
/// /// The type implements our algorithm.
/// pub struct Method;
///
/// impl<F: ObjFunc> Algorithm<F> for Method {
///     fn generation(&mut self, ctx: &mut Ctx<F>, rng: &Rng) {
///         /* implement the method */
///     }
/// }
/// ```
///
/// The complete algorithm will be implemented by the [`Solver`](crate::Solver)
/// type automatically. All you have to do is implement the "initialization"
/// method and "generation" method, which are corresponded to the
/// [`Algorithm::init()`] and [`Algorithm::generation()`] respectively.
pub trait Algorithm<F: ObjFunc> {
    /// Initialization implementation.
    ///
    /// The information of the [`Ctx`] can be obtained or modified at this phase
    /// preliminarily.
    ///
    /// The default behavior is do nothing.
    #[inline]
    #[allow(unused_variables)]
    fn init(&mut self, ctx: &mut Ctx<F>, rng: &Rng) {}

    /// Processing implementation of each generation.
    fn generation(&mut self, ctx: &mut Ctx<F>, rng: &Rng);
}
