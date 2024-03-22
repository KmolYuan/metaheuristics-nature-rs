use crate::prelude::*;

/// Algorithm configurations. A trait for preparing the algorithm.
///
/// The setting type is actually a builder of the [`AlgCfg::Algorithm`] type.
///
/// Please note that the setting should not overlap with the [`SolverBuilder`].
pub trait AlgCfg {
    /// Associated algorithm.
    type Algorithm<F: ObjFunc>: Algorithm<F> + 'static;
    /// Create the algorithm.
    fn algorithm<F: ObjFunc>(self) -> Self::Algorithm<F>;
    /// Default population number.
    fn pop_num() -> usize {
        200
    }
}

/// The methods of the metaheuristic algorithms.
///
/// 1. Implement [`AlgCfg`] trait then indicate to a "method" type.
/// 1. Implement `Algorithm` trait on the "method" type.
///
/// Usually, the "method" type that implements this trait will not leak from the
/// API. All most common dataset is store in the [`Ctx`] type. So the "method"
/// type is used to store the additional data if any.
///
/// ```
/// use metaheuristics_nature::prelude::*;
///
/// /// A setting with fields.
/// #[derive(Default)]
/// pub struct MySetting1 {
///     my_option: u32,
/// }
///
/// /// The implementation of the structure with fields.
/// impl AlgCfg for MySetting1 {
///     type Algorithm<F: ObjFunc> = Method;
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
/// impl AlgCfg for MySetting2 {
///     type Algorithm<F: ObjFunc> = Method;
///     fn algorithm<F: ObjFunc>(self) -> Self::Algorithm<F> {
///         Method
///     }
/// }
///
/// /// The type implements our algorithm.
/// pub struct Method;
///
/// impl<F: ObjFunc> Algorithm<F> for Method {
///     fn generation(&mut self, ctx: &mut Ctx<F>, rng: &mut Rng) {
///         /* implement the method */
///     }
/// }
/// ```
///
/// The complete algorithm will be implemented by the [`Solver`](crate::Solver)
/// type automatically. All you have to do is implement the "initialization"
/// method and "generation" method, which are corresponded to the
/// [`Algorithm::init()`] and [`Algorithm::generation()`] respectively.
///
/// The generic type `F: ObjFunc` is the objective function marker, which is
/// used to allow storing the types that are related to the objective function
/// for the implementor `Self`. An actual example is
/// [`crate::methods::pso::Method`].
pub trait Algorithm<F: ObjFunc>: MaybeParallel {
    /// Initialization implementation.
    ///
    /// The information of the [`Ctx`] can be obtained or modified at this phase
    /// preliminarily.
    ///
    /// The default behavior is do nothing.
    #[inline]
    #[allow(unused_variables)]
    fn init(&mut self, ctx: &mut Ctx<F>, rng: &mut Rng) {}

    /// Processing implementation of each generation.
    fn generation(&mut self, ctx: &mut Ctx<F>, rng: &mut Rng);
}

/// Implement for `Box<dyn Algorithm<F>>`.
///
/// See also [`SolverBox`].
impl<F: ObjFunc, T: Algorithm<F> + ?Sized> Algorithm<F> for Box<T> {
    #[inline]
    fn init(&mut self, ctx: &mut Ctx<F>, rng: &mut Rng) {
        self.as_mut().init(ctx, rng);
    }

    #[inline]
    fn generation(&mut self, ctx: &mut Ctx<F>, rng: &mut Rng) {
        self.as_mut().generation(ctx, rng);
    }
}
