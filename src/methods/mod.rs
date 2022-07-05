//! Pre-implemented optimization methods.
//!
//! Each methods are also has some variants on implementation,
//! current methods are just designed for application.

#[cfg(any(feature = "std", feature = "libm"))]
pub use self::fa::Fa;
#[cfg(any(feature = "std", feature = "libm"))]
pub use self::rga::Rga;
#[cfg(any(feature = "std", feature = "libm"))]
pub use self::tlbo::Tlbo;
pub use self::{de::De, pso::Pso};

pub mod de;
#[cfg(any(feature = "std", feature = "libm"))]
pub mod fa;
pub mod pso;
#[cfg(any(feature = "std", feature = "libm"))]
pub mod rga;
#[cfg(any(feature = "std", feature = "libm"))]
pub mod tlbo;
