//! Pre-implemented optimization methods.

#[cfg(any(feature = "std", feature = "libm"))]
pub use self::fa::Fa;
#[cfg(any(feature = "std", feature = "libm"))]
pub use self::rga::Rga;
#[cfg(any(feature = "std", feature = "libm"))]
pub use self::tlbo::Tlbo;
pub use self::{de::De, pso::Pso};

pub mod de;
#[cfg(any(feature = "std", feature = "libm"))]
#[cfg_attr(doc_cfg, doc(cfg(any(feature = "std", feature = "libm"))))]
pub mod fa;
pub mod pso;
#[cfg(any(feature = "std", feature = "libm"))]
#[cfg_attr(doc_cfg, doc(cfg(any(feature = "std", feature = "libm"))))]
pub mod rga;
#[cfg(any(feature = "std", feature = "libm"))]
#[cfg_attr(doc_cfg, doc(cfg(any(feature = "std", feature = "libm"))))]
pub mod tlbo;
