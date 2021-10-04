//! Pre-implemented optimization methods.
#[cfg(all(feature = "std", feature = "libm"))]
compile_error!("Can not enable \"std\" and \"libm\" at the same time.");

pub use self::de::De;
#[cfg(any(feature = "std", feature = "libm"))]
pub use self::fa::Fa;
pub use self::pso::Pso;
#[cfg(any(feature = "std", feature = "libm"))]
pub use self::rga::Rga;
#[cfg(any(feature = "std", feature = "libm"))]
pub use self::tlbo::Tlbo;

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
