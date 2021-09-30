//! Pre-implemented optimization methods.
pub use self::de::De;
#[cfg(feature = "std")]
pub use self::fa::Fa;
pub use self::pso::Pso;
#[cfg(feature = "std")]
pub use self::rga::Rga;
#[cfg(feature = "std")]
pub use self::tlbo::Tlbo;

pub mod de;
#[cfg(feature = "std")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "std")))]
pub mod fa;
pub mod pso;
#[cfg(feature = "std")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "std")))]
pub mod rga;
#[cfg(feature = "std")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "std")))]
pub mod tlbo;
