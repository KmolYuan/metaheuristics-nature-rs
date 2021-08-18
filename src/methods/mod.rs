//! Pre-implemented optimization methods.
pub use self::de::De;
pub use self::fa::Fa;
pub use self::pso::Pso;
pub use self::rga::Rga;
pub use self::tlbo::Tlbo;

pub mod de;
pub mod fa;
pub mod pso;
pub mod rga;
pub mod tlbo;
