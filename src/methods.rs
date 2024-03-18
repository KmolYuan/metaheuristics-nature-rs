//! Pre-implemented optimization methods.
//!
//! Each methods are also has some variants on implementation,
//! current methods are just designed for application.
pub use self::{
    de::{De, Strategy},
    fa::Fa,
    pso::Pso,
    rga::Rga,
    tlbo::Tlbo,
};

pub mod de;
pub mod fa;
pub mod pso;
pub mod rga;
pub mod tlbo;
