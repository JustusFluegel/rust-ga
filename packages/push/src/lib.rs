#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
// I'm not sure how I feel about this, but my
// naming has _frequently_ violated this rule,
// and I'm not 100% sure I'd want to avoid those
// violations.
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod genome;
pub mod instruction;
pub mod push_vm;

#[cfg(feature = "macros")]
pub use push_macros::*;

// This is to allow using the macros from push-macros inside this crate as
// push-macros relies on the crate `push` beeing present in the root module and
// this provides that (just an alias to self).
#[cfg(feature = "macros")]
extern crate self as push;

#[cfg(feature = "macros")]
pub use collectable;
