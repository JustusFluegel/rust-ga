#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]

pub mod child_maker;
pub mod generation;
pub mod generator;
pub mod genome;
pub mod individual;
pub mod operator;
pub mod population;
pub mod test_results;