use std::{
    error::Error,
    fmt::{Debug, Display},
};

use miette::{Diagnostic, LabeledSpan, Severity, SourceCode};

/// Error that occurs when trying to perform
/// [`UniformXo`](super::uniform_xo::UniformXo)
/// or [`TwoPointXo`](super::two_point_xo::TwoPointXo) on genomes of differing
/// lengths
#[derive(
    Debug, thiserror::Error, Diagnostic, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[error("Attempted to perform Crossover on genomes of different lengths {0} and {1}")]
#[diagnostic(help = "Ensure your genomes are of uniform length")]
pub struct DifferentGenomeLength(pub usize, pub usize);

/// Error that occurs when performing crossover using
/// [`UniformXo`](super::uniform_xo::UniformXo)
/// or [`TwoPointXo`](super::two_point_xo::TwoPointXo)
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CrossoverGeneError<E> {
    /// Attempted to crossover genomes with differing lengths
    DifferentGenomeLength(DifferentGenomeLength),
    /// Some other error specific to a crossover operation
    Crossover(E),
}

// We need to hand implement all these traits because `derive` for
// `thiserror::Error` and `miette::Diagnostic` don't
// handle generics well in this context. Hopefully that will be fixed in
// the future and we can simplify this considerably.

impl<E> Error for CrossoverGeneError<E>
where
    E: Error + 'static,
    Self: Debug + Display,
{
    fn source(&self) -> ::core::option::Option<&(dyn Error + 'static)> {
        match self {
            Self::DifferentGenomeLength(transparent) => Error::source(transparent),
            Self::Crossover(source) => Some(source),
        }
    }
}
impl<E> Display for CrossoverGeneError<E> {
    fn fmt(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        match self {
            Self::DifferentGenomeLength(g) => Display::fmt(&g, formatter),
            Self::Crossover(_) => formatter.write_str("Failed to crossover segment"),
        }
    }
}
impl<E> From<DifferentGenomeLength> for CrossoverGeneError<E> {
    fn from(source: DifferentGenomeLength) -> Self {
        Self::DifferentGenomeLength(source)
    }
}
impl<E> Diagnostic for CrossoverGeneError<E>
where
    E: Error + Diagnostic + 'static,
{
    fn code(&self) -> Option<Box<dyn Display + '_>> {
        match self {
            Self::DifferentGenomeLength(unnamed, ..) => unnamed.code(),
            Self::Crossover(unnamed, ..) => unnamed.code(),
        }
    }
    fn help(&self) -> Option<Box<dyn Display + '_>> {
        match self {
            Self::DifferentGenomeLength(unnamed, ..) => unnamed.help(),
            Self::Crossover(unnamed, ..) => unnamed.help(),
        }
    }
    fn severity(&self) -> Option<Severity> {
        match self {
            Self::DifferentGenomeLength(unnamed, ..) => unnamed.severity(),
            Self::Crossover(unnamed, ..) => unnamed.severity(),
        }
    }
    fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan> + '_>> {
        match self {
            Self::DifferentGenomeLength(unnamed, ..) => unnamed.labels(),
            Self::Crossover(unnamed, ..) => unnamed.labels(),
        }
    }
    fn source_code(&self) -> Option<&dyn SourceCode> {
        match self {
            Self::DifferentGenomeLength(unnamed, ..) => unnamed.source_code(),
            Self::Crossover(unnamed, ..) => unnamed.source_code(),
        }
    }
    fn related(&self) -> Option<Box<dyn Iterator<Item = &dyn Diagnostic> + '_>> {
        match self {
            Self::DifferentGenomeLength(unnamed, ..) => unnamed.related(),
            Self::Crossover(unnamed, ..) => unnamed.related(),
        }
    }
    fn url(&self) -> Option<Box<dyn Display + '_>> {
        match self {
            Self::DifferentGenomeLength(unnamed, ..) => unnamed.url(),
            Self::Crossover(unnamed, ..) => unnamed.url(),
        }
    }
    fn diagnostic_source(&self) -> Option<&dyn Diagnostic> {
        match self {
            Self::DifferentGenomeLength(unnamed, ..) => unnamed.diagnostic_source(),
            Self::Crossover(unnamed, ..) => unnamed.diagnostic_source(),
        }
    }
}
