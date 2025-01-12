use std::error::Error;

use miette::Diagnostic;
use rand::{
    Rng,
    seq::{IndexedRandom, WeightError},
};

use super::{DynSelector, Selector, error::EmptyPopulation};
use crate::population::Population;

pub struct DynWeighted<P: Population> {
    selectors: Vec<(Box<dyn DynSelector<P> + Send + Sync>, usize)>,
}

impl<P: Population> std::fmt::Debug for DynWeighted<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynWeighted")
            .field("selectors", &self.selectors.len())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, thiserror::Error, Diagnostic)]
pub enum DynWeightedError {
    #[error(transparent)]
    #[diagnostic(transparent)]
    EmptyPopulation(#[from] EmptyPopulation),

    #[error(transparent)]
    #[diagnostic(help = "Ensure that the weights are all non-negative and add to more than zero")]
    ZeroWeightSum(#[from] WeightError),

    #[error(transparent)]
    Other(Box<dyn Error + Send + Sync>),
}

impl<P: Population> DynWeighted<P> {
    // Since we should never have an empty collection of weighted selectors,
    // the `new` implementation takes an initial selector so `selectors` is
    // guaranteed to never be empty.
    #[must_use]
    pub fn new<S>(selector: S, weight: usize) -> Self
    where
        S: Selector<P, Error: Error + Send + Sync + 'static> + Send + Sync + 'static,
    {
        Self {
            selectors: vec![(Box::new(selector), weight)],
        }
    }

    #[must_use]
    pub fn with_selector<S>(mut self, selector: S, weight: usize) -> Self
    where
        S: Selector<P, Error: Error + Send + Sync + 'static> + Send + Sync + 'static,
    {
        self.selectors.push((Box::new(selector), weight));
        self
    }
}

impl<P> Selector<P> for DynWeighted<P>
where
    P: Population,
{
    type Error = DynWeightedError;

    fn select<'pop, R: Rng + ?Sized>(
        &self,
        population: &'pop P,
        rng: &mut R,
    ) -> Result<&'pop P::Individual, Self::Error> {
        let (selector, _) = self.selectors.choose_weighted(rng, |(_, w)| *w)?;
        selector
            .select(population, rng)
            .map_err(DynWeightedError::Other)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use test_strategy::proptest;

    use super::DynWeighted;
    use crate::operator::selector::{Selector, best::Best, worst::Worst};

    #[proptest]
    fn best_or_worst(#[map(|v: [i32;10]| v.into())] pop: Vec<i32>) {
        let mut rng = rand::rng();
        // We'll make a selector that has a 50/50 chance of choosing the highest
        // or lowest value.
        let weighted = DynWeighted::new(Best, 1).with_selector(Worst, 1);
        let selection = weighted.select(&pop, &mut rng).unwrap();
        let extremes: [&i32; 2] = pop.iter().minmax().into_option().unwrap().into();
        assert!(extremes.contains(&selection));
    }
}
