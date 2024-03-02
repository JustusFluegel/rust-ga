use std::marker::PhantomData;

use anyhow::Result;

use super::{composable::Wrappable, Composable, Operator};
use crate::{
    individual::{ec::EcIndividual, scorer::Scorer},
    population::Population,
};

pub struct GenomeScorer<GM, S, R> {
    genome_maker: GM,
    scorer: S,
    _result: PhantomData<R>,
}

impl<G, S, R> GenomeScorer<G, S, R> {
    pub const fn new(genome_maker: G, scorer: S) -> Self {
        Self {
            genome_maker,
            scorer,
            _result: PhantomData::<R>,
        }
    }
}

impl<G, S, R> Wrappable<G> for GenomeScorer<G, S, R> {
    type Context = S;

    fn construct(genome_maker: G, scorer: Self::Context) -> Self {
        Self::new(genome_maker, scorer)
    }
}

// scorer: &Genome -> TestResults<R>
impl<'pop, GM, S, R, P> Operator<&'pop P> for GenomeScorer<GM, S, R>
where
    P: Population,
    GM: Operator<&'pop P>,
    S: Scorer<GM::Output, R>,
{
    type Output = EcIndividual<GM::Output, R>;

    fn apply(&self, population: &'pop P, rng: &mut rand::rngs::ThreadRng) -> Result<Self::Output> {
        let genome = self.genome_maker.apply(population, rng)?;
        let score = self.scorer.score(&genome);
        // TODO: We probably don't want to bake in `EcIndividual` here, but instead
        //   have things be more general than that.
        Ok(EcIndividual::new(genome, score))
    }
}
impl<GM, S, R> Composable for GenomeScorer<GM, S, R> {}
