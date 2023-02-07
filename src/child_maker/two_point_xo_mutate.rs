use super::ChildMaker;
use crate::{
    bitstring::Bitstring,
    individual::{ec::EcIndividual, Individual},
    operator::{Composable, Operator},
    recombinator::{
        mutate_with_one_over_length::MutateWithOneOverLength, two_point_xo::TwoPointXo,
    },
    selector::Selector,
    test_results::TestResults,
};
use rand::rngs::ThreadRng;
use std::iter::Sum;

#[derive(Clone)]
pub struct TwoPointXoMutate<'a> {
    pub scorer: &'a (dyn Fn(&[bool]) -> Vec<i64> + Sync),
}

impl<'a> TwoPointXoMutate<'a> {
    pub fn new(scorer: &'a (dyn Fn(&[bool]) -> Vec<i64> + Sync)) -> Self {
        Self { scorer }
    }
}

// TODO: Try this as a closure and see if we still get lifetime
//   capture problems.
fn make_child_genome(parent_genomes: [Bitstring; 2], rng: &mut ThreadRng) -> Bitstring {
    TwoPointXo
        .then(MutateWithOneOverLength)
        .apply(parent_genomes, rng)
}

impl<'a, S, R> ChildMaker<Vec<EcIndividual<Bitstring, TestResults<R>>>, S> for TwoPointXoMutate<'a>
where
    S: Selector<Vec<EcIndividual<Bitstring, TestResults<R>>>>,
    R: Sum + Copy + From<i64>,
{
    fn make_child(
        &self,
        rng: &mut ThreadRng,
        population: &Vec<EcIndividual<Bitstring, TestResults<R>>>,
        selector: &S,
    ) -> EcIndividual<Bitstring, TestResults<R>> {
        let first_parent = selector.select(rng, population);
        let second_parent = selector.select(rng, population);

        let parent_genomes = [
            first_parent.genome().clone(),
            second_parent.genome().clone(),
        ];

        let mutated_genome = make_child_genome(parent_genomes, rng);

        let test_results = (self.scorer)(&mutated_genome)
            .into_iter()
            .map(From::from)
            .sum();
        EcIndividual::new(mutated_genome, test_results)
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::bitstring::count_ones;

    use super::*;

    #[test]
    fn smoke_test() {
        let mut rng = thread_rng();

        let first_parent = EcIndividual::new_bitstring(100, count_ones, &mut rng);
        let second_parent = EcIndividual::new_bitstring(100, count_ones, &mut rng);

        let first_genome = first_parent.genome().clone();
        let second_genome = second_parent.genome().clone();

        let child_genome = make_child_genome([first_genome, second_genome], &mut rng);

        let first_genome = first_parent.genome();
        let second_genome = second_parent.genome();

        let num_in_either_parent = child_genome
            .into_iter()
            .enumerate()
            .filter(|(pos, val)| *val == first_genome[*pos] || *val == second_genome[*pos])
            .count();
        assert!(
            num_in_either_parent > 90 && num_in_either_parent < 100,
            "{num_in_either_parent} wasn't in the expected range"
        );
    }
}
