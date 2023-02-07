use std::ops::Not;

use num_traits::ToPrimitive;
use rand::rngs::ThreadRng;

use crate::operator::{Composable, Operator};

use super::{mutate_with_rate::MutateWithRate, Recombinator};

pub struct MutateWithOneOverLength;

impl<T> Recombinator<Vec<T>> for MutateWithOneOverLength
where
    T: Clone + Not<Output = T>,
{
    fn recombine(&self, genome: &[&Vec<T>], rng: &mut ThreadRng) -> Vec<T> {
        self.apply(genome[0].clone(), rng)
    }
}

impl<T> Operator<Vec<T>> for MutateWithOneOverLength
where
    T: Not<Output = T>,
{
    type Output = Vec<T>;

    fn apply(&self, genome: Vec<T>, rng: &mut ThreadRng) -> Self::Output {
        let mutation_rate = genome.len().to_f32().map_or(f32::MIN_POSITIVE, |l| 1.0 / l);
        let mutator = MutateWithRate::new(mutation_rate);
        mutator.apply(genome, rng)
    }
}
impl Composable for MutateWithOneOverLength {}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use crate::{
        bitstring::make_random,
        recombinator::{mutate_with_one_over_length::MutateWithOneOverLength, Recombinator},
    };

    // This test is stochastic, so I'm going to ignore it most of the time.
    #[test]
    #[ignore]
    fn mutate_one_over_does_not_change_much() {
        let mut rng = rand::thread_rng();
        let num_bits = 100;
        let parent_bits = make_random(num_bits, &mut rng);

        let child_bits = MutateWithOneOverLength.recombine(&[&parent_bits], &mut rng);

        let num_differences = zip(parent_bits, child_bits)
            .filter(|(p, c)| *p != *c)
            .count();
        println!("Num differences = {num_differences}");
        assert!(
            0 < num_differences,
            "We're expecting at least one difference"
        );
        assert!(
            num_differences < num_bits / 10,
            "We're not expecting lots of differences, and got {num_differences}."
        );
    }
}
