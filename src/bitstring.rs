#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]

use std::borrow::Borrow;

use rand::seq::SliceRandom;
use rand::{rngs::ThreadRng, Rng};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::individual::Individual;
use crate::population::Population;

pub type Bitstring = Vec<bool>;

#[must_use]
pub fn count_ones(bits: &[bool]) -> i64 {
    bits.iter().filter(|&&bit| bit).count() as i64
}

fn all_same(bits: &[bool]) -> bool {
    bits.iter().all(|&bit| bit == bits[0])
}

#[must_use]
pub fn hiff(bits: &[bool]) -> i64 {
    if bits.len() < 2 {
        bits.len() as i64
    } else {
        let half_len = bits.len() / 2;
        let mut score = hiff(&bits[..half_len]) + hiff(&bits[half_len..]);
        if all_same(bits) {
            score += bits.len() as i64;
        }
        score
    }
}

pub fn make_random(len: usize, rng: &mut ThreadRng) -> Bitstring {
    (0..len).map(|_| rng.gen_bool(0.5)).collect()
}

/// # Panics
///
/// Will panic if the two parents don't have the same length.
pub fn uniform_xo(first_parent: &[bool], second_parent: &[bool], rng: &mut ThreadRng) -> Bitstring {
    // The two parents should have the same length.
    assert!(first_parent.len() == second_parent.len());
    let len = first_parent.len();
    (0..len).map(|i| 
        if rng.gen_bool(0.5) { 
            first_parent[i] 
        } else { 
            second_parent[i] 
        }).collect()
}

impl Individual<Bitstring> {
    pub fn new_bitstring<R>(bit_length: usize, compute_score: impl Fn(&R) -> i64 + Send + Sync, rng: &mut ThreadRng) -> Self
    where
        Bitstring: Borrow<R>,
        R: ?Sized
    {
        Self::new(
                |rng| make_random(bit_length, rng), 
                compute_score,
                rng)
    }
}

// TODO: I need to deal with the fact that this computes the score multiple times
// if I chain things like mutation and crossover. This is related to the need to
// parameterize the recombination operators, and I'll probably need to have some
// kind of vector of recombination operatorss that act on the Bitstrings, and then
// computes the score once at the end.
// 
// An alternative would be to use the Lazy eval tools and say that the score of
// an individual is computed lazily. That would mean that "intermediate" Individuals
// wouldn't have their score calculated since it's never used. That's a fairly
// heavy weight solution, though, so it would probably be nice to not go down
// that road if we don't have to.
//
// I also wonder if there are places where implementing the `From` trait would
// make sense. In principle we should be able to switch back and forth between
// `Bitstring` and `Individual` pretty freely, but I don't know if we can
// parameterize that with the score function.  
//
// This has hiff cooked in and needs to be parameterized on the score calculator.
impl Individual<Bitstring> {
    #[must_use]
    pub fn uniform_xo(&self, other_parent: &Self, rng: &mut ThreadRng) -> Self {
        let genome = uniform_xo(&self.genome, &other_parent.genome, rng);
        let score = hiff(&genome);
        Self { genome, score }
    }

    #[must_use]
    pub fn two_point_xo(&self, other_parent: &Self, compute_score: impl Fn(&[bool]) -> i64, rng: &mut ThreadRng) -> Self {
        let len = self.genome.len();
        let mut genome = self.genome.clone();
        let other_parent = other_parent.genome.clone();
        // let mut indices = (0..len).collect::<Vec<usize>>();
        // indices.shuffle(&mut rng);
        // let (first, second) = indices.split_at(rng.gen_range(0..len));
        let first = rng.gen_range(0..len);
        let second = rng.gen_range(first..len);
        if first < second {
            genome[first..second].clone_from_slice(&other_parent[first..second]);
        } else {
            genome[second..first].clone_from_slice(&other_parent[second..first]);
        }
        let score = compute_score(&genome);
        Self { genome, score }
    }

    #[must_use]
    pub fn mutate_one_over_length(&self, compute_score: impl Fn(&[bool]) -> i64, rng: &mut ThreadRng) -> Self {
        let length = self.genome.len() as f32;
        self.mutate_with_rate(1.0 / length, compute_score, rng)
    }

    #[must_use]
    pub fn mutate_with_rate(&self, mutation_rate: f32, compute_score: impl Fn(&[bool]) -> i64, rng: &mut ThreadRng) -> Self {
        let new_genome: Vec<bool> = self.genome.iter().map(|bit| {
            let r: f32 = rng.gen();
            if r < mutation_rate {
                !*bit
            } else {
                *bit
            }
        }).collect();
        let score = compute_score(&new_genome);
        Self { genome: new_genome, score }
    }
}

#[cfg(test)]
mod test {
    use std::iter::zip;

    use super::*;

    // This test is stochastic, so I'm going to ignore it most of the time.
    #[test]
    #[ignore]
    fn mutate_one_over_does_not_change_much() {
        let mut rng = rand::thread_rng();
        let num_bits = 100;
        let parent: Individual<Bitstring> = Individual::new_bitstring(num_bits, count_ones, &mut rng);
        let child = parent.mutate_one_over_length(count_ones, &mut rng);

        let num_differences = zip(parent.genome, child.genome).filter(|(p, c)| *p != *c).count();
        println!("Num differences = {num_differences}");
        assert!(0 < num_differences, "We're expecting at least one difference");
        assert!(num_differences < num_bits / 10, "We're not expecting lots of differences, and got {num_differences}.");
    }

    // This test is stochastic, so I'm going to ignore it most of the time.
    #[test]
    #[ignore]
    fn mutate_with_rate_does_not_change_much() {
        let mut rng = rand::thread_rng();
        let num_bits = 100;
        let parent: Individual<Bitstring> = Individual::new_bitstring(num_bits, count_ones, &mut rng);
        let child = parent.mutate_with_rate(0.05, count_ones, &mut rng);

        let num_differences = zip(parent.genome, child.genome).filter(|(p, c)| *p != *c).count();
        println!("Num differences = {num_differences}");
        assert!(0 < num_differences, "We're expecting at least one difference");
        assert!(num_differences < num_bits / 10, "We're not expecting lots of differences, and got {num_differences}.");
    }    
}

// Todo: Change this to use the new parameterized constructor.
impl Population<Bitstring> {
    pub fn new_bitstring_population<R>(
        pop_size: usize, 
        bit_length: usize, 
        compute_score: impl Fn(&R) -> i64 + Send + Sync) 
    -> Self
    where
        Bitstring: Borrow<R>,
        R: ?Sized
    {
        Self::new(
            pop_size,
            |rng| make_random(bit_length, rng),
            compute_score
        )
    }

    /// # Panics
    /// 
    /// Will panic if the population is empty.
    #[must_use]
    pub fn best_individual(&self) -> &Individual<Bitstring> {
        assert!(!self.individuals.is_empty());
        #[allow(clippy::unwrap_used)]
        self
            .individuals
            .iter()
            .max_by_key(
                |ind| ind.score
            )
            .unwrap()
    }

    /// # Panics
    /// 
    /// Panics if the population is empty.
    // This does uniform selection when it needs to take a parameterized selection function.
    // TODO: Do a performance test to see if there's a difference between this use of
    // `into_par_iter()` and `collect()` and the approach taken in `population.rs`.
    #[must_use]
    pub fn next_generation(&self, compute_score: impl Fn(&[bool]) -> i64 + Send + Sync) -> Self 
    {
        let previous_individuals = &self.individuals;
        assert!(!previous_individuals.is_empty());
        let pop_size = previous_individuals.len();
        let individuals = (0..pop_size)
            .into_par_iter()
            .map_init(
                rand::thread_rng,
                |rng, i| {
                    if i > 0 {
                        // These `unwraps()` are OK because we know the previous population isn't empty
                        // thanks to the assertion a few lines up.
                        #[allow(clippy::unwrap_used)]
                        let first_parent = previous_individuals.choose(rng).unwrap();
                        // let second_parent = previous_individuals.choose(rng).unwrap();
                        // let first_parent = self.best_individual();
                        let second_parent = self.best_individual();
                        // first_parent.uniform_xo(second_parent, rng).mutate(rng)
                        first_parent
                            .two_point_xo(second_parent, &compute_score, rng)
                            .mutate_one_over_length(&compute_score, rng)
                    } else {
                        self
                            .best_individual()
                            .clone()
                    }
                }
            ).collect();
        Self { individuals }
    }
}
