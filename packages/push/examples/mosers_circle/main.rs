// TODO: since inner attributes are unstable, we can't use rustversion here.
// Once we revert this commit, this is proper again.
#![allow(clippy::allow_attributes_without_reason)]
#![allow(
    clippy::arithmetic_side_effects,
    // reason = "The tradeoff safety <> ease of writing arguably lies on the ease of writing side \
    //           for example code."
)]

pub mod args;

use std::{convert::Infallible, ops::Not};

use anyhow::{ensure, Result};
use clap::Parser;
use ec_core::{
    distributions::collection::ConvertToCollectionGenerator,
    generation::Generation,
    individual::{ec::WithScorer, scorer::FnScorer},
    operator::{
        genome_extractor::GenomeExtractor,
        genome_scorer::GenomeScorer,
        mutator::Mutate,
        selector::{
            best::Best, lexicase::Lexicase, tournament::Tournament, weighted::Weighted, Select,
            Selector,
        },
        Composable,
    },
    test_results::{self, TestResults},
    uniform_distribution_of,
};
use ec_linear::mutator::umad::Umad;
use num_traits::CheckedSub;
use push::{
    evaluation::{Case, Cases, WithTargetFn},
    genome::plushy::{ConvertToGeneGenerator, Plushy},
    instruction::{
        instruction_error::PushInstructionError, variable_name::VariableName, Instruction,
        IntInstruction, IntInstructionError, NumOpens, PushInstruction,
    },
    push_vm::{program::PushProgram, push_state::PushState, stack::PushOnto, HasStack, State},
};
use rand::{distr::Distribution, thread_rng};

use crate::args::{Args, RunModel};

/*
 * This is an implementation of the "moser's circle" problem,
 * which is a common tale on how not to trust initial patterns in a series.
 *
 * For more information on the problem see the wikipedia article
 * https://en.wikipedia.org/wiki/Dividing_a_circle_into_areas
 * or this excellent video by 3blue1brown:
 * https://www.youtube.com/watch?v=YtkIWDE36qU
 */

const PENALTY_VALUE: i64 = 1_000_000_000;

const fn gcd(mut m: u128, mut n: u128) -> u128 {
    if m == 0 || n == 0 {
        return m | n;
    }

    let shift = (m | n).trailing_zeros();

    m >>= m.trailing_zeros();
    n >>= n.trailing_zeros();

    while m != n {
        if m > n {
            m -= n;
            m >>= m.trailing_zeros();
        } else {
            n -= m;
            n >>= n.trailing_zeros();
        }
    }

    m << shift
}

// See http://blog.plover.com/math/choose.html for the idea.
fn binom_coeff(mut n: u64, mut k: u64) -> u128 {
    if k > n {
        return 0;
    }

    if let Some(diff) = n.checked_sub(k) {
        if diff < k {
            k = diff;
        }
    }

    let mut result: u128 = 1;

    for divisor in 1..=u128::from(k) {
        // avoid overflows
        let gcd = gcd(result, divisor);
        result = result / gcd * (u128::from(n) / (divisor / gcd));
        n -= 1;
    }

    result
}

#[derive(Clone, Eq, PartialEq, Debug)]
struct BinomInstr;

impl<S> Instruction<S> for BinomInstr
where
    S: HasStack<i64>,
{
    type Error = PushInstructionError;

    fn perform(&self, state: S) -> push::error::InstructionResult<S, Self::Error> {
        state
            .stack::<i64>()
            .top2()
            .map_err(PushInstructionError::from)
            .and_then(|(n, k)| {
                i64::try_from(binom_coeff(
                    u64::try_from(*n).map_err(|_| IntInstructionError::Overflow {
                        op: IntInstruction::Add,
                    })?,
                    u64::try_from(*k).map_err(|_| IntInstructionError::Overflow {
                        op: IntInstruction::Add,
                    })?,
                ))
                .map_err(|_| IntInstructionError::Overflow {
                    op: IntInstruction::Add,
                })
                .map_err(Into::into)
            })
            .replace_on(2, state)
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
enum Instructions {
    InputVar(VariableName),
    Int(IntInstruction),
    Binom(BinomInstr),
}

impl Instruction<PushState> for Instructions {
    type Error = PushInstructionError;

    fn perform(&self, state: PushState) -> push::error::InstructionResult<PushState, Self::Error> {
        match self {
            Self::Int(p) => p.perform(state),
            Self::Binom(b) => b.perform(state),
            Self::InputVar(v) => state.with_input(v),
        }
    }
}

impl From<IntInstruction> for Instructions {
    fn from(value: IntInstruction) -> Self {
        Self::Int(value)
    }
}

impl From<BinomInstr> for Instructions {
    fn from(value: BinomInstr) -> Self {
        Self::Binom(value)
    }
}

impl From<VariableName> for Instructions {
    fn from(value: VariableName) -> Self {
        Self::InputVar(value)
    }
}

impl NumOpens for Instructions {
    fn num_opens(&self) -> usize {
        match self {
            _ => 0,
        }
    }
}

#[rustversion::attr(
    before(1.81),
    allow(clippy::as_conversions, clippy::cast_possible_truncation)
)]
#[rustversion::attr(
    since(1.81),
    expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "We currently don't have a u64 stack so we use the i64 one - since we know the \
                  inputs and the function this cast will always be fine"
    )
)]
fn target_fn(input: u64) -> u64 {
    1 + binom_coeff(input, 2) as u64 + binom_coeff(input, 4) as u64
}

#[rustversion::attr(before(1.81), allow(clippy::unwrap_used))]
#[rustversion::attr(
    since(1.81),
    expect(
        clippy::unwrap_used,
        reason = "This will panic if the program is longer than the allowed max stack size. We \
                  arguably should check that and return an error here."
    )
)]
fn build_push_state(
    program: impl DoubleEndedIterator<Item = PushProgram> + ExactSizeIterator,
    input: i64,
) -> PushState {
    PushState::builder()
        .with_max_stack_size(1000)
        .with_program(program)
        // This will return an error if the program is longer than the allowed
        // max stack size.
        // We arguably should check that and return an error here.
        .unwrap()
        .with_int_input("x", input)
        .build()
}

fn score_program(
    program: impl DoubleEndedIterator<Item = PushProgram> + ExactSizeIterator,
    Case { input, output }: Case<i64>,
) -> i64 {
    let state = build_push_state(program, input);

    let Ok(state) = state.run_to_completion() else {
        // Do some logging, perhaps?
        return PENALTY_VALUE;
    };

    let Ok(&answer) = state.stack::<i64>().top() else {
        // Do some logging, perhaps?
        return PENALTY_VALUE;
    };

    (answer - output).abs()
}

fn score_genome(
    genome: &Plushy,
    training_cases: &Cases<i64>,
) -> TestResults<test_results::Error<i128>> {
    let program: Vec<PushProgram> = genome.clone().into();

    training_cases
        .iter()
        .map(|&case| i128::from(score_program(program.iter().cloned(), case)))
        .collect()
}

#[rustversion::attr(
    before(1.81),
    allow(
        clippy::as_conversions,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )
)]
#[rustversion::attr(
    since(1.81),
    expect(
        clippy::as_conversions,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "We currently don't have a u64 stack so we use the i64 one - since we know the \
                  inputs and the function this cast will always be fine"
    )
)]
fn main() -> Result<()> {
    // FIXME: Respect the max_genome_length input
    let Args {
        run_model,
        population_size,
        max_initial_instructions,
        num_generations,
        ..
    } = Args::parse();

    let mut rng = thread_rng();

    // Inputs from 1 to 10 (inclusive).
    // TODO: add a u64 stack type, don't cast to i64
    let training_cases = (1..=100).with_target_fn(|i| target_fn(*i as u64) as i64);

    // dbg!(training_cases);
    // return Ok(());
    /*
     * The `scorer` will need to take an evolved program (sequence of
     * instructions) and run it on all the inputs, collecting together the
     * errors, i.e., the absolute difference between the returned value and
     * the expected value.
     *
     * The target polynomial is 1 + (n choose 2) + (n choose 4)
     */
    let scorer = FnScorer(|genome: &Plushy| score_genome(genome, &training_cases));

    let num_test_cases = 10;

    let selector = Weighted::new(Best, 1)
        .with_selector(Lexicase::new(num_test_cases), 5)
        .with_selector(Tournament::binary(), population_size - 1);

    let gene_generator = uniform_distribution_of![<Instructions>
        IntInstruction::Push(1),
        IntInstruction::Push(24),
        IntInstruction::Push(6),
        IntInstruction::Push(23),
        IntInstruction::Push(18),
        IntInstruction::Add,
        IntInstruction::Subtract,
        IntInstruction::Power,
        IntInstruction::Multiply,
        IntInstruction::ProtectedDivide,
        VariableName::from("x")
    ]
    .into_gene_generator();

    let population = gene_generator
        .to_collection_generator(max_initial_instructions)
        .with_scorer(scorer)
        .into_collection_generator(population_size)
        .sample(&mut rng);

    ensure!(population.is_empty().not());

    let best = Best.select(&population, &mut rng)?;
    println!("Best initial individual is {best}");

    let umad = Umad::new(0.1, 0.1, &gene_generator);

    let make_new_individual = Select::new(selector)
        .then(GenomeExtractor)
        .then(Mutate::new(umad))
        .wrap::<GenomeScorer<_, _>>(scorer);

    let mut generation = Generation::new(make_new_individual, population);

    // TODO: It might be useful to insert some kind of logging system so we can
    // make this less imperative in nature.

    for generation_number in 0..num_generations {
        match run_model {
            RunModel::Serial => generation.serial_next()?,
            RunModel::Parallel => generation.par_next()?,
        }

        let best = Best.select(generation.population(), &mut rng)?;
        // TODO: Change 2 to be the smallest number of digits needed for
        // num_generations-1.
        println!("Generation {generation_number:2} best is {best}\n");

        if best.test_results.total_result.error == 0 {
            println!("SUCCESS");
            break;
        }
    }

    Ok(())
}
