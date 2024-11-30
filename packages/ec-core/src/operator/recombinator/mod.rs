use rand::rngs::ThreadRng;

use super::{Composable, Operator};

/// Recombine (usually two or more) genomes into a new
/// genome of the same type,
///
/// - `GS` represents the type for the _group_ of input genomes. This is
///   typically a tuple, an array, or some other collection of genomes that will
///   be recombined.
/// - `Output`: An associated type indicating what the result type of the
///   recombination action will be.
///
/// Typically `Output` will be the same as the type of the
/// individual genomes contained in `GS`, but that isn't captured
/// (or required) here.
///
/// Implementations of this trait are typically representation dependent,
/// so see crates like [`ec_linear`](../../../ec_linear/recombinator/index.html)
/// for examples of recombinators on linear genomes.
///
/// # See also
///
/// See [`Recombine`] for a wrapper that converts a `Recombinator` into an
/// [`Operator`], allowing recombinators to be used in chains of operators.
///
/// # Examples
///
/// In this example, we define a `Recombinator` that swaps one randomly
/// chosen element from the first parent with the corresponding element
/// from the second parent. We then use this recombinator to create a
/// new genome from two parents, confirming that the child genome has
/// all but one element from the first parent and one element from the
/// second parent.
///
/// ```
/// # use rand::thread_rng;
/// # use rand::rngs::ThreadRng;
/// # use rand::Rng;
/// # use ec_core::operator::recombinator::Recombinator;
/// #
/// struct SwapOne;
/// type Parents = (Vec<u8>, Vec<u8>);
///
/// impl Recombinator<Parents> for SwapOne {
///     type Output = Vec<u8>;
///
///     fn recombine(
///         &self,
///         (mut first_parent, second_parent): Parents,
///         rng: &mut ThreadRng,
///     ) -> anyhow::Result<Vec<u8>> {
///         let index = rng.gen_range(0..first_parent.len());
///         first_parent[index] = second_parent[index];
///         Ok(first_parent)
///     }
/// }
///
/// // Swpapping one element from the first parent with the second parent
/// // should result in a child with three zeros and a single one.
/// let first_parent = vec![0, 0, 0, 0];
/// let second_parent = vec![1, 1, 1, 1];
/// let child = SwapOne
///     .recombine((first_parent, second_parent), &mut thread_rng())
///     .unwrap();
/// let num_zeros = child.iter().filter(|&&x| x == 0).count();
/// let num_ones = child.iter().filter(|&&x| x == 1).count();
/// assert_eq!(num_zeros, 3);
/// assert_eq!(num_ones, 1);
/// ```
pub trait Recombinator<GS> {
    /// The type of the output genome after recombination. This is typically
    /// the same as the type of genomes in the input group of type `GS`, but
    /// that isn't strictly required here.
    type Output;

    /// Recombine the given `genomes` returning a new genome of type `Output`.
    ///
    /// # Errors
    ///
    /// This will return an error if there is an error recombining the given
    /// parent genomes. This will usually be because the given `genomes` are
    /// invalid in some way, thus making recombination impossible.
    fn recombine(&self, genomes: GS, rng: &mut ThreadRng) -> anyhow::Result<Self::Output>;
}

// TODO: Why does `rustfmt` mangle the `CountTree::apply()` method?

/// A wrapper that converts a `Recombinator` into an `Operator`,
///
/// This allows the inclusion of `Recombinator`s in chains of operators.
///
/// # See also
///
/// See [`Recombinator`] for the details of that trait.
///
/// See [the `operator` module docs](crate::operator#wrappers) for more on
/// the design decisions behind using wrappers to convert things like a
/// [`Recombinator`] into an [`Operator`].
///
/// # Examples
///
/// Here we illustrate the implementation of `SwapFirst`, a simple
/// [`Recombinator`] that acts on a pair of vectors, replacing the first element
/// from the first vector with the first element from the the second vector. We
/// then wrap that in a [`Recombine`] to create an [`Operator`]. Calling
/// [`Operator::apply`] on the resulting operator should then be the same as
/// calling [`Recombinator::recombine`] directly on the recombinator.
///
/// ```
/// # use rand::{rngs::ThreadRng, thread_rng};
/// #
/// # use ec_core::operator::{Operator, recombinator::{Recombinator, Recombine}};
/// // A simple `Recombinator` that swaps one element from the second parent
/// // into the corresponding position in the first parent.
/// struct SwapFirst;
///
/// impl<T: Copy> Recombinator<(Vec<T>, Vec<T>)> for SwapFirst {
///     type Output = Vec<T>;
///
///     fn recombine(
///         &self,
///         (mut first_parent, second_parent): (Vec<T>, Vec<T>),
///         _: &mut ThreadRng,
///     ) -> anyhow::Result<Vec<T>> {
///         first_parent[0] = second_parent[0];
///         Ok(first_parent)
///     }
/// }
///
/// let first_parent = vec![0, 0, 0, 0];
/// let second_parent = vec![1, 1, 1, 1];
///
/// let recombinator = SwapFirst;
/// let recombinator_result = recombinator
///     .recombine(
///         (first_parent.clone(), second_parent.clone()),
///         &mut thread_rng(),
///     )
///     .unwrap();
///
/// // Wrap the recombinator in a `Recombine` to make it an `Operator`.
/// let recombine = Recombine::new(recombinator);
/// let operator_result = recombine
///     .apply((first_parent, second_parent), &mut thread_rng())
///     .unwrap();
///
/// assert_eq!(recombinator_result, operator_result);
/// ```
///
/// Because [`Recombine`] and [`Operator`] are both [`Composable`], you can
/// chain them together using the [`Composable::then()`] method, and then
/// [`Operator::apply()`] the resulting chain to an input. In the examples below
/// we apply the chain to a pair of genomes, one all `true` and one all
/// `false`. The chain will first swap apply `SwapOne` to swap a randomly chosen
/// element (a `false`) from the second parent into the first parent, and then
/// `CountTrue` will count the number of `true` values in the resulting genome,
/// which should be 3.
///
/// ```
/// # use rand::{rngs::ThreadRng, thread_rng, Rng};
/// #
/// # use ec_core::operator::recombinator::{Recombinator, Recombine};
/// # use ec_core::operator::{Composable, Operator};
/// #
/// // A simple `Recombinator` that swaps one element from the second parent
/// // into the corresponding position in the first parent.
/// struct SwapOne;
///
/// impl<T: Copy> Recombinator<(Vec<T>, Vec<T>)> for SwapOne {
///     type Output = Vec<T>;
///
///     fn recombine(
///         &self,
///         (mut first_parent, second_parent): (Vec<T>, Vec<T>),
///         rng: &mut ThreadRng,
///     ) -> anyhow::Result<Vec<T>> {
///         let index = rng.gen_range(0..first_parent.len());
///         first_parent[index] = second_parent[index];
///         Ok(first_parent)
///     }
/// }
///
/// // A simple `Operator` that takes a `Vec<bool>` and returns the number
/// // of `true` values in the genome.
/// struct CountTrue;
///
/// impl Operator<Vec<bool>> for CountTrue {
///     type Output = usize;
///     type Error = anyhow::Error;
///
///     fn apply(&self, genome: Vec<bool>, _: &mut ThreadRng) -> Result<Self::Output, Self::Error> {
///         Ok(genome.iter().filter(|&&x| x).count())
///     }
/// }
/// impl Composable for CountTrue {}
///
/// // If we swap in exactly one value from the `second_parent` we
/// // should have 3 `true` values in the resulting genome.
/// let first_parent = vec![true, true, true, true];
/// let second_parent = vec![false, false, false, false];
///
/// let recombinator = SwapOne;
/// let recombine = Recombine::new(recombinator);
/// let count_true = CountTrue;
/// let chain = recombine.then(count_true);
///
/// let count = chain
///     .apply((first_parent, second_parent), &mut thread_rng())
///     .unwrap();
/// assert_eq!(count, 3);
/// ```
///
/// We can also pass a _reference_ to a [`Recombinator`] (i.e.,
/// `&Recombinator`) to a [`Recombine`], allowing us to pass references to
/// recombinators into chains of operators without requiring or giving up
/// ownership of the recombinator.
///
/// ```
/// # use rand::{rngs::ThreadRng, thread_rng, Rng};
/// #
/// # use ec_core::operator::recombinator::{Recombinator, Recombine};
/// # use ec_core::operator::{Composable, Operator};
/// #
/// # // A simple `Recombinator` that swaps one element from the second parent
/// # // into the corresponding position in the first parent.
/// # struct SwapOne;
/// #
/// # impl<T: Copy> Recombinator<(Vec<T>, Vec<T>)> for SwapOne {
/// #     type Output = Vec<T>;
/// #
/// #     fn recombine(
/// #         &self,
/// #         (mut first_parent, second_parent): (Vec<T>, Vec<T>),
/// #         rng: &mut ThreadRng,
/// #     ) -> anyhow::Result<Vec<T>> {
/// #         let index = rng.gen_range(0..first_parent.len());
/// #         first_parent[index] = second_parent[index];
/// #         Ok(first_parent)
/// #     }
/// # }
/// #
/// # // A simple `Operator` that takes a `Vec<bool>` and returns the number
/// # // of `true` values in the genome.
/// # struct CountTrue;
/// #
/// # impl Operator<Vec<bool>> for CountTrue {
/// #     type Output = usize;
/// #     type Error = anyhow::Error;
/// #
/// #     fn apply(&self, genome: Vec<bool>, _: &mut ThreadRng) ->
/// # Result<Self::Output, Self::Error> {         Ok(genome.iter().filter(|&&x|
/// # x).count())     }
/// # }
/// # impl Composable for CountTrue {}
/// #
/// # // If we swap in exactly one value from the `second_parent` we
/// # // should have 3 `true` values in the resulting genome.
/// let first_parent = vec![true, true, true, true];
/// let second_parent = vec![false, false, false, false];
///
/// let recombine = Recombine::new(&SwapOne);
/// let chain = recombine.then(CountTrue);
///
/// let count = chain
///     .apply((first_parent, second_parent), &mut thread_rng())
///     .unwrap();
/// assert_eq!(count, 3);
/// ```
pub struct Recombine<R> {
    /// The wrapped [`Recombinator`] that this [`Recombine`] will apply
    recombinator: R,
}

impl<R> Recombine<R> {
    pub const fn new(recombinator: R) -> Self {
        Self { recombinator }
    }
}

impl<R, G> Operator<G> for Recombine<R>
where
    R: Recombinator<G>,
{
    type Output = R::Output;
    type Error = anyhow::Error;

    /// Apply the wrapped [`Recombinator`] as an [`Operator`] to the given
    /// genomes.
    fn apply(&self, genomes: G, rng: &mut ThreadRng) -> Result<Self::Output, Self::Error> {
        self.recombinator.recombine(genomes, rng)
    }
}
impl<R> Composable for Recombine<R> {}

/// Implement [`Recombinator`] for a reference to a [`Recombinator`].
/// This allows us to wrap a reference to a [`Recombinator`] in a [`Recombine`]
/// operator, allowing recombinators to be used in chains of operators.
impl<R, GS> Recombinator<GS> for &R
where
    R: Recombinator<GS>,
{
    type Output = R::Output;

    fn recombine(&self, genomes: GS, rng: &mut ThreadRng) -> anyhow::Result<Self::Output> {
        (**self).recombine(genomes, rng)
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "Panicking is the best way to deal with errors in unit tests"
)]
#[cfg(test)]
mod tests {
    use rand::{Rng, rngs::ThreadRng, thread_rng};

    use super::{Recombinator, Recombine};
    use crate::operator::{Composable, Operator};

    // A simple `Recombinator` that swaps one element from the second parent
    // into the corresponding position in the first parent.
    struct SwapOne;

    impl<T: Copy> Recombinator<(Vec<T>, Vec<T>)> for SwapOne {
        type Output = Vec<T>;

        fn recombine(
            &self,
            (mut first_parent, second_parent): (Vec<T>, Vec<T>),
            rng: &mut ThreadRng,
        ) -> anyhow::Result<Vec<T>> {
            let index = rng.gen_range(0..first_parent.len());
            first_parent[index] = second_parent[index];
            Ok(first_parent)
        }
    }

    // A simple `Operator` that takes a `Vec<bool>` and returns the number
    // of `true` values in the genome.
    struct CountTrue;

    impl Operator<Vec<bool>> for CountTrue {
        type Output = usize;
        type Error = anyhow::Error;

        fn apply(&self, genome: Vec<bool>, _: &mut ThreadRng) -> Result<Self::Output, Self::Error> {
            Ok(genome.iter().filter(|&&x| x).count())
        }
    }
    impl Composable for CountTrue {}

    #[test]
    fn swap_one() {
        let first_parent = vec![0, 0, 0, 0];
        let second_parent = vec![1, 1, 1, 1];
        let child = SwapOne
            .recombine((first_parent, second_parent), &mut thread_rng())
            .unwrap();
        // `child` should be all zeros except for the one place where
        // a one was swapped in from the second parent.
        let num_zeros = child.iter().filter(|&&x| x == 0).count();
        let num_ones = child.iter().filter(|&&x| x == 1).count();
        assert_eq!(num_zeros, 3);
        assert_eq!(num_ones, 1);
    }

    #[test]
    fn can_wrap_recombinator() {
        let first_parent = vec![0, 0, 0, 0];
        let second_parent = vec![1, 1, 1, 1];

        let recombinator = SwapOne;
        // Wrap the recombinator in a `Recombine` to make it an `Operator`.
        let recombine = Recombine::new(recombinator);

        let child = recombine
            .apply((first_parent, second_parent), &mut thread_rng())
            .unwrap();
        let num_zeros = child.iter().filter(|&&x| x == 0).count();
        let num_ones = child.iter().filter(|&&x| x == 1).count();
        assert_eq!(num_zeros, 3);
        assert_eq!(num_ones, 1);
    }

    #[test]
    fn can_wrap_recombinator_reference() {
        let first_parent = vec![0, 0, 0, 0];
        let second_parent = vec![1, 1, 1, 1];

        let recombinator = SwapOne;
        // Wrap a reference to the recombinator in a `Recombine` to make it an
        // `Operator`.
        let recombine = Recombine::new(&recombinator);

        let child = recombine
            .apply((first_parent, second_parent), &mut thread_rng())
            .unwrap();
        let num_zeros = child.iter().filter(|&&x| x == 0).count();
        let num_ones = child.iter().filter(|&&x| x == 1).count();
        assert_eq!(num_zeros, 3);
        assert_eq!(num_ones, 1);
    }

    #[test]
    fn can_chain_recombinator_and_operator() {
        // If we swap in exactly one value from the `second_parent` we
        // should have 3 `true` values in the resulting genome.
        let first_parent = vec![true, true, true, true];
        let second_parent = vec![false, false, false, false];

        let recombinator = SwapOne;
        let recombine = Recombine::new(recombinator);
        let count_true = CountTrue;
        let chain = recombine.then(count_true);

        let count = chain
            .apply((first_parent, second_parent), &mut thread_rng())
            .unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn can_chain_with_recombinator_reference() {
        // If we swap in exactly one value from the `second_parent` we
        // should have 3 `true` values in the resulting genome.
        let first_parent = vec![true, true, true, true];
        let second_parent = vec![false, false, false, false];

        let recombinator = SwapOne;
        // Wrap a reference to the recombinator in a `Recombine` to make it an
        // `Operator`.
        let recombine = Recombine::new(&recombinator);
        let count_true = CountTrue;
        let chain = recombine.then(count_true);

        let count = chain
            .apply((first_parent, second_parent), &mut thread_rng())
            .unwrap();
        assert_eq!(count, 3);
    }
}
