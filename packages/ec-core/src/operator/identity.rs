use std::convert::Infallible;

use rand::rngs::ThreadRng;

use super::{Composable, Operator};

/// An [`Operator`] that always returns its input value.
///
/// # See also
///
/// See [`Operator`] and [`Composable`].
///
/// # Examples
///
/// ```
/// # use ec_core::operator::{Operator, identity::Identity};
/// # use rand::rng;
/// #
/// let mut rng = rng();
/// // This will always return the value that is passed in.
/// let identity = Identity;
///
/// assert_eq!(identity.apply(3, &mut rng).unwrap(), 3);
/// assert_eq!(identity.apply("string", &mut rng).unwrap(), "string");
/// assert_eq!(identity.apply(Some(7), &mut rng).unwrap(), Some(7));
/// ```
pub struct Identity;

impl<T> Operator<T> for Identity {
    /// The output type (and value) of this [`Operator`] is the same as that of
    /// whatever input is passed in.
    type Output = T;
    /// This [`Operator`] can't fail
    type Error = Infallible;

    /// Always return the input value.
    fn apply(&self, input: T, _: &mut ThreadRng) -> Result<Self::Output, Self::Error> {
        Ok(input)
    }
}

impl Composable for Identity {}

#[cfg(test)]
mod tests {
    use rand::rng;

    use crate::operator::{Operator, identity::Identity};

    #[test]
    fn returns_input() {
        let mut rng = rng();
        // This should always return its input.
        let identity = Identity;

        assert_eq!(identity.apply(3, &mut rng).unwrap(), 3);
        assert_eq!(identity.apply("string", &mut rng).unwrap(), "string");
        assert_eq!(identity.apply(Some(7), &mut rng).unwrap(), Some(7));
    }
}
