use crate::{
    error::InstructionResult,
    push_vm::{push_state::PushState, stack::StackError, PushInteger},
};
use std::{fmt::Debug, fmt::Display, sync::Arc};

pub use self::{bool::BoolInstruction, int::IntInstruction};
use self::{bool::BoolInstructionError, int::IntInstructionError};

mod bool;
mod int;

/*
 * exec_if requires a boolean and two (additional) values on the exec stack.
 * If the bool is true, we remove the second of the two exec stack values,
 * and if it's false, we remove the first.
 */

/*
 * exec_while requires a boolean and one additional value on the exec stack.
 * If the bool is true, then you push a copy of the "body" onto the exec, followed
 * by another copy of exec_while.
 */

/*
 * Instructions that are generic over stacks:
 *
 * - push
 * - pop
 * - dup (int_dup, exec_dup, bool_dup, ...)
 */

/// Error
///
/// - `state`: The state of the system _before_ attempting to perform
///     the instruction that generated this error
/// - `error`: The cause of this error
/// - `error_kind`: Whether this error is `Fatal` (i.e., whether program execution
///     should terminate immediately) or `Recoverable` (i.e., this instruction
///     should just be skipped and the program execution continues with the
///     next instruction).
#[derive(Debug)]
pub struct RecoverableError<S, E> {
    // Without the `Box` the size of this Error ended up being 156 bytes
    // with a `PushState` and a `PushInstructionError`. That led to a Clippy
    // warning (https://rust-lang.github.io/rust-clippy/master/index.html#/result_large_err)
    // our `Error` was then larger than the 128 byte limit. They recommended boxing
    // the big piece (the state in our case), and doing that brought the size down to
    // 40 bytes. Since `Error`s are only constructed through `::fatal()` or `::recoverable()`,
    // we'd nicely encapsulated this and only had to make changes in those two places to
    // get things working.
    state: Box<S>,
    error: E,
}

impl<S, E> RecoverableError<S, E> {
    pub fn into_state(self) -> S {
        *self.state
    }

    pub fn into_fatal(self) -> FatalError<S, E> {
        FatalError {
            state: self.state,
            error: self.error,
        }
    }
}

#[derive(Debug)]
pub struct FatalError<S, E> {
    // Without the `Box` the size of this Error ended up being 156 bytes
    // with a `PushState` and a `PushInstructionError`. That led to a Clippy
    // warning (https://rust-lang.github.io/rust-clippy/master/index.html#/result_large_err)
    // our `Error` was then larger than the 128 byte limit. They recommended boxing
    // the big piece (the state in our case), and doing that brought the size down to
    // 40 bytes. Since `Error`s are only constructed through `::fatal()` or `::recoverable()`,
    // we'd nicely encapsulated this and only had to make changes in those two places to
    // get things working.
    state: Box<S>,
    error: E,
}

#[derive(Debug)]
pub enum Error<S, E> {
    Recoverable(RecoverableError<S, E>),
    Fatal(FatalError<S, E>),
}

pub type InstructionResult<S, E> = core::result::Result<S, Error<S, E>>;

impl<S, E> Error<S, E> {
    pub fn fatal(state: S, error: impl Into<E>) -> Self {
        Self::Fatal(FatalError {
            state: Box::new(state),
            error: error.into(),
        })
    }

    pub fn recoverable(state: S, error: impl Into<E>) -> Self {
        Self::Recoverable(RecoverableError {
            state: Box::new(state),
            error: error.into(),
        })
    }

    pub const fn is_recoverable(&self) -> bool {
        matches!(self, Self::Recoverable(_))
    }

    pub const fn is_fatal(&self) -> bool {
        matches!(self, Self::Fatal(_))
    }

    pub fn state(&self) -> &S {
        match self {
            Self::Recoverable(error) => &error.state,
            Self::Fatal(error) => &error.state,
        }
    }

    pub const fn error(&self) -> &E {
        match self {
            Self::Recoverable(error) => &error.error,
            Self::Fatal(error) => &error.error,
        }
    }

    pub fn into_state(self) -> S {
        match self {
            Self::Recoverable(error) => *error.state,
            Self::Fatal(error) => *error.state,
        }
    }
}

impl<S, E> From<RecoverableError<S, E>> for Error<S, E> {
    fn from(value: RecoverableError<S, E>) -> Self {
        Self::Recoverable(value)
    }
}
impl<S, E> From<FatalError<S, E>> for Error<S, E> {
    fn from(value: FatalError<S, E>) -> Self {
        Self::Fatal(value)
    }
}

pub trait MakeError<E, S, V>: Sized {
    fn make_fatal(self, state: impl Into<Box<S>>) -> Result<V, FatalError<S, E>>;
    fn make_recoverable(self, state: impl Into<Box<S>>) -> Result<V, RecoverableError<S, E>>;
}

impl<E, S, V> MakeError<E, S, V> for Result<V, E> {
    fn make_fatal(self, state: impl Into<Box<S>>) -> Result<V, FatalError<S, E>> {
        self.map_err(|error| FatalError {
            state: state.into(),
            error,
        })
    }

    fn make_recoverable(self, state: impl Into<Box<S>>) -> Result<V, RecoverableError<S, E>> {
        self.map_err(|error| RecoverableError {
            state: state.into(),
            error,
        })
    }
}

pub trait TryRecover<T> {
    type Error;

    /// # Errors
    ///
    /// `x.try_recover()` returns an error if `x` is not a `Recoverable` error type.
    fn try_recover(self) -> Result<T, Self::Error>;
}

impl<S, E> TryRecover<S> for Result<S, Error<S, E>> {
    type Error = FatalError<S, E>;

    fn try_recover(self) -> Result<S, FatalError<S, E>> {
        match self {
            Ok(s) => Ok(s),
            Err(Error::Recoverable(s)) => Ok(s.into_state()),
            Err(Error::Fatal(error)) => Err(error),
        }
    }
}

impl<S, E> TryRecover<S> for Result<S, RecoverableError<S, E>> {
    type Error = Infallible;

    fn try_recover(self) -> Result<S, Infallible> {
        match self {
            Ok(s) => Ok(s),
            Err(s) => Ok(s.into_state()),
        }
    }
}

#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum PushInstructionError {
    #[error(transparent)]
    StackError(#[from] StackError),
    #[error("Exceeded the maximum step limit {step_limit}")]
    StepLimitExceeded { step_limit: usize },
    #[error(transparent)]
    Int(#[from] IntInstructionError),
    #[error(transparent)]
    Bool(#[from] BoolInstructionError),
}

/// Maps a (presumably error) type into an `InstructionResult`.
/// This is in fact used to convert `InstructionResult<S, E1>`
/// into `InstructionResult<S, E2>`, i.e. do `map_err()` on
/// the inner error types of an `InstructionResult`, preserving
/// the other fields in `Error`.
pub trait MapInstructionError<S, E> {
    ///  
    /// # Errors
    ///
    /// This always returns an error type.
    fn map_err_into(self) -> InstructionResult<S, E>;
}

// MizardX@Twitch's initial suggestion here had `E2` as a generic on the
// _function_ `map_err_into()` instead of at the `impl` level. That provided
// some additional flexibility, although it wasn't that we would use it.
// The current approach (suggested by esitsu@Twitch) simplified the
// `MapInstructionError` trait in a nice way, so I went with that.
impl<S, E1, E2> MapInstructionError<S, E2> for InstructionResult<S, E1>
where
    E1: Into<E2>,
{
    fn map_err_into(self) -> InstructionResult<S, E2> {
        self.map_err(|e| e.map_inner_err(Into::into))
    }
}

pub trait Instruction<S> {
    type Error;

    /// # Errors
    ///
    /// This returns an error if the instruction being performed
    /// returns some kind of error. This could include things like
    /// stack over- or underflows, or numeric errors like integer overflow.
    fn perform(&self, state: S) -> InstructionResult<S, Self::Error>;
}

impl<S, E> Instruction<S> for Box<dyn Instruction<S, Error = E>> {
    type Error = E;

    fn perform(&self, state: S) -> InstructionResult<S, E> {
        self.as_ref().perform(state)
    }
}

// impl<F> Instruction for F
// where
//     F: Fn(dyn State) -> dyn State
// {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariableName(Arc<str>);

impl From<&str> for VariableName {
    fn from(s: &str) -> Self {
        Self(Arc::from(s))
    }
}

impl Display for VariableName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod variable_name_test {
    use std::collections::HashMap;

    use super::*;

    #[test]
    #[allow(clippy::unwrap_used)]
    fn variable_name() {
        let x = VariableName::from("x");
        let x2 = VariableName::from("x");
        assert_eq!(x, x2);
        let y = VariableName::from("y");
        assert_ne!(x, y);

        let mut map = HashMap::new();
        map.insert(x.clone(), 5);
        map.insert(y.clone(), 7);

        assert_eq!(map.get(&x).unwrap(), &5);
        assert_eq!(map.get(&y).unwrap(), &7);
        assert_eq!(map.len(), 2);

        assert_eq!(map.get(&x2).unwrap(), &5);

        let z = VariableName::from("z");
        assert_eq!(map.get(&z), None);
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum PushInstruction {
    InputVar(VariableName),
    BoolInstruction(BoolInstruction),
    IntInstruction(IntInstruction),
}

impl Debug for PushInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputVar(arg0) => write!(f, "{arg0}"),
            Self::BoolInstruction(arg0) => write!(f, "Bool-{arg0}"),
            Self::IntInstruction(arg0) => write!(f, "Int-{arg0}"),
        }
    }
}

impl PushInstruction {
    #[must_use]
    pub fn push_bool(b: bool) -> Self {
        BoolInstruction::Push(b).into()
    }

    #[must_use]
    pub fn push_int(i: PushInteger) -> Self {
        IntInstruction::Push(i).into()
    }
}

impl Instruction<PushState> for PushInstruction {
    type Error = PushInstructionError;

    fn perform(&self, state: PushState) -> InstructionResult<PushState, Self::Error> {
        match self {
            Self::InputVar(var_name) => {
                // TODO: Should `push_input` return the new state?
                //   Or add a `with_input` that returns the new state and keep `push_input`?
                state.with_input(var_name)
            }
            Self::BoolInstruction(i) => i.perform(state),
            Self::IntInstruction(i) => i.perform(state),
        }
    }
}
