use ordered_float::OrderedFloat;
use printing::{PrintNewline, PrintPeriod, PrintSpace, PrintString};

pub use self::{
    bool::BoolInstruction,
    exec::ExecInstruction,
    float::FloatInstruction,
    int::{IntInstruction, IntInstructionError},
};
use self::{instruction_error::PushInstructionError, variable_name::VariableName};
use crate::{
    error::{InstructionResult, MapInstructionError},
    push_vm::push_state::PushState,
};

pub mod common;
pub mod instruction_error;
pub mod printing;
pub mod variable_name;

mod bool;
mod exec;
mod float;
mod int;

/*
 * exec_if requires a boolean and two (additional) values on the exec stack.
 * If the bool is true, we remove the second of the two exec stack values,
 * and if it's false, we remove the first.
 */

/*
 * exec_while requires a boolean and one additional value on the exec stack.
 * If the bool is true, then you push a copy of the "body" onto the exec,
 * followed by another copy of exec_while.
 */

/*
 * Instructions that are generic over stacks:
 *
 * - push
 * - pop
 * - dup (int_dup, exec_dup, bool_dup, ...)
 */

pub trait Instruction<S> {
    type Error;

    /// # Errors
    ///
    /// This returns an error if the instruction being performed
    /// returns some kind of error. This could include things like
    /// stack over- or underflows, or numeric errors like integer overflow.
    fn perform(&self, state: S) -> InstructionResult<S, Self::Error>;
}

static_assertions::assert_obj_safe!(Instruction<(), Error = ()>);

impl<S, E> Instruction<S> for Box<dyn Instruction<S, Error = E>> {
    type Error = E;

    fn perform(&self, state: S) -> InstructionResult<S, E> {
        self.as_ref().perform(state)
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
#[non_exhaustive]
pub enum PushInstruction {
    InputVar(VariableName),
    Exec(ExecInstruction),
    BoolInstruction(BoolInstruction),
    IntInstruction(IntInstruction),
    FloatInstruction(FloatInstruction),
    PrintSpace(PrintSpace),
    PrintNewline(PrintNewline),
    PrintPeriod(PrintPeriod),
    PrintString(PrintString),
}

impl PushInstruction {
    #[must_use]
    pub fn push_bool(b: bool) -> Self {
        BoolInstruction::push(b).into()
    }

    #[must_use]
    pub fn push_int(i: i64) -> Self {
        IntInstruction::push(i).into()
    }

    // #[must_use]
    // pub fn push_float(f: f64) -> Self {
    //     FloatInstruction::push(f).into()
    // }

    #[must_use]
    pub fn push_float(f: OrderedFloat<f64>) -> Self {
        FloatInstruction::push_ordered_float(f).into()
    }
}

impl Instruction<PushState> for PushInstruction {
    type Error = PushInstructionError;

    fn perform(&self, state: PushState) -> InstructionResult<PushState, Self::Error> {
        match self {
            Self::InputVar(var_name) => {
                // TODO: Should `push_input` return the new state?
                // Or add a `with_input` that returns the new state and keep `push_input`?
                state.with_input(var_name)
            }
            Self::Exec(i) => i.perform(state),
            Self::BoolInstruction(i) => i.perform(state),
            Self::IntInstruction(i) => i.perform(state),
            Self::FloatInstruction(i) => i.perform(state),
            Self::PrintString(i) => i.perform(state).map_err_into(),
            Self::PrintSpace(i) => i.perform(state).map_err_into(),
            Self::PrintNewline(i) => i.perform(state).map_err_into(),
            Self::PrintPeriod(i) => i.perform(state).map_err_into(),
        }
    }
}

pub trait NumOpens {
    fn num_opens(&self) -> usize {
        0
    }
}

static_assertions::assert_obj_safe!(NumOpens);

impl NumOpens for PushInstruction {
    fn num_opens(&self) -> usize {
        match self {
            Self::Exec(i) => i.num_opens(),
            _ => 0,
        }
    }
}

impl std::fmt::Display for PushInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputVar(instruction) => write!(f, "{instruction}"),
            Self::Exec(instruction) => write!(f, "Exec-{instruction}"),
            Self::BoolInstruction(instruction) => write!(f, "Bool-{instruction}"),
            Self::IntInstruction(instruction) => write!(f, "Int-{instruction}"),
            Self::FloatInstruction(instruction) => write!(f, "Float-{instruction}"),
            Self::PrintString(instruction) => write!(f, "PrintString({})", instruction.0),
            Self::PrintSpace(_) => write!(f, "PrintSpace"),
            Self::PrintNewline(_) => write!(f, "PrintNewline"),
            Self::PrintPeriod(_) => write!(f, "PrintPeriod"),
        }
    }
}
