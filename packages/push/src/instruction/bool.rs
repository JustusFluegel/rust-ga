use crate::push_vm::{
    stack::traits::{
        has_stack::{HasStack, HasStackMut},
        pop::{PopHead, PopHeadIn},
        push::{AttemptPushHead, PushHead, PushHeadIn},
        size::{SizeLimit, SizeLimitOf, StackSize},
        TypedStack,
    },
    state::with_state::{AddState, WithStateOps},
};

use super::{
    Instruction, InstructionResult, PushInstruction, PushInstructionError, SpecifyFatality,
};
use std::ops::Not;
use strum_macros::EnumIter;

#[derive(Debug, strum_macros::Display, Clone, PartialEq, Eq, EnumIter)]
pub enum BoolInstruction {
    Push(bool),
    Not,
    Or,
    And,
    Xor,
    Implies,
    // Do we really want either of these? Do they get used?
    // BooleanInvertFirstThenAnd,
    // BoolInvertSecondThenAnd,
    FromInt,
    // BoolFromFloat,
}

#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum BoolInstructionError {}

impl<S> Instruction<S> for BoolInstruction
where
    S: Clone + HasStackMut<bool> + HasStackMut<i64>,
    <S as HasStack<bool>>::StackType:
        TypedStack<Item = bool> + PushHead + PopHead + StackSize + SizeLimit,
    <S as HasStack<i64>>::StackType: TypedStack<Item = i64> + PopHead,
{
    type Error = PushInstructionError;

    // TODO: This only "works" because all the stack operations are "transactional",
    //   i.e., things like `pop2()` either completely succeed or return an error without
    //   modifying the (mutable) state. (This is done by checking that the size of the
    //   relevant stack is big enough before removing any elements.) If any stack operations
    //   were _not_ "transactional" then we could end up passing an inconsistent state
    //   to the call to `Error::recoverable_error()`, which would be bad. Because the `pop`
    //   and `push` calls aren't together, we can still have inconsistent states in the
    //   call to `Error::fatal_error()`. For example, if the boolean is full and the
    //   instruction is `BoolFromInt`, we could pop off an integer before we realize there's
    //   no room to push on the new boolean. We can special case that, but the burden lies
    //   on the programmer, with no help from the type system.

    /*
    // Get the nth character from a string and push it on the char stack.
    //   - n comes from the int stack
    //   - string comes from the string stack
    //   - result goes on the char stack
    let transaction: Transaction<PushState> = state.transaction();

    let string = transaction.try_pop<String>()?; // This version has the transaction be mutable.
    let (string, transaction) = transaction.try_pop<String>()?; // This version returns a "new" transaction.

    let (index, transaction) = transaction.try_pop<PushInteger>()?;
    let c = string.chars.nth(index)?;
    let transaction = transaction.try_push<char>(c)?;
    let new_state = transaction.close()?; // Can closing actually fail?
     */

    // [pop string] then [pop integer] contains a closure with a tuple of (string, int)

    // state.transaction().pop::<String>().with_min_length(1).and_pop::<Integer>().then_push::<Char>(|(s, i)| s.chars.nth(i))
    // state.transaction().pop::<String>().with_min_length(1).and_pop::<Integer>().charAt().then_push::<Char>()
    // state.transaction().pop::<String>().with_min_length(1).and_pop::<Integer>().map::<Char>(|(s, i)| s.chars.nth(i)).then_push::<Char>()
    // Then you wouldn't be able to chain on that and query what you would push onto the stack so maybe not ideal.

    // Options:
    //   - Make operations reversible (undo/redo)
    //   - Hold operations in some kind of queue and apply the at the end
    //     when we know they'll all work

    fn perform(&self, state: &mut S) -> InstructionResult<&mut S, Self::Error> {
        let bool_stack = state.stack_mut::<bool>();

        match self {
            Self::Push(b) => state.push_head_in::<bool>(*b).make_recoverable()?,
            Self::Not => bool_stack
                .pop_head()
                .map(Not::not)
                .with_state(state)
                .make_recoverable()?
                .attempt_push_head()
                .make_fatal()?,
            Self::And => bool_stack
                .pop_n_head()
                .map(|(x, y)| x && y)
                .with_state(state)
                .make_recoverable()?
                .attempt_push_head()
                .make_fatal()?,
            Self::Or => bool_stack
                .pop_n_head()
                .map(|(x, y)| x || y)
                .with_state(state)
                .make_recoverable()?
                .attempt_push_head()
                .make_fatal()?,
            Self::Xor => bool_stack
                .pop_n_head()
                .map(|(x, y)| x != y)
                .with_state(state)
                .make_recoverable()?
                .attempt_push_head()
                .make_fatal()?,
            Self::Implies => bool_stack
                .pop_n_head()
                .map(|(x, y)| !x || y)
                .with_state(state)
                .make_recoverable()?
                .attempt_push_head()
                .make_fatal()?,
            Self::FromInt => state
                .not_full::<bool>()
                .make_recoverable()?
                .pop_head_in::<i64>()
                .map_value(|i| i != 0)
                .make_fatal()?
                .attempt_push_head()
                .make_fatal()?,
        };

        Ok(())
    }
}

impl From<BoolInstruction> for PushInstruction {
    fn from(instr: BoolInstruction) -> Self {
        Self::BoolInstruction(instr)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod property_tests {
    use crate::{
        instruction::{BoolInstruction, Instruction, PushState},
        push_vm::stack::traits::{get::GetHead, size::StackSize},
    };
    use proptest::{prop_assert_eq, proptest};
    use strum::IntoEnumIterator;

    fn all_instructions() -> Vec<BoolInstruction> {
        BoolInstruction::iter().collect()
    }

    proptest! {
        #[test]
        fn ops_do_not_crash(instr in proptest::sample::select(all_instructions()),
                x in proptest::bool::ANY, y in proptest::bool::ANY, i in proptest::num::i64::ANY) {
            let state = PushState::builder([])
                .with_bool_values(vec![x, y])
                .with_int_values(vec![i])
                .build();
            let _ = instr.perform(&mut state).unwrap();
        }

        #[test]
        fn and_is_correct(x in proptest::bool::ANY, y in proptest::bool::ANY) {
            let state = PushState::builder([])
                .with_bool_values(vec![x, y])
                .build();
            BoolInstruction::And.perform(&mut state).unwrap();
            prop_assert_eq!(state.bool.size(), 1);
            prop_assert_eq!(*state.bool.head().unwrap(), x && y);
        }

        #[test]
        fn implies_is_correct(x in proptest::bool::ANY, y in proptest::bool::ANY) {
            let state = PushState::builder([])
                .with_bool_values(vec![x, y])
                .build();
            BoolInstruction::Implies.perform(&mut state).unwrap();
            prop_assert_eq!(state.bool.size(), 1);
            prop_assert_eq!(*state.bool.head().unwrap(), !x || y);
        }
    }
}
