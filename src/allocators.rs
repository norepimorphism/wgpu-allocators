use wgpu::BufferAddress;

use std::ops::Range;

use crate::{Allocator, Deallocator, NonZeroBufferAddress};

pub struct Stack {
    pointer: BufferAddress,
}

impl Allocator for Stack {
    fn new(size: NonZeroBufferAddress) -> Self {
        Self { pointer: size.get() }
    }

    fn alloc(
        &mut self,
        size: NonZeroBufferAddress,
        alignment: NonZeroBufferAddress,
    ) -> Option<Range<BufferAddress>> {
        let end = self.pointer;
        self.pointer = end.checked_sub(size.get())? & create_alignment_bitmask(alignment);

        Some(self.pointer..end)
    }
}

impl Deallocator for Stack {
    fn dealloc(&mut self, range: Range<BufferAddress>) -> Result<(), ()> {
        if range.start == self.pointer {
            // Because, during normal operation, no two overlapping allocations will ever exist, we
            // know that, if a range from a given allocation begins at `self.pointer`, it must be
            // the most recent allocation. We don't even need to check the end of the range.

            self.pointer = range.end;

            Ok(())
        } else {
            // The given range does not represent the most recent allocation, so it cannot be
            // deallocated yet.
            Err(())
        }
    }
}

fn create_alignment_bitmask(alignment: NonZeroBufferAddress) -> u64 {
    // SAFETY: `alignment` is a nonzero unsigned integer, so its value must be greater than or equal
    // to 1. Thus, subtracting one will never result in underflow.
    !unsafe { alignment.get().unchecked_sub(1) }
}
