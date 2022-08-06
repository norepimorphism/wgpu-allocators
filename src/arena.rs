use wgpu::BufferAddress;

use std::ops::{Index, IndexMut, Range};

use crate::{Allocator, Heap, HeapUsages, NonZeroBufferAddress};

/// A user-provided function that calculates the size, in bytes, of a new heap given a
/// [`NewHeapSizeContext`].
type CalculateNewHeapSize = fn(NewHeapSizeContext) -> NonZeroBufferAddress;

/// Context for calculating the size, in bytes, of a new heap.
///
/// Such a context is passed to a [`CalculateNewHeapSize`].
pub struct NewHeapSizeContext {
    /// The size, in bytes, of the first allocation to be made on the new heap.
    ///
    /// The [`CalculateNewHeapSize`] that this context is passed to must produce a size greater than
    /// or equal to this value.
    pub first_alloc_size: NonZeroBufferAddress,
}

fn classify_size(size: NonZeroBufferAddress) -> usize {
    let size = size.get();

    // This tells us how many zeros are on the left-side of the binary representation of `size`, but
    // it *also* tells us how many bits are *not* leading zeros&mdash;we just have to subtract this
    // value from the total number of bits in `size`.
    let leading_zeros = size.leading_zeros();
    let total_bits = 8 * std::mem::size_of_val(&size);
    // SAFETY: The number of leading zeros in `size` cannot exceed the total number of bits.
    let not_leading_zeros = unsafe {
        // Note: it's OK to cast `leading_zeros` to `usize` as it can't possibly overflow `usize` on
        // any system&mdash;we're not dealing with 512-bit integers here.
        total_bits.unchecked_sub(leading_zeros as usize)
    };

    // If `not_leading_zeros` is the number of bits that aren't leading zeros, then
    // `not_leading_zeros` must be the zero-based index of the leftmost 1 bit.
    // SAFETY: `size` is based on a `NonZeroBufferAddress`, so it must be nonzero.
    unsafe { not_leading_zeros.unchecked_sub(1) }
}

impl<A> Default for SizePool<A> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

/// A set of heaps and associated allocators in the same size class.
///
/// In a [`HeapArena`], contained heaps and allocators are stored in pools based on size, allowing
/// for a more performant allocation algorithm than a naive linear search. Specifically, each pool
/// is assigned a *size class*, which is the position of the leftmost 1 bit in the binary
/// representation of a heap's size&mdash;in other words, the size class is the exponent `n` where
/// the size of a heap rounded-down to the nearest power of 2 is `2^n`.
///
/// There is an exception to this&mdash;[`HeapArena::tiny_pool`], which is for heaps and allocators
/// of size 1 to 4,096 bytes (exclusive). Another way of thinking about this is that it contains
/// heaps and allocators from size classes 0 to 11 (inclusive).
#[derive(Debug)]
struct SizePool<A>(Vec<(Heap, A)>);

impl<A> HeapArena<A> {
    /// Creates a new `HeapArena`.
    pub fn new(
        usage: HeapUsages,
        calc_new_heap_size: CalculateNewHeapSize,
    ) -> Self {
        Self {
            tiny_pool: SizePool::default(),
            size_pools: Vec::new(),
            usage,
            calc_new_heap_size,
        }
    }
}

#[derive(Debug)]
pub struct HeapArena<A> {
    /// A [`SizePool`] for heaps and allocators of size 1 to 4,096 bytes (inclusive).
    ///
    /// This is separated from [`Self::size_pools`] as it seemed silly to allocate pools for size
    /// classes of 0, 1, 2, etc., which represent very small heaps that should probably never be
    /// created in practice.
    tiny_pool: SizePool<A>,
    /// The size pools of heaps and allocators that make up this arena's backing storage.
    ///
    /// See [`SizePool`] for details on how a size pool is laid out internally.
    ///
    /// This field orders pools from lowest to highest size class, beginning at 12. Therefore, index
    /// 0 is for heaps of size 4,096 to 8,192 bytes (exclusive), index 1 is for heaps of size 8,192
    /// to 16,384 bytes (exclusive), and so on.
    size_pools: Vec<SizePool<A>>,
    /// The usage for all heaps within this arena.
    usage: HeapUsages,
    /// Calculates the size of a new heap created by [`Self::expand`].
    calc_new_heap_size: CalculateNewHeapSize,
}

impl<A: Allocator> HeapArena<A> {
    pub fn unmap(&self) {
        for (heap, _) in self.tiny_pool.0.iter() {
            heap.unmap();
        }
        for pool in self.size_pools.iter() {
            for (heap, _) in pool.0.iter() {
                heap.unmap();
            }
        }
    }

    pub fn alloc(
        &mut self,
        device: &wgpu::Device,
        size: NonZeroBufferAddress,
        alignment: NonZeroBufferAddress,
    ) -> Allocation {
        let size_class = classify_size(size);
        let pool = if size_class < 12 {
            &mut self.tiny_pool
        } else {
            // SAFETY: `size_class` is at least 12, so this will never underflow.
            let index = unsafe { size_class.unchecked_sub(12) };

            &mut self.size_pools[index]
        };

        Self::alloc_in_pool(
            device,
            pool,
            size,
            size_class,
            alignment,
            self.usage,
            self.calc_new_heap_size,
        )
    }

    fn alloc_in_pool(
        device: &wgpu::Device,
        pool: &mut SizePool<A>,
        size: NonZeroBufferAddress,
        size_class: usize,
        alignment: NonZeroBufferAddress,
        heap_usage: HeapUsages,
        calc_new_heap_size: CalculateNewHeapSize,
    ) -> Allocation {
        for (index_in_pool, (_, allocator)) in pool
            .0
            .iter_mut()
            .rev()
            .enumerate()
        {
            if let Some(range_in_heap) = allocator.alloc(size, alignment) {
                return Allocation {
                    arena_key: ArenaKey { size_class, index_in_pool },
                    range_in_heap,
                };
            }
        }

        // None of the existing heaps can hold our allocation, so we'll have to create a new one.

        let new_heap_size = (calc_new_heap_size)(NewHeapSizeContext {
            first_alloc_size: size,
        });
        if new_heap_size < size {
            panic!(
                "heap size is too small; must be able to store first allocation of size {} bytes",
                size.get(),
            );
        }

        let (_, allocator) = pool.expand(device, new_heap_size, heap_usage);
        let range_in_heap = allocator.alloc(size, alignment).unwrap();

        Allocation {
            arena_key: ArenaKey {
                size_class,
                // SAFETY: We just appended to this pool, so its length must be nonzero.
                index_in_pool: unsafe { pool.0.len().unchecked_sub(1) },
            },
            range_in_heap,
        }
    }
}

impl<A: Allocator> SizePool<A> {
    fn expand(
        &mut self,
        device: &wgpu::Device,
        new_heap_size: NonZeroBufferAddress,
        usage: HeapUsages,
    ) -> &mut (Heap, A) {
        let heap = Heap::new(device, new_heap_size, usage);
        let allocator = A::new(&heap);
        self.0.push((heap, allocator));

        // SAFETY: We just pushed a new heap/allocator pair.
        unsafe { self.0.last_mut().unwrap_unchecked() }
    }
}

#[derive(Debug)]
pub struct Allocation {
    pub arena_key: ArenaKey,
    /// The result from [`Allocator::alloc`]. To be used with the heap represented by
    /// [`Self::arena_key`].
    pub range_in_heap: Range<BufferAddress>,
}

#[derive(Debug)]
pub struct ArenaKey {
    size_class: usize,
    index_in_pool: usize,
}

impl<A> Index<ArenaKey> for HeapArena<A> {
    type Output = (Heap, A);

    fn index(&self, key: ArenaKey) -> &Self::Output {
        if key.size_class < 12 {
            &self.tiny_pool.0[key.index_in_pool]
        } else {
            // SAFETY: `size_class` is at least 12, so this will never underflow.
            let pool = &self.size_pools[unsafe { key.size_class.unchecked_sub(12) }];

            &pool.0[key.index_in_pool]
        }
    }
}

impl<A> IndexMut<ArenaKey> for HeapArena<A> {
    fn index_mut(&mut self, key: ArenaKey) -> &mut Self::Output {
        if key.size_class < 12 {
            &mut self.tiny_pool.0[key.index_in_pool]
        } else {
            // SAFETY: `size_class` is at least 12, so this will never underflow.
            let pool = &mut self.size_pools[unsafe {
                key.size_class.unchecked_sub(12)
            }];

            &mut pool.0[key.index_in_pool]
        }
    }
}
