//! High-level allocators for WGPU.

#![feature(unchecked_math)]

mod allocators;

use wgpu::{BufferAddress, BufferUsages};

use std::ops::Range;

pub use allocators::*;

pub type NonZeroBufferAddress = std::num::NonZeroU64;

pub struct Allocation {
    range: Range<BufferAddress>,
}

impl Allocation {
    pub fn slice<'a, A>(&self, heap: &'a Heap<A>) -> wgpu::BufferSlice<'a> {
        heap.staging_buffer.slice(self.range.clone())
    }

    pub fn flush<A>(&self, heap: &Heap<A>, encoder: &mut wgpu::CommandEncoder) {
        flush(heap, encoder, self.range.clone());
    }
}

pub trait Allocator {
    fn new(size: NonZeroBufferAddress) -> Self;

    fn alloc(
        &mut self,
        size: NonZeroBufferAddress,
        alignment: NonZeroBufferAddress,
    ) -> Option<Range<BufferAddress>>;
}

pub trait Deallocator: Allocator {
    fn dealloc(&mut self, range: Range<BufferAddress>) -> Result<(), ()>;
}

impl<A: Allocator> Heap<A> {
    /// Creates a new `Heap` with the given allocator.
    ///
    /// # Safety
    ///
    /// The `new` function itself does not require unsafety *per se*; rather, the `unsafe` modifier
    /// guards usage of the returned `Heap`. In particular, it is UB to mix-and-match allocations
    /// returned by [`Heap::alloc`] between different heaps&mdash;[`Heap::alloc`] returns an
    /// allocation that is only suitable for the heap it was called on. While there are mechanisms
    /// to enforce this rule from `wgpu-allocations` at runtime, it was easiest as well as zero-cost
    /// to place the burden of enforcement on consumers of this crate.
    pub unsafe fn new(
        device: &wgpu::Device,
        size: NonZeroBufferAddress,
    ) -> Self {
        Self {
            staging_buffer: create_buffer(
                device,
                size.get(),
                BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
            ),
            gpu_buffer: create_buffer(
                device,
                size.get(),
                BufferUsages::COPY_DST,
            ),
            allocator: A::new(size),
            size,
        }
    }
}

fn create_buffer(device: &wgpu::Device, size: u64, usage: BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage,
        mapped_at_creation: false,
    })
}

pub struct Heap<A> {
    staging_buffer: wgpu::Buffer,
    gpu_buffer: wgpu::Buffer,
    allocator: A,
    size: NonZeroBufferAddress,
}

impl<A: Allocator> Heap<A> {
    pub fn alloc(
        &mut self,
        size: NonZeroBufferAddress,
        alignment: NonZeroBufferAddress,
    ) -> Option<Allocation> {
        self.allocator.alloc(size, alignment).map(|range| Allocation { range })
    }
}

impl<A: Deallocator> Heap<A> {
    pub fn dealloc(
        &mut self,
        allocation: Allocation,
    ) -> Result<(), ()> {
        self.allocator.dealloc(allocation.range)
    }
}

impl<A> Heap<A> {
    pub fn flush(&self, encoder: &mut wgpu::CommandEncoder) {
        flush(self, encoder, 0..self.size.get());
    }
}

fn flush<A>(
    heap: &Heap<A>,
    encoder: &mut wgpu::CommandEncoder,
    range: Range<BufferAddress>,
) {
    encoder.copy_buffer_to_buffer(
        &heap.staging_buffer,
        range.start,
        &heap.gpu_buffer,
        range.start,
        range
            .end
            .checked_sub(range.start)
            .expect("range is backwards; end should not be less than start"),
    );
}
