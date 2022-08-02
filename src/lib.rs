//! High-level allocators for WGPU.

mod allocators;

use wgpu::{BufferAddress, BufferSlice};

pub use allocators::*;

pub type NonZeroBufferAddress = std::num::NonZeroU64;

pub struct Allocation<'a> {
    slice: BufferSlice<'a>,
    range: Range<BufferAddress>,
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
    pub fn new(
        device: &wgpu::Device,
        size: NonZeroBufferAddress,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size.get(),
            usage: wgpu::BufferUsages::empty(),
            mapped_at_creation: false,
        });
        let allocator = A::new(size);

        Self { buffer, allocator }
    }
}

pub struct Heap<A> {
    buffer: wgpu::Buffer,
    allocator: A,
}

impl<A: Allocator> Heap<A> {
    pub fn alloc<'a>(
        &'a mut self,
        size: NonZeroBufferAddress,
        alignment: NonZeroBufferAddress,
    ) -> Option<Allocation<'a>> {
        self.allocator.alloc(size, alignment).map(|range| {
            Allocation {
                slice: self.buffer.slice(range),
                range: range,
            }
        })
    }
}

impl<A: Deallocator> Heap<A> {
    pub fn dealloc<'a>(
        &'a mut self,
        allocation: Allocation,
    ) -> Result<(), ()> {
        self.allocator.dealloc(allocation.range)
    }
}
