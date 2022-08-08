//! High-level allocators for WGPU.

#![feature(unchecked_math)]

mod allocators;
pub mod arena;

use wgpu::{BufferAddress, BufferUsages};

use std::ops::Range;

pub use allocators::*;
pub use arena::HeapArena;

pub type NonZeroBufferAddress = std::num::NonZeroU64;

pub trait Allocator {
    fn new(heap: &Heap) -> Self where Self: Sized;

    fn alloc(
        &mut self,
        size: NonZeroBufferAddress,
        alignment: NonZeroBufferAddress,
    ) -> Option<Range<BufferAddress>>;

    /// # Safety
    ///
    /// `range` must be a valid allocation previously returned by this allocator.
    unsafe fn dealloc(&mut self, range: Range<BufferAddress>) -> Result<(), ()>;
}

bitflags::bitflags! {
    pub struct HeapUsages: u32 {
        /// Allows a heap buffer to be the index buffer in a draw operation.
        const INDEX = BufferUsages::INDEX.bits();
        /// Allows a heap buffer to be the vertex buffer in a draw operation.
        const VERTEX = BufferUsages::VERTEX.bits();
        /// Allows a heap buffer to be a [`wgpu::BufferBindingType::Uniform`] inside a bind group.
        const UNIFORM = BufferUsages::UNIFORM.bits();
        /// Allows a heap buffer to be a [`wgpu::BufferBindingType::Storage`] inside a bind group.
        const STORAGE = BufferUsages::STORAGE.bits();
        /// Allows a heap buffer to be the indirect buffer in an indirect draw call.
        const INDIRECT = BufferUsages::INDIRECT.bits();
    }
}

impl HeapUsages {
    fn as_buffer_usages(self) -> BufferUsages {
        // SAFETY: TODO
        unsafe { BufferUsages::from_bits_unchecked(self.bits()) }
    }
}

impl Heap {
    pub fn new(
        device: &wgpu::Device,
        size: NonZeroBufferAddress,
        usage: HeapUsages,
    ) -> Self {
        Heap {
            staging_buffer: create_buffer(
                device,
                size.get(),
                BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
                true,
            ),
            gpu_buffer: create_buffer(
                device,
                size.get(),
                BufferUsages::COPY_DST | usage.as_buffer_usages(),
                false,
            ),
            size,
        }
    }
}

fn create_buffer(
    device: &wgpu::Device,
    size: u64,
    usage: BufferUsages,
    is_mapped_at_creation: bool,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage,
        mapped_at_creation: is_mapped_at_creation,
    })
}

#[derive(Debug)]
pub struct Heap {
    staging_buffer: wgpu::Buffer,
    gpu_buffer: wgpu::Buffer,
    size: NonZeroBufferAddress,
}

impl Heap {
    /// The size, in bytes, of this heap.
    pub fn size(&self) -> NonZeroBufferAddress {
        self.size
    }

    pub fn write_and_flush(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        range: Range<BufferAddress>,
        contents: &[u8],
    ) {
        self.write(range.clone(), contents);
        self.flush_range(encoder, range);
    }

    pub fn write(
        &self,
        range: Range<BufferAddress>,
        contents: &[u8],
    ) {
        let slice = self.staging_buffer.slice(range.clone());
        slice.get_mapped_range_mut().copy_from_slice(contents);
    }

    pub fn slice<'a>(&'a self, range: Range<BufferAddress>) -> wgpu::BufferSlice<'a> {
        self.gpu_buffer.slice(range)
    }

    pub fn binding<'a>(&'a self, range: Range<BufferAddress>) -> wgpu::BufferBinding<'a> {
        wgpu::BufferBinding {
            buffer: &self.gpu_buffer,
            offset: range.start,
            size: Some(
                NonZeroBufferAddress::new(get_range_size(&range))
                    .expect("buffer binding size is zero; must be nonzero")
            ),
        }
    }

    pub fn flush(&self, encoder: &mut wgpu::CommandEncoder) {
        self.flush_range(encoder, 0..self.size.get());
    }

    pub fn flush_range(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        range: Range<BufferAddress>,
    ) {
        encoder.copy_buffer_to_buffer(
            &self.staging_buffer,
            range.start,
            &self.gpu_buffer,
            range.start,
            get_range_size(&range),
        );
    }

    pub fn unmap(&self) {
        self.staging_buffer.unmap();
    }

    pub fn destroy(&self) {
        self.staging_buffer.destroy();
        self.gpu_buffer.destroy();
    }
}

fn get_range_size(range: &Range<BufferAddress>) -> BufferAddress {
    range
        .end
        .checked_sub(range.start)
        .expect("range is backwards; end should not be less than start")
}
