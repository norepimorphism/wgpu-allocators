use wgpu::BufferAddress;

fn align_addr(addr: BufferAddress, alignment: BufferAddress) -> BufferAddress {
    (addr + alignment - 1) & !(alignment - 1)
}

pub trait Alloc {
    fn alloc(
        &mut self,
        size: BufferAddress,
        alignment: BufferAddress,
    ) -> Option<Range<BufferAddress>>;
}

pub trait Dealloc {
    fn dealloc(&mut self, slice: Range<BufferAddress>) -> Result<(), ()>;
}

impl<A> Heap<A> {
    pub fn new(
        device: &wgpu::Device,
        initial_size: BufferAddress,
        allocator: A,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: initial_size,
            usage: wgpu::BufferUsages::empty(),
            mapped_at_creation: false,
        });

        Self { buffer, allocator }
    }
}

pub struct Heap<A> {
    buffer: wgpu::Buffer,
    allocator: A,
}

impl<A: Alloc> Heap<A> {
    pub fn alloc<'a>(
        &'a mut self,
        size: BufferAddress,
        alignment: BufferAddress,
    ) -> Option<BufferSlice<'a>> {
        self.allocator.alloc(size, alignment).map(|range| self.buffer.slice(range))
    }
}

pub struct Stack {
    pointer: BufferAddress,
}

impl Alloc for Stack {
    fn alloc(
        &mut self,
        size: BufferAddress,
        alignment: BufferAddress,
    ) -> Option<Range<BufferAddress>> {
        let end = self.pointer;

        let padded_size = align_addr(size, alignment);
        self.pointer = self.pointer.checked_sub(padded_size)?;

        Some(self.pointer..end)
    }
}
