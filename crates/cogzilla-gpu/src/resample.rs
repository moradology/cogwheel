use crate::context::GpuError;

/// GPU-resident `f32` raster slab (e.g. a decoded COG block or stitched tile neighborhood).
/// Upload once, then resample repeatedly across views/tiles/zooms with zero host->device copy.
pub struct GpuSlab {
    pub(crate) buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
}

impl GpuSlab {
    /// Allocate a `STORAGE` buffer with `mapped_at_creation: true` and invoke `fill` directly
    /// on the mapped `&mut [f32]` slice (`height * stride` floats) — eliminating intermediate
    /// CPU `Vec<f32>` allocations and extra copies.
    pub fn from_mapped_fill<F>(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        stride: u32,
        fill: F,
    ) -> Result<Self, GpuError>
    where
        F: FnOnce(&mut [f32]) -> Result<(), GpuError>,
    {
        if width == 0 || height == 0 || stride < width {
            return Err(GpuError::InvalidDimensions(format!(
                "invalid slab: {width}x{height} stride={stride}"
            )));
        }
        let total_elems = (height as usize)
            .checked_mul(stride as usize)
            .ok_or_else(|| GpuError::InvalidDimensions("slab size overflow".into()))?;
        let byte_size = total_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| GpuError::InvalidDimensions("slab byte size overflow".into()))?
            as u64;
        let limits = device.limits();
        let max_bytes = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size as u64);
        if byte_size > max_bytes {
            return Err(GpuError::InvalidDimensions(format!(
                "slab buffer {byte_size} B exceeds adapter limit {max_bytes} B"
            )));
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-slab"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        let res = {
            let mut mapped = buffer.slice(..).get_mapped_range_mut();
            let dst: &mut [f32] = bytemuck::cast_slice_mut(&mut mapped);
            fill(dst)
        };
        buffer.unmap();
        res?;
        Ok(Self {
            buffer,
            width,
            height,
            stride,
        })
    }

    /// Upload an `f32` raster slab to GPU memory for repeated warm resamples.
    pub fn upload(
        device: &wgpu::Device,
        data: &[f32],
        width: u32,
        height: u32,
        stride: u32,
    ) -> Result<Self, GpuError> {
        let min_elems = (height as usize)
            .checked_mul(stride as usize)
            .ok_or_else(|| GpuError::InvalidDimensions("height * stride overflow".into()))?;
        if data.len() < min_elems {
            return Err(GpuError::InvalidDimensions(format!(
                "invalid slab: width={width}, height={height}, stride={stride}, len={}",
                data.len()
            )));
        }
        Self::from_mapped_fill(device, width, height, stride, |dst| {
            dst.copy_from_slice(&data[..min_elems]);
            Ok(())
        })
    }
}
