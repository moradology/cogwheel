//! Shared `wgpu` bind-group-layout, pipeline, mapped-slab, readback, and rescale helpers.

use crate::context::GpuError;

#[inline]
pub(crate) fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[inline]
pub(crate) fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[inline]
pub(crate) fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Safe denominator for `[min, max]` linear rescaling (guards `0`, subnormal, `NaN`, `Inf`).
#[inline]
pub(crate) fn safe_range(min: f32, max: f32) -> f32 {
    let range = max - min;
    if range.is_finite() && range.abs() > 1e-20 {
        range
    } else {
        1.0
    }
}

/// `(255.0 / safe_range, 254.0 / safe_range)` for RGBA8 / IndexedU8 GPU rescale uniforms.
#[inline]
pub(crate) fn safe_inv_ranges(min: f32, max: f32) -> (f32, f32) {
    let r = safe_range(min, max);
    (255.0 / r, 254.0 / r)
}

/// Allocate a `STORAGE` buffer of `total_elems` `f32`s with `mapped_at_creation: true`,
/// enforcing adapter `max_buffer_size` and `max_storage_buffer_binding_size`, and invoke `fill`
/// directly on the mapped slice before unmapping.
pub(crate) fn create_mapped_f32_storage<F>(
    device: &wgpu::Device,
    label: &str,
    total_elems: usize,
    fill: F,
) -> Result<wgpu::Buffer, GpuError>
where
    F: FnOnce(&mut [f32]) -> Result<(), GpuError>,
{
    let byte_size = total_elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| GpuError::InvalidDimensions(format!("{label} byte size overflow")))?
        as u64;
    let limits = device.limits();
    let max_bytes = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size as u64);
    if byte_size > max_bytes {
        return Err(GpuError::InvalidDimensions(format!(
            "{label} {byte_size} B exceeds adapter limit {max_bytes} B"
        )));
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
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
    Ok(buffer)
}

/// Synchronously map `staging_buf[..byte_len]` for reading, invoke `f` on the mapped bytes,
/// and unmap the buffer before returning.
pub(crate) fn readback_staging_sync<R>(
    device: &wgpu::Device,
    staging_buf: &wgpu::Buffer,
    byte_len: u64,
    f: impl FnOnce(&[u8]) -> R,
) -> Result<R, GpuError> {
    let slice = staging_buf.slice(..byte_len);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| GpuError::BufferMap(e.to_string()))?
        .map_err(|e| GpuError::BufferMap(e.to_string()))?;
    let out = {
        let mapped = slice.get_mapped_range();
        f(&mapped)
    };
    staging_buf.unmap();
    Ok(out)
}

/// Unpack 4-byte-padded `IndexedU8` GPU output rows into tightly packed `dst_w * dst_h` bytes
/// (single-copy `to_vec()` when `dst_w % 4 == 0`).
pub(crate) fn unpack_indexed_u8(raw: &[u8], dst_w: usize, dst_h: usize, words_per_row: usize) -> Vec<u8> {
    if dst_w.is_multiple_of(4) {
        raw[..dst_w * dst_h].to_vec()
    } else {
        let row_bytes = words_per_row * 4;
        let mut out = Vec::with_capacity(dst_w * dst_h);
        for py in 0..dst_h {
            let start = py * row_bytes;
            out.extend_from_slice(&raw[start..start + dst_w]);
        }
        out
    }
}
