use bytemuck::{Pod, Zeroable};

use crate::context::GpuError;

/// GPU-resident stacked multi-band `f32` slab (`num_bands` contiguous `height * stride` planes).
pub struct GpuMultiBandSlab {
    pub(crate) buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub band_stride: u32,
    pub num_bands: u32,
}

impl GpuMultiBandSlab {
    /// Allocate a `STORAGE` buffer with `mapped_at_creation: true` and invoke `fill` directly
    /// on the mapped `&mut [f32]` slice (`num_bands * band_stride` floats) — eliminating all
    /// intermediate CPU `Vec<f32>` allocations and extra copies, with no renderer mutex lock.
    pub fn from_mapped_fill<F>(
        device: &wgpu::Device,
        num_bands: usize,
        width: u32,
        height: u32,
        stride: u32,
        fill: F,
    ) -> Result<Self, GpuError>
    where
        F: FnOnce(&mut [f32]) -> Result<(), GpuError>,
    {
        if num_bands == 0 || num_bands > 8 || width == 0 || height == 0 || stride < width {
            return Err(GpuError::InvalidDimensions(format!(
                "invalid multi-band slab: bands={num_bands}, {width}x{height} stride={stride}"
            )));
        }
        let band_stride = (height as usize)
            .checked_mul(stride as usize)
            .ok_or_else(|| GpuError::InvalidDimensions("band_stride overflow".into()))?;
        let total_elems = band_stride
            .checked_mul(num_bands)
            .ok_or_else(|| GpuError::InvalidDimensions("total_elems overflow".into()))?;
        let buffer = crate::wgpu_util::create_mapped_f32_storage(
            device,
            "cogzilla-multiband-slab",
            total_elems,
            fill,
        )?;
        Ok(Self {
            buffer,
            width,
            height,
            stride,
            band_stride: band_stride as u32,
            num_bands: num_bands as u32,
        })
    }

    /// Upload 1..=8 co-registered `f32` bands into a stacked GPU-resident `GpuMultiBandSlab`.
    pub fn upload_bands(
        device: &wgpu::Device,
        bands: &[&[f32]],
        width: u32,
        height: u32,
        stride: u32,
    ) -> Result<Self, GpuError> {
        let band_stride = (height as usize)
            .checked_mul(stride as usize)
            .ok_or_else(|| GpuError::InvalidDimensions("band_stride overflow".into()))?;
        for b in bands {
            if b.len() < band_stride {
                return Err(GpuError::InvalidDimensions(format!(
                    "band slice len {} < {band_stride}",
                    b.len()
                )));
            }
        }
        Self::from_mapped_fill(device, bands.len(), width, height, stride, |dst| {
            for (b, plane_dst) in bands.iter().zip(dst.chunks_exact_mut(band_stride)) {
                plane_dst.copy_from_slice(&b[..band_stride]);
            }
            Ok(())
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuInstr {
    pub opcode: u32,
    pub imm_u32: u32,
    pub imm_f32: f32,
    pub _pad: u32,
}

impl GpuInstr {
    pub const fn push_const(v: f32) -> Self {
        Self { opcode: 0, imm_u32: 0, imm_f32: v, _pad: 0 }
    }
    pub const fn load_band(slot: u32) -> Self {
        Self { opcode: 1, imm_u32: slot, imm_f32: 0.0, _pad: 0 }
    }
    pub const fn op(opcode: u32) -> Self {
        Self { opcode, imm_u32: 0, imm_f32: 0.0, _pad: 0 }
    }
    pub const fn op_f32(opcode: u32, imm_f32: f32) -> Self {
        Self { opcode, imm_u32: 0, imm_f32, _pad: 0 }
    }
}

/// Pre-fused RGB LUT + optional CIE LCH (D65) saturation for GPU `color_formula` execution.
#[derive(Debug, Clone)]
pub struct GpuColorFormula {
    pub rgb_lut: Option<[u32; 256]>,
    pub saturation: Option<f32>,
}

impl GpuColorFormula {
    /// Pack separate `[u8; 256]` R/G/B LUTs (e.g. from Gamma + Sigmoidal) into 256 `u32` words.
    pub fn pack_rgb_luts(r: &[u8; 256], g: &[u8; 256], b: &[u8; 256]) -> [u32; 256] {
        let mut out = [0u32; 256];
        for i in 0..256 {
            out[i] = (r[i] as u32) | ((g[i] as u32) << 8) | ((b[i] as u32) << 16);
        }
        out
    }

    /// Build Gamma + Sigmoidal LUT + optional LCH Saturation directly.
    pub fn from_gamma_sigmoidal_sat(
        gamma: Option<f32>,
        sigmoidal: Option<(f32, f32)>,
        saturation: Option<f32>,
    ) -> Self {
        let rgb_lut = if gamma.is_some() || sigmoidal.is_some() {
            let mut lut = [0u8; 256];
            for (i, slot) in lut.iter_mut().enumerate() {
                let mut v = (i as f32) / 255.0;
                if let Some(g) = gamma {
                    v = v.clamp(0.0, 1.0).powf(1.0 / g);
                }
                if let Some((contrast, bias)) = sigmoidal {
                    let s0 = 1.0 / (1.0 + (contrast * bias).exp());
                    let s1 = 1.0 / (1.0 + (contrast * (bias - 1.0)).exp());
                    let sv = 1.0 / (1.0 + (contrast * (bias - v)).exp());
                    v = ((sv - s0) / (s1 - s0)).clamp(0.0, 1.0);
                }
                *slot = (v * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            Some(Self::pack_rgb_luts(&lut, &lut, &lut))
        } else {
            None
        };
        Self {
            rgb_lut,
            saturation,
        }
    }
}

#[cfg(test)]
mod tests {
    use cogzilla_kernel::expr_vm::Op;

    /// Pin that every `case <N>u:` in `common_expr_vm.wgsl` is in exact 1-to-1 bijection
    /// with `Op::GPU_WIRE_OPCODES` (the single source of truth in `cogzilla-kernel`).
    #[test]
    fn wgsl_switch_matches_kernel_wire_opcodes() {
        let wgsl = include_str!("shaders/common_expr_vm.wgsl");
        let mut wgsl_cases: Vec<u32> = Vec::new();
        for line in wgsl.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("case ")
                && let Some(num_str) = rest.split("u:").next()
                && let Ok(n) = num_str.parse::<u32>()
            {
                wgsl_cases.push(n);
            }
        }
        let expected: Vec<u32> = Op::GPU_WIRE_OPCODES.iter().map(|&(_, id)| id).collect();
        assert_eq!(
            wgsl_cases, expected,
            "common_expr_vm.wgsl switch cases drifted from Op::GPU_WIRE_OPCODES"
        );
    }
}
