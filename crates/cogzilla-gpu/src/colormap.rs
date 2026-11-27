use crate::policy::GpuOp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuColormapMode {
    Bilinear,
    Cubic,
    Lanczos3,
    Nearest,
}

impl GpuColormapMode {
    pub(crate) fn as_u32(self) -> u32 {
        match self {
            Self::Bilinear => 0,
            Self::Cubic => 1,
            Self::Lanczos3 => 2,
            Self::Nearest => 3,
        }
    }

    pub fn to_policy_op(self) -> GpuOp {
        match self {
            Self::Bilinear | Self::Nearest => GpuOp::FusedColormapBilinear,
            Self::Cubic => GpuOp::FusedColormapCubic,
            Self::Lanczos3 => GpuOp::FusedColormapLanczos3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPixelOutput {
    /// 4 bytes/pixel (`R, G, B, A`, transparent `[0,0,0,0]` for nodata).
    Rgba8,
    /// 1 byte/pixel (`0` = nodata, `1..=255` = palette index), 4× smaller PCIe readback.
    IndexedU8,
}

/// Pack a 256-entry RGB colormap LUT into opaque little-endian RGBA8 `u32` words for GPU shaders.
pub fn pack_rgb_lut(rgb: &[[u8; 3]; 256]) -> [u32; 256] {
    let mut packed = [0u32; 256];
    for (i, &[r, g, b]) in rgb.iter().enumerate() {
        packed[i] = (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | (0xffu32 << 24);
    }
    packed
}
