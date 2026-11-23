use std::sync::OnceLock;

use crate::colormap_data::*;

/// L1-resident 256-entry colormap LUT in both RGBA8 (`[u32; 256]`) and opaque RGB8 (`[[u8; 3]; 256]`).
#[derive(Clone)]
#[repr(align(64))]
pub struct ColormapLut {
    pub rgba_u32: [u32; 256],
    pub rgb: [[u8; 3]; 256],
}

impl ColormapLut {
    pub fn from_rgb_table(rgb: [[u8; 3]; 256]) -> Self {
        let mut rgba_u32 = [0u32; 256];
        for i in 0..256 {
            let c = rgb[i];
            rgba_u32[i] = u32::from_le_bytes([c[0], c[1], c[2], 255]);
        }
        Self { rgba_u32, rgb }
    }

    /// Parse TiTiler custom JSON colormap `{"0":[r,g,b,a], "128":[r,g,b,a], ...}` (discrete or interpolated).
    pub fn from_json_str(json_str: &str) -> Result<Self, String> {
        let val: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| format!("invalid colormap JSON: {e}"))?;
        let obj = val
            .as_object()
            .ok_or_else(|| "colormap JSON must be an object mapping index -> [r,g,b(,a)]".to_string())?;
        let mut rgba_u32 = [0u32; 256];
        let mut rgb = [[0u8; 3]; 256];
        for (k, v) in obj {
            let Ok(idx) = k.parse::<usize>() else {
                continue;
            };
            if idx >= 256 {
                continue;
            }
            if let Some(arr) = v.as_array() {
                if arr.len() >= 3 {
                    let r = arr[0].as_u64().unwrap_or(0).min(255) as u8;
                    let g = arr[1].as_u64().unwrap_or(0).min(255) as u8;
                    let b = arr[2].as_u64().unwrap_or(0).min(255) as u8;
                    let a = arr.get(3).and_then(|x| x.as_u64()).unwrap_or(255).min(255) as u8;
                    rgb[idx] = [r, g, b];
                    rgba_u32[idx] = u32::from_le_bytes([r, g, b, a]);
                }
            }
        }
        Ok(Self { rgba_u32, rgb })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColormapName {
    Grayscale,
    Viridis,
    Magma,
    Plasma,
    Inferno,
    Turbo,
    Terrain,
    Coolwarm,
    Ndvi,
    /// Index into the full 211-entry `rio-tiler` colormap catalog (`COLORMAP_CATALOG_NAMES`).
    Catalog(u16),
}

impl ColormapName {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        let lower = s.to_ascii_lowercase();
        if let Some(idx) = crate::colormap_catalog::lookup_catalog_index(&lower) {
            return Some(Self::Catalog(idx));
        }
        match lower.as_str() {
            "gray" | "grey" | "grayscale" => Some(Self::Grayscale),
            "viridis" => Some(Self::Viridis),
            "magma" => Some(Self::Magma),
            "plasma" => Some(Self::Plasma),
            "inferno" => Some(Self::Inferno),
            "turbo" => Some(Self::Turbo),
            "terrain" => Some(Self::Terrain),
            "coolwarm" => Some(Self::Coolwarm),
            "ndvi" | "rdylgn" => Some(Self::Ndvi),
            _ => None,
        }
    }

    pub fn lut(self) -> &'static ColormapLut {
        match self {
            Self::Catalog(idx) => crate::colormap_catalog::catalog_lut(idx),
            Self::Grayscale => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| ColormapLut::from_rgb_table(GRAYSCALE_RGB))
            }
            Self::Viridis => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| ColormapLut::from_rgb_table(VIRIDIS_RGB))
            }
            Self::Magma => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| ColormapLut::from_rgb_table(MAGMA_RGB))
            }
            Self::Plasma => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| ColormapLut::from_rgb_table(PLASMA_RGB))
            }
            Self::Inferno => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| ColormapLut::from_rgb_table(INFERNO_RGB))
            }
            Self::Turbo => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| ColormapLut::from_rgb_table(TURBO_RGB))
            }
            Self::Terrain => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| ColormapLut::from_rgb_table(TERRAIN_RGB))
            }
            Self::Coolwarm => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| ColormapLut::from_rgb_table(COOLWARM_RGB))
            }
            Self::Ndvi => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| ColormapLut::from_rgb_table(NDVI_RGB))
            }
        }
    }
}
