use std::collections::HashMap;
use std::sync::OnceLock;

use crate::colormap_data::*;

/// Parse a TiTiler JSON color value: `[r, g, b]` / `[r, g, b, a]` or `"#rrggbb"` / `"#rrggbbaa"`.
fn parse_json_rgba(v: &serde_json::Value) -> Option<[u8; 4]> {
    if let Some(arr) = v.as_array() {
        if arr.len() >= 3 {
            let r = arr[0].as_u64().unwrap_or(0).min(255) as u8;
            let g = arr[1].as_u64().unwrap_or(0).min(255) as u8;
            let b = arr[2].as_u64().unwrap_or(0).min(255) as u8;
            let a = arr.get(3).and_then(|x| x.as_u64()).unwrap_or(255).min(255) as u8;
            return Some([r, g, b, a]);
        }
    } else if let Some(s) = v.as_str() {
        let hex = s.trim().strip_prefix('#').unwrap_or(s.trim());
        if (hex.len() == 6 || hex.len() == 8) && hex.is_ascii() {
            if let (Ok(r), Ok(g), Ok(b)) = (
                u8::from_str_radix(&hex[0..2], 16),
                u8::from_str_radix(&hex[2..4], 16),
                u8::from_str_radix(&hex[4..6], 16),
            ) {
                let a = if hex.len() == 8 {
                    u8::from_str_radix(&hex[6..8], 16).unwrap_or(255)
                } else {
                    255
                };
                return Some([r, g, b, a]);
            }
        }
    }
    None
}

/// Custom TiTiler JSON `?colormap=` supporting both discrete class maps (`{"10": [r,g,b,a], ...}`)
/// with arbitrary signed integer keys (including `< 0` and `>= 256`) and interval colormaps
/// (`[[[min, max], [r,g,b,a]], ...]`).
#[derive(Clone)]
pub enum CustomColormap {
    Discrete {
        lut: Box<ColormapLut>,
        extra: HashMap<i64, [u8; 4]>,
        /// When true (`?rescale=` was NOT explicitly supplied), look up raw integer values directly
        /// instead of rescaling by the dtype range (fixes classified `Uint16`/`Int16`/`Float32` COGs).
        direct_lookup: bool,
    },
    Intervals(Vec<(f32, f32, [u8; 4])>),
}

impl CustomColormap {
    /// Parse a TiTiler custom JSON colormap (object for discrete classes, array for intervals).
    pub fn from_json_str(json_str: &str, rescale_explicit: bool) -> Result<Self, String> {
        let val: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| format!("invalid colormap JSON: {e}"))?;
        if let Some(obj) = val.as_object() {
            let mut rgba_u32 = [0u32; 256];
            let mut rgb = [[0u8; 3]; 256];
            let mut extra = HashMap::new();
            for (k, v) in obj {
                let Ok(key) = k.trim().parse::<i64>() else {
                    continue;
                };
                let Some([r, g, b, a]) = parse_json_rgba(v) else {
                    continue;
                };
                if (0..=255).contains(&key) {
                    let idx = key as usize;
                    rgb[idx] = [r, g, b];
                    rgba_u32[idx] = u32::from_le_bytes([r, g, b, a]);
                } else {
                    extra.insert(key, [r, g, b, a]);
                }
            }
            Ok(Self::Discrete {
                lut: Box::new(ColormapLut { rgba_u32, rgb }),
                extra,
                direct_lookup: !rescale_explicit,
            })
        } else if let Some(arr) = val.as_array() {
            let mut intervals = Vec::with_capacity(arr.len());
            for item in arr {
                let pair = item.as_array().ok_or_else(|| {
                    "interval colormap entries must be [[min, max], color]".to_string()
                })?;
                if pair.len() != 2 {
                    return Err("interval colormap entries must be [[min, max], color]".to_string());
                }
                let bounds = pair[0].as_array().ok_or_else(|| {
                    "interval colormap bounds must be [min, max]".to_string()
                })?;
                if bounds.len() != 2 {
                    return Err("interval colormap bounds must be [min, max]".to_string());
                }
                let lo = bounds[0]
                    .as_f64()
                    .ok_or_else(|| "interval min must be a number".to_string())?
                    as f32;
                let hi = bounds[1]
                    .as_f64()
                    .ok_or_else(|| "interval max must be a number".to_string())?
                    as f32;
                let color = parse_json_rgba(&pair[1]).ok_or_else(|| {
                    "interval color must be [r,g,b(,a)] or #rrggbb(aa)".to_string()
                })?;
                intervals.push((lo, hi, color));
            }
            Ok(Self::Intervals(intervals))
        } else {
            Err(
                "colormap JSON must be an object {idx: color} or interval array [[[min,max], color], ...]"
                    .to_string(),
            )
        }
    }

    /// True when this colormap indexes raw (unrescaled) pixel values directly.
    #[inline(always)]
    pub fn uses_direct_lookup(&self) -> bool {
        match self {
            Self::Discrete { direct_lookup, .. } => *direct_lookup,
            Self::Intervals(_) => true,
        }
    }

    /// Map a finite pixel value `v` to `[r, g, b, a]`.
    #[inline(always)]
    pub fn lookup(&self, v: f32, rescale_min: f32, inv_range: f32) -> [u8; 4] {
        match self {
            Self::Discrete {
                lut,
                extra,
                direct_lookup,
            } => {
                if *direct_lookup {
                    let key = v.round() as i64;
                    if (0..=255).contains(&key) {
                        lut.rgba_u32[key as usize].to_le_bytes()
                    } else {
                        extra.get(&key).copied().unwrap_or([0, 0, 0, 0])
                    }
                } else {
                    let idx = ((v - rescale_min) * inv_range).clamp(0.0, 255.0) as usize;
                    lut.rgba_u32[idx].to_le_bytes()
                }
            }
            Self::Intervals(intervals) => {
                for &(lo, hi, rgba) in intervals {
                    if v >= lo && v < hi {
                        return rgba;
                    }
                }
                [0, 0, 0, 0]
            }
        }
    }
}

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

    /// Parse TiTiler custom JSON colormap `{"0":[r,g,b,a], "128":"#rrggbb", ...}`.
    pub fn from_json_str(json_str: &str) -> Result<Self, String> {
        let val: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| format!("invalid colormap JSON: {e}"))?;
        let obj = val
            .as_object()
            .ok_or_else(|| "colormap JSON must be an object mapping index -> [r,g,b(,a)] or #rrggbb(aa)".to_string())?;
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
            } else if let Some(s) = v.as_str() {
                let hex = s.trim().strip_prefix('#').unwrap_or(s.trim());
                if (hex.len() == 6 || hex.len() == 8) && hex.is_ascii() {
                    if let (Ok(r), Ok(g), Ok(b)) = (
                        u8::from_str_radix(&hex[0..2], 16),
                        u8::from_str_radix(&hex[2..4], 16),
                        u8::from_str_radix(&hex[4..6], 16),
                    ) {
                        let a = if hex.len() == 8 {
                            u8::from_str_radix(&hex[6..8], 16).unwrap_or(255)
                        } else {
                            255
                        };
                        rgb[idx] = [r, g, b];
                        rgba_u32[idx] = u32::from_le_bytes([r, g, b, a]);
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_colormap_discrete_direct_and_extra_keys_work_without_rescaling() {
        let json = r##"{"-1":[1,2,3,255],"10":[0,100,0,255],"80":[0,100,200,255],"300":"#ff880080"}"##;
        // Without explicit `?rescale=`, Uint16 rescale (0, 65535) must NOT crush classes into idx 0.
        let cmap = CustomColormap::from_json_str(json, false).unwrap();
        assert!(cmap.uses_direct_lookup());
        let inv_u16 = 255.0 / 65535.0;
        assert_eq!(cmap.lookup(-1.0, 0.0, inv_u16), [1, 2, 3, 255]);
        assert_eq!(cmap.lookup(10.0, 0.0, inv_u16), [0, 100, 0, 255]);
        assert_eq!(cmap.lookup(80.0, 0.0, inv_u16), [0, 100, 200, 255]);
        assert_eq!(cmap.lookup(300.0, 0.0, inv_u16), [255, 136, 0, 128]);
        assert_eq!(cmap.lookup(99.0, 0.0, inv_u16), [0, 0, 0, 0]);

        // With explicit `?rescale=0,1000`, lookup rescales into 0..=255.
        let rescaled = CustomColormap::from_json_str(r#"{"0":[10,20,30,255],"255":[200,210,220,255]}"#, true).unwrap();
        assert!(!rescaled.uses_direct_lookup());
        assert_eq!(rescaled.lookup(0.0, 0.0, 255.0 / 1000.0), [10, 20, 30, 255]);
        assert_eq!(rescaled.lookup(1000.0, 0.0, 255.0 / 1000.0), [200, 210, 220, 255]);
    }

    #[test]
    fn custom_colormap_intervals_match_half_open_bounds() {
        let json = r##"[[[0,100],[0,0,255,255]],[[100,500],"#00ff00"]]"##;
        let cmap = CustomColormap::from_json_str(json, false).unwrap();
        assert!(cmap.uses_direct_lookup());
        assert_eq!(cmap.lookup(-0.1, 0.0, 1.0), [0, 0, 0, 0]);
        assert_eq!(cmap.lookup(0.0, 0.0, 1.0), [0, 0, 255, 255]);
        assert_eq!(cmap.lookup(99.9, 0.0, 1.0), [0, 0, 255, 255]);
        assert_eq!(cmap.lookup(100.0, 0.0, 1.0), [0, 255, 0, 255]);
        assert_eq!(cmap.lookup(499.9, 0.0, 1.0), [0, 255, 0, 255]);
        assert_eq!(cmap.lookup(500.0, 0.0, 1.0), [0, 0, 0, 0]);
    }
}
