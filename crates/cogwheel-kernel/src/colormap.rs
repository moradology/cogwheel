use std::sync::OnceLock;

/// L1-resident 256-entry colormap LUT in both RGBA8 (`[u32; 256]`) and opaque RGB8 (`[[u8; 3]; 256]`).
#[repr(align(64))]
pub struct ColormapLut {
    pub rgba_u32: [u32; 256],
    pub rgb: [[u8; 3]; 256],
}

impl ColormapLut {
    fn from_stops(stops: &[(f32, [u8; 3])]) -> Self {
        let mut rgba_u32 = [0u32; 256];
        let mut rgb = [[0u8; 3]; 256];
        for i in 0..256 {
            let t = (i as f32) / 255.0;
            let mut seg = 0usize;
            while seg + 1 < stops.len() && t > stops[seg + 1].0 {
                seg += 1;
            }
            let c = if seg + 1 >= stops.len() {
                stops[stops.len() - 1].1
            } else {
                let (t0, c0) = stops[seg];
                let (t1, c1) = stops[seg + 1];
                let f = if (t1 - t0).abs() < 1e-6 {
                    0.0
                } else {
                    ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
                };
                [
                    (c0[0] as f32 + (c1[0] as f32 - c0[0] as f32) * f).round() as u8,
                    (c0[1] as f32 + (c1[1] as f32 - c0[1] as f32) * f).round() as u8,
                    (c0[2] as f32 + (c1[2] as f32 - c0[2] as f32) * f).round() as u8,
                ]
            };
            rgb[i] = c;
            rgba_u32[i] = u32::from_le_bytes([c[0], c[1], c[2], 255]);
        }
        Self { rgba_u32, rgb }
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
}

impl ColormapName {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
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
            Self::Grayscale => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| ColormapLut::from_stops(&[(0.0, [0, 0, 0]), (1.0, [255, 255, 255])]))
            }
            Self::Viridis => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| {
                    ColormapLut::from_stops(&[
                        (0.00, [68, 1, 84]),
                        (0.13, [72, 40, 120]),
                        (0.25, [62, 73, 137]),
                        (0.38, [49, 104, 142]),
                        (0.50, [38, 130, 142]),
                        (0.63, [31, 158, 137]),
                        (0.75, [53, 183, 121]),
                        (0.88, [109, 205, 89]),
                        (1.00, [253, 231, 37]),
                    ])
                })
            }
            Self::Magma => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| {
                    ColormapLut::from_stops(&[
                        (0.00, [0, 0, 4]),
                        (0.25, [81, 18, 124]),
                        (0.50, [183, 55, 121]),
                        (0.75, [251, 136, 97]),
                        (1.00, [252, 253, 191]),
                    ])
                })
            }
            Self::Plasma => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| {
                    ColormapLut::from_stops(&[
                        (0.00, [13, 8, 135]),
                        (0.25, [126, 3, 168]),
                        (0.50, [204, 71, 120]),
                        (0.75, [248, 149, 64]),
                        (1.00, [240, 249, 33]),
                    ])
                })
            }
            Self::Inferno => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| {
                    ColormapLut::from_stops(&[
                        (0.00, [0, 0, 4]),
                        (0.25, [87, 16, 110]),
                        (0.50, [188, 55, 84]),
                        (0.75, [249, 142, 9]),
                        (1.00, [252, 255, 164]),
                    ])
                })
            }
            Self::Turbo => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| {
                    ColormapLut::from_stops(&[
                        (0.00, [48, 18, 59]),
                        (0.20, [70, 134, 251]),
                        (0.40, [27, 229, 181]),
                        (0.60, [164, 252, 60]),
                        (0.80, [251, 126, 33]),
                        (1.00, [122, 4, 3]),
                    ])
                })
            }
            Self::Terrain => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| {
                    ColormapLut::from_stops(&[
                        (0.00, [51, 51, 153]),
                        (0.15, [0, 153, 255]),
                        (0.25, [0, 204, 102]),
                        (0.50, [255, 255, 153]),
                        (0.75, [153, 102, 51]),
                        (1.00, [255, 255, 255]),
                    ])
                })
            }
            Self::Coolwarm => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| {
                    ColormapLut::from_stops(&[
                        (0.00, [59, 76, 192]),
                        (0.25, [124, 159, 249]),
                        (0.50, [221, 221, 221]),
                        (0.75, [244, 152, 122]),
                        (1.00, [180, 4, 38]),
                    ])
                })
            }
            Self::Ndvi => {
                static LUT: OnceLock<ColormapLut> = OnceLock::new();
                LUT.get_or_init(|| {
                    ColormapLut::from_stops(&[
                        (0.00, [165, 0, 38]),
                        (0.25, [244, 109, 67]),
                        (0.50, [255, 255, 191]),
                        (0.75, [102, 189, 99]),
                        (1.00, [0, 104, 55]),
                    ])
                })
            }
        }
    }
}
