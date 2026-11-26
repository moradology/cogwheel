#[derive(Debug, Clone)]
enum ColorStage {
    Lut {
        r: [u8; 256],
        g: [u8; 256],
        b: [u8; 256],
    },
    Saturation(f32),
}

/// Precompiled `rio-color` / TiTiler `?color_formula=` pipeline
/// (supports `Gamma`, `Sigmoidal`, and `Saturation`, with either space-separated or
/// comma-separated ops, e.g. `"Gamma RGB 1.5 Sigmoidal RGB 15 0.35 Saturation 1.2"`).
/// Consecutive LUT ops (Gamma & Sigmoidal) are fused into 3x 256-byte LUTs while preserving
/// exact left-to-right ordering around `Saturation` stages.
#[derive(Debug, Clone)]
pub struct ColorFormula {
    stages: Vec<ColorStage>,
}

fn parse_channels(tok: &str) -> Result<(bool, bool, bool), String> {
    if tok.is_empty() {
        return Err("empty channel specifier (expected R, G, B, or combination like RGB)".into());
    }
    let (mut r, mut g, mut b) = (false, false, false);
    for ch in tok.chars() {
        match ch.to_ascii_uppercase() {
            'R' | '1' => r = true,
            'G' | '2' => g = true,
            'B' | '3' => b = true,
            other => {
                return Err(format!(
                    "invalid color_formula channel '{other}' in '{tok}' (expected R, G, B, or 1, 2, 3)"
                ));
            }
        }
    }
    Ok((r, g, b))
}

fn flush_lut(
    stages: &mut Vec<ColorStage>,
    r_f: &mut [f32; 256],
    g_f: &mut [f32; 256],
    b_f: &mut [f32; 256],
    has_lut: &mut bool,
) {
    if !*has_lut {
        return;
    }
    let mut r = [0u8; 256];
    let mut g = [0u8; 256];
    let mut b = [0u8; 256];
    for i in 0..256 {
        r[i] = (r_f[i] * 255.0).round().clamp(0.0, 255.0) as u8;
        g[i] = (g_f[i] * 255.0).round().clamp(0.0, 255.0) as u8;
        b[i] = (b_f[i] * 255.0).round().clamp(0.0, 255.0) as u8;
        let v = (i as f32) / 255.0;
        r_f[i] = v;
        g_f[i] = v;
        b_f[i] = v;
    }
    stages.push(ColorStage::Lut { r, g, b });
    *has_lut = false;
}

impl ColorFormula {
    pub fn parse(formula: &str) -> Result<Self, String> {
        const MAX_FORMULA_CHARS: usize = 2048;
        const MAX_FORMULA_STAGES: usize = 32;
        if formula.len() > MAX_FORMULA_CHARS {
            return Err(format!(
                "color_formula length ({}) exceeds limit ({MAX_FORMULA_CHARS})",
                formula.len()
            ));
        }
        let tokens: Vec<&str> = formula
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|t| !t.is_empty())
            .collect();

        let mut stages = Vec::new();
        let mut r_f = [0.0f32; 256];
        let mut g_f = [0.0f32; 256];
        let mut b_f = [0.0f32; 256];
        for i in 0..256 {
            let v = (i as f32) / 255.0;
            r_f[i] = v;
            g_f[i] = v;
            b_f[i] = v;
        }
        let mut has_lut = false;

        let mut idx = 0usize;
        while idx < tokens.len() {
            match tokens[idx].to_ascii_lowercase().as_str() {
                "gamma" => {
                    if idx + 2 >= tokens.len() {
                        return Err("Gamma requires: Gamma <R|G|B|RGB> <value>".into());
                    }
                    let (cr, cg, cb) = parse_channels(tokens[idx + 1])?;
                    let g: f32 = tokens[idx + 2]
                        .parse()
                        .map_err(|_| "invalid gamma value")?;
                    if !g.is_finite() || g <= 0.0 {
                        return Err("gamma value must be a finite number > 0".into());
                    }
                    let inv_g = 1.0 / g;
                    has_lut = true;
                    for i in 0..256 {
                        if cr { r_f[i] = r_f[i].clamp(0.0, 1.0).powf(inv_g); }
                        if cg { g_f[i] = g_f[i].clamp(0.0, 1.0).powf(inv_g); }
                        if cb { b_f[i] = b_f[i].clamp(0.0, 1.0).powf(inv_g); }
                    }
                    idx += 3;
                }
                "sigmoidal" => {
                    if idx + 3 >= tokens.len() {
                        return Err(
                            "Sigmoidal requires: Sigmoidal <R|G|B|RGB> <contrast> <bias>".into(),
                        );
                    }
                    let (cr, cg, cb) = parse_channels(tokens[idx + 1])?;
                    let contrast: f32 = tokens[idx + 2]
                        .parse()
                        .map_err(|_| "invalid sigmoidal contrast")?;
                    let bias: f32 = tokens[idx + 3]
                        .parse()
                        .map_err(|_| "invalid sigmoidal bias")?;
                    if !contrast.is_finite() || !bias.is_finite() {
                        return Err("sigmoidal contrast and bias must be finite".into());
                    }
                    idx += 4;
                    if contrast == 0.0 {
                        continue;
                    }
                    // Rio-color sigmoidal: positive contrast increases contrast; negative contrast
                    // decreases contrast (inverse sigmoidal curve).
                    let alpha = bias;
                    let beta = contrast.abs();
                    let lo = 1.0 / (1.0 + (beta * alpha).exp());
                    let hi = 1.0 / (1.0 + (beta * (alpha - 1.0)).exp());
                    let span = (hi - lo).max(1e-6);
                    let sig = |x: f32| -> f32 {
                        let x = x.clamp(0.0, 1.0);
                        if contrast > 0.0 {
                            let y = 1.0 / (1.0 + (beta * (alpha - x)).exp());
                            ((y - lo) / span).clamp(0.0, 1.0)
                        } else {
                            let y = (lo + x * span).clamp(1e-6, 1.0 - 1e-6);
                            (alpha - ((1.0 / y) - 1.0).ln() / beta).clamp(0.0, 1.0)
                        }
                    };
                    has_lut = true;
                    for i in 0..256 {
                        if cr { r_f[i] = sig(r_f[i]); }
                        if cg { g_f[i] = sig(g_f[i]); }
                        if cb { b_f[i] = sig(b_f[i]); }
                    }
                }
                "saturation" => {
                    if idx + 1 >= tokens.len() {
                        return Err("Saturation requires: Saturation <proportion>".into());
                    }
                    let sat: f32 = tokens[idx + 1]
                        .parse()
                        .map_err(|_| "invalid saturation value")?;
                    if !sat.is_finite() || sat < 0.0 {
                        return Err("saturation value must be finite >= 0".into());
                    }
                    flush_lut(&mut stages, &mut r_f, &mut g_f, &mut b_f, &mut has_lut);
                    stages.push(ColorStage::Saturation(sat));
                    idx += 2;
                }
                other => return Err(format!("unsupported color_formula op '{other}'")),
            }
        }
        flush_lut(&mut stages, &mut r_f, &mut g_f, &mut b_f, &mut has_lut);
        if stages.len() > MAX_FORMULA_STAGES {
            return Err(format!(
                "color_formula has {} stages (max {MAX_FORMULA_STAGES})",
                stages.len()
            ));
        }
        Ok(Self { stages })
    }

    /// Extract a single-pass GPU-compatible `(optional RGB LUT, optional LCH saturation)` when the
    /// formula is representable as an optional fused LUT stage followed by an optional saturation
    /// stage (e.g. `"Gamma RGB 1.5 Sigmoidal RGB 15 0.35 Saturation 1.2"`). Returns `None` for
    /// multi-stage formulas with LUT ops *after* a Saturation stage.
    #[allow(clippy::type_complexity)]
    pub fn as_lut_then_saturation(
        &self,
    ) -> Option<(Option<(&[u8; 256], &[u8; 256], &[u8; 256])>, Option<f32>)> {
        match self.stages.as_slice() {
            [] => Some((None, None)),
            [ColorStage::Lut { r, g, b }] => Some((Some((r, g, b)), None)),
            [ColorStage::Saturation(s)] => Some((None, Some(*s))),
            [ColorStage::Lut { r, g, b }, ColorStage::Saturation(s)] => {
                Some((Some((r, g, b)), Some(*s)))
            }
            _ => None,
        }
    }

    #[inline(always)]
    pub fn apply_rgb(&self, mut r: u8, mut g: u8, mut b: u8) -> (u8, u8, u8) {
        for stage in &self.stages {
            match stage {
                ColorStage::Lut {
                    r: lr,
                    g: lg,
                    b: lb,
                } => {
                    r = lr[r as usize];
                    g = lg[g as usize];
                    b = lb[b as usize];
                }
                ColorStage::Saturation(sat) => {
                    // Match `rio-color` LCH saturation: sRGB -> linear RGB -> CIE XYZ (D65) ->
                    // CIE Lab -> scale chroma `(a, b) *= sat` (exact equivalent of `C *= sat` at
                    // constant hue `H`) -> CIE XYZ -> linear RGB -> sRGB.
                    let (nr, ng, nb) = apply_lch_saturation(r, g, b, *sat);
                    r = nr;
                    g = ng;
                    b = nb;
                }
            }
        }
        (r, g, b)
    }

    pub fn apply_buffer_in_place(&self, buf: &mut [u8], stride: usize) {
        if stride < 3 {
            return;
        }
        // Fast hoisted single-LUT path (the common Gamma / Sigmoidal case with no Saturation).
        if let [ColorStage::Lut { r: lr, g: lg, b: lb }] = self.stages.as_slice() {
            let mut i = 0;
            while i + stride <= buf.len() {
                if stride == 4 && buf[i + 3] == 0 {
                    i += 4;
                    continue;
                }
                buf[i] = lr[buf[i] as usize];
                buf[i + 1] = lg[buf[i + 1] as usize];
                buf[i + 2] = lb[buf[i + 2] as usize];
                i += stride;
            }
            return;
        }
        let mut i = 0;
        while i + stride <= buf.len() {
            if stride == 4 && buf[i + 3] == 0 {
                i += 4;
                continue;
            }
            let (r, g, b) = self.apply_rgb(buf[i], buf[i + 1], buf[i + 2]);
            buf[i] = r;
            buf[i + 1] = g;
            buf[i + 2] = b;
            i += stride;
        }
    }
}

/// Bit-exact 256-entry sRGB u8 -> linear f32 LUT (avoids 3x `powf(2.4)` per pixel in LCH Saturation).
#[inline(always)]
fn srgb_to_lin_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let u = (i as f32) * (1.0 / 255.0);
            *slot = if u <= 0.04045 {
                u * (1.0 / 12.92)
            } else {
                ((u + 0.055) * (1.0 / 1.055)).powf(2.4)
            };
        }
        t
    })
}

/// CIE LCH (D65) saturation matching `rio-color`'s `Saturation` operation.
/// Scaling chroma `C *= sat` at constant hue `H` is mathematically identical to scaling
/// CIE Lab `(a, b)` by `sat` directly without trigonometric round-trips.
#[inline(always)]
fn apply_lch_saturation(r: u8, g: u8, b: u8, sat: f32) -> (u8, u8, u8) {
    let lin_lut = srgb_to_lin_lut();
    let srgb_to_lin = |c: u8| -> f32 { lin_lut[c as usize] };
    #[inline(always)]
    fn lin_to_srgb(u: f32) -> u8 {
        let c = u.clamp(0.0, 1.0);
        let s = if c <= 0.0031308 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round().clamp(0.0, 255.0) as u8
    }
    #[inline(always)]
    fn lab_f(t: f32) -> f32 {
        if t > 0.0088564516 {
            t.cbrt()
        } else {
            7.787037 * t + (16.0 / 116.0)
        }
    }
    #[inline(always)]
    fn lab_f_inv(t: f32) -> f32 {
        if t > 0.20689655 {
            t * t * t
        } else {
            (t - (16.0 / 116.0)) * (1.0 / 7.787037)
        }
    }

    let rl = srgb_to_lin(r);
    let gl = srgb_to_lin(g);
    let bl = srgb_to_lin(b);

    // sRGB -> CIE XYZ (D65)
    let x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl;

    // XYZ -> Lab (D65 whitepoint: 0.95047, 1.0, 1.08883)
    let fx = lab_f(x * (1.0 / 0.95047));
    let fy = lab_f(y);
    let fz = lab_f(z * (1.0 / 1.08883));
    let a = 500.0 * (fx - fy) * sat;
    let b_lab = 200.0 * (fy - fz) * sat;

    // Lab -> XYZ
    let fx2 = fy + a * (1.0 / 500.0);
    let fz2 = fy - b_lab * (1.0 / 200.0);
    let x2 = 0.95047 * lab_f_inv(fx2);
    let y2 = lab_f_inv(fy);
    let z2 = 1.08883 * lab_f_inv(fz2);

    // XYZ -> linear sRGB
    let rl2 = 3.2404542 * x2 - 1.5371385 * y2 - 0.4985314 * z2;
    let gl2 = -0.9692660 * x2 + 1.8760108 * y2 + 0.0415560 * z2;
    let bl2 = 0.0556434 * x2 - 0.2040259 * y2 + 1.0572252 * z2;

    (lin_to_srgb(rl2), lin_to_srgb(gl2), lin_to_srgb(bl2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_separated_and_ordered_saturation_and_channel_validation() {
        // Space-separated rio-color formula must match comma-separated
        let f_space = ColorFormula::parse("Gamma RGB 2.0 Saturation 0.0 Sigmoidal RGB 5 0.5").unwrap();
        let f_comma = ColorFormula::parse("Gamma RGB 2.0, Saturation 0.0, Sigmoidal RGB 5 0.5").unwrap();
        assert_eq!(f_space.apply_rgb(200, 100, 50), f_comma.apply_rgb(200, 100, 50));

        // Ordering: Saturation 0.0 -> Gamma R 2.0 must produce R != G (unlike Gamma first then Saturation 0.0)
        let sat_then_gamma = ColorFormula::parse("Saturation 0.0 Gamma R 2.0").unwrap();
        let (r1, g1, b1) = sat_then_gamma.apply_rgb(200, 100, 50);
        assert!(r1 > g1 && g1 == b1, "expected red boosted after desaturate, got ({r1},{g1},{b1})");

        // Negative sigmoidal contrast is valid (reduces contrast)
        assert!(ColorFormula::parse("Sigmoidal RGB -5 0.5").is_ok());

        // Unknown channel letters must be rejected
        assert!(ColorFormula::parse("Gamma XYZ 1.5").is_err());
    }
}
