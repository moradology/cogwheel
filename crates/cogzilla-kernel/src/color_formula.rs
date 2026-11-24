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
            'R' => r = true,
            'G' => g = true,
            'B' => b = true,
            other => {
                return Err(format!(
                    "invalid color_formula channel '{other}' in '{tok}' (expected R, G, B)"
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
        Ok(Self { stages })
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
                    let rf = r as f32;
                    let gf = g as f32;
                    let bf = b as f32;
                    let luma = 0.299 * rf + 0.587 * gf + 0.114 * bf;
                    r = (luma + (rf - luma) * sat).round().clamp(0.0, 255.0) as u8;
                    g = (luma + (gf - luma) * sat).round().clamp(0.0, 255.0) as u8;
                    b = (luma + (bf - luma) * sat).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        (r, g, b)
    }

    pub fn apply_buffer_in_place(&self, buf: &mut [u8], stride: usize) {
        let mut i = 0;
        while i + 3 <= buf.len() {
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
