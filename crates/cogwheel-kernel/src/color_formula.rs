/// Precompiled `rio-color` / TiTiler `?color_formula=` pipeline
/// (supports `Gamma`, `Sigmoidal`, and `Saturation`, e.g. `"Gamma RGB 1.5, Sigmoidal RGB 15 0.35, Saturation 1.2"`).
/// Gamma & Sigmoidal are fused into 3x 256-byte LUTs (1 table lookup per channel!).
#[derive(Debug, Clone)]
pub struct ColorFormula {
    pub lut_r: [u8; 256],
    pub lut_g: [u8; 256],
    pub lut_b: [u8; 256],
    pub has_lut: bool,
    pub saturation: Option<f32>,
}

impl ColorFormula {
    pub fn parse(formula: &str) -> Result<Self, String> {
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
        let mut saturation = None;

        for raw_op in formula.split(',') {
            let parts: Vec<&str> = raw_op.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            match parts[0].to_ascii_lowercase().as_str() {
                "gamma" => {
                    if parts.len() < 3 {
                        return Err("Gamma requires: Gamma <R|G|B|RGB> <value>".into());
                    }
                    let ch = parts[1].to_ascii_uppercase();
                    let g: f32 = parts[2].parse().map_err(|_| "invalid gamma value")?;
                    let inv_g = 1.0 / g.max(1e-4);
                    has_lut = true;
                    for i in 0..256 {
                        if ch.contains('R') { r_f[i] = r_f[i].clamp(0.0, 1.0).powf(inv_g); }
                        if ch.contains('G') { g_f[i] = g_f[i].clamp(0.0, 1.0).powf(inv_g); }
                        if ch.contains('B') { b_f[i] = b_f[i].clamp(0.0, 1.0).powf(inv_g); }
                    }
                }
                "sigmoidal" => {
                    if parts.len() < 4 {
                        return Err("Sigmoidal requires: Sigmoidal <R|G|B|RGB> <contrast> <bias>".into());
                    }
                    let ch = parts[1].to_ascii_uppercase();
                    let contrast: f32 = parts[2].parse().map_err(|_| "invalid sigmoidal contrast")?;
                    let bias: f32 = parts[3].parse().map_err(|_| "invalid sigmoidal bias")?;
                    let alpha = bias;
                    let beta = contrast;
                    let sig = |x: f32| -> f32 {
                        let lo = 1.0 / (1.0 + (beta * alpha).exp());
                        let hi = 1.0 / (1.0 + (beta * (alpha - 1.0)).exp());
                        let y = 1.0 / (1.0 + (beta * (alpha - x)).exp());
                        ((y - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0)
                    };
                    has_lut = true;
                    for i in 0..256 {
                        if ch.contains('R') { r_f[i] = sig(r_f[i]); }
                        if ch.contains('G') { g_f[i] = sig(g_f[i]); }
                        if ch.contains('B') { b_f[i] = sig(b_f[i]); }
                    }
                }
                "saturation" => {
                    if parts.len() < 2 {
                        return Err("Saturation requires: Saturation <proportion>".into());
                    }
                    let sat: f32 = parts[1].parse().map_err(|_| "invalid saturation value")?;
                    saturation = Some(sat.max(0.0));
                }
                other => return Err(format!("unsupported color_formula op '{other}'")),
            }
        }

        let mut lut_r = [0u8; 256];
        let mut lut_g = [0u8; 256];
        let mut lut_b = [0u8; 256];
        for i in 0..256 {
            lut_r[i] = (r_f[i] * 255.0).round().clamp(0.0, 255.0) as u8;
            lut_g[i] = (g_f[i] * 255.0).round().clamp(0.0, 255.0) as u8;
            lut_b[i] = (b_f[i] * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        Ok(Self {
            lut_r,
            lut_g,
            lut_b,
            has_lut,
            saturation,
        })
    }

    #[inline(always)]
    pub fn apply_rgb(&self, mut r: u8, mut g: u8, mut b: u8) -> (u8, u8, u8) {
        if self.has_lut {
            r = self.lut_r[r as usize];
            g = self.lut_g[g as usize];
            b = self.lut_b[b as usize];
        }
        if let Some(sat) = self.saturation {
            let rf = r as f32;
            let gf = g as f32;
            let bf = b as f32;
            let luma = 0.299 * rf + 0.587 * gf + 0.114 * bf;
            r = (luma + (rf - luma) * sat).clamp(0.0, 255.0) as u8;
            g = (luma + (gf - luma) * sat).clamp(0.0, 255.0) as u8;
            b = (luma + (bf - luma) * sat).clamp(0.0, 255.0) as u8;
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
