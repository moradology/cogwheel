#[derive(Debug, Clone)]
enum ColorStage {
    Lut {
        r: [u8; 256],
        g: [u8; 256],
        b: [u8; 256],
    },
    Saturation(f64),
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

/// rio-tiler's step for one operation: the byte as a fraction of 255 in double precision
/// (`to_math_type`), the operation, and back to a byte by truncation (`scale_dtype`).
fn op_lut(op: impl Fn(f64) -> f64) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, out) in lut.iter_mut().enumerate() {
        *out = (op(i as f64 / 255.0) * 255.0) as u8;
    }
    lut
}

/// Bytes through the LUT stages pending since the last `Saturation`, one table a channel.
struct Pending {
    luts: [[u8; 256]; 3],
    any: bool,
}

impl Pending {
    fn new() -> Self {
        let mut identity = [0u8; 256];
        for (i, v) in identity.iter_mut().enumerate() {
            *v = i as u8;
        }
        Self { luts: [identity; 3], any: false }
    }

    /// One more operation on the chosen channels: its table after the ones before, which is
    /// rio-tiler's byte after byte.
    fn then(&mut self, channels: (bool, bool, bool), op: &[u8; 256]) {
        for (lut, chosen) in self.luts.iter_mut().zip([channels.0, channels.1, channels.2]) {
            if chosen {
                for v in lut.iter_mut() {
                    *v = op[usize::from(*v)];
                }
            }
        }
        self.any = true;
    }

    fn flush(&mut self, stages: &mut Vec<ColorStage>) {
        if self.any {
            let [r, g, b] = self.luts;
            stages.push(ColorStage::Lut { r, g, b });
        }
        *self = Self::new();
    }
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
        let mut pending = Pending::new();

        let mut idx = 0usize;
        while idx < tokens.len() {
            match tokens[idx].to_ascii_lowercase().as_str() {
                "gamma" => {
                    if idx + 2 >= tokens.len() {
                        return Err("Gamma requires: Gamma <R|G|B|RGB> <value>".into());
                    }
                    let channels = parse_channels(tokens[idx + 1])?;
                    let g: f64 = tokens[idx + 2]
                        .parse()
                        .map_err(|_| "invalid gamma value")?;
                    if !g.is_finite() || g <= 0.0 {
                        return Err("gamma value must be a finite number > 0".into());
                    }
                    // color-operations' `gamma`: `arr ** (1.0 / g)`.
                    let inv_g = 1.0 / g;
                    pending.then(channels, &op_lut(|x| x.powf(inv_g)));
                    idx += 3;
                }
                "sigmoidal" => {
                    if idx + 3 >= tokens.len() {
                        return Err(
                            "Sigmoidal requires: Sigmoidal <R|G|B|RGB> <contrast> <bias>".into(),
                        );
                    }
                    let channels = parse_channels(tokens[idx + 1])?;
                    let contrast: f64 = tokens[idx + 2]
                        .parse()
                        .map_err(|_| "invalid sigmoidal contrast")?;
                    let bias: f64 = tokens[idx + 3]
                        .parse()
                        .map_err(|_| "invalid sigmoidal bias")?;
                    if !contrast.is_finite() || !bias.is_finite() {
                        return Err("sigmoidal contrast and bias must be finite".into());
                    }
                    if !(0.0..=1.0).contains(&bias) {
                        return Err("sigmoidal bias must be in [0, 1]".into());
                    }
                    idx += 4;
                    if contrast.abs() < 1e-12 {
                        continue;
                    }
                    // color-operations' `sigmoidal`, term for term: positive contrast raises
                    // contrast, negative lowers it (the inverse curve).
                    let (alpha, beta) = (if bias == 0.0 { f64::EPSILON } else { bias }, contrast);
                    let sig = |x: f64| -> f64 {
                        if beta > 0.0 {
                            let numerator = 1.0 / (1.0 + (beta * (alpha - x)).exp()) - 1.0 / (1.0 + (beta * alpha).exp());
                            let denominator = 1.0 / (1.0 + (beta * (alpha - 1.0)).exp()) - 1.0 / (1.0 + (beta * alpha).exp());
                            numerator / denominator
                        } else {
                            let inner = (x / (1.0 + (beta * alpha - beta).exp())) - (x / (1.0 + (beta * alpha).exp()))
                                + (1.0 / (1.0 + (beta * alpha).exp()));
                            ((beta * alpha) - ((1.0 / inner) - 1.0).ln()) / beta
                        }
                    };
                    pending.then(channels, &op_lut(sig));
                }
                "saturation" => {
                    if idx + 1 >= tokens.len() {
                        return Err("Saturation requires: Saturation <proportion>".into());
                    }
                    let sat: f64 = tokens[idx + 1]
                        .parse()
                        .map_err(|_| "invalid saturation value")?;
                    if !sat.is_finite() || sat < 0.0 {
                        return Err("saturation value must be finite >= 0".into());
                    }
                    pending.flush(&mut stages);
                    stages.push(ColorStage::Saturation(sat));
                    idx += 2;
                }
                other => return Err(format!("unsupported color_formula op '{other}'")),
            }
        }
        pending.flush(&mut stages);
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
            [ColorStage::Saturation(s)] => Some((None, Some(*s as f32))),
            [ColorStage::Lut { r, g, b }, ColorStage::Saturation(s)] => {
                Some((Some((r, g, b)), Some(*s as f32)))
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

/// sRGB byte -> linear, in double precision: color-operations' `_rgb_to_xyz` companding of
/// `i / 255`, once for each of the 256 bytes.
#[inline(always)]
fn srgb_to_lin_lut() -> &'static [f64; 256] {
    static LUT: std::sync::OnceLock<[f64; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f64; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let u = i as f64 / 255.0;
            *slot = if u <= 0.04045 { u / 12.92 } else { ((u + 0.055) / 1.055).powf(2.4) };
        }
        t
    })
}

/// The byte rio-tiler makes of a linear value: sRGB companding, held to 0..1, times 255,
/// truncated. color-operations' `_xyz_to_rgb` and `scale_dtype`, term for term.
#[inline(always)]
fn lin_to_byte_exact(c: f64) -> u8 {
    let s = if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s.clamp(0.0, 1.0) * 255.0) as u8
}

/// For each byte `k`, the linear value from which [`lin_to_byte_exact`] gives `k` or more:
/// the companding undone at `k / 255`. A value is placed by a binary search over these
/// instead of a power per channel.
fn byte_thresholds() -> &'static [f64; 256] {
    static T: std::sync::OnceLock<[f64; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0.0f64; 256];
        for (k, slot) in t.iter_mut().enumerate().skip(1) {
            let s = k as f64 / 255.0;
            *slot = if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) };
        }
        t
    })
}

/// [`lin_to_byte_exact`] without its power: the thresholds place the value, and only a
/// value within a hair of one, where the two could part, is worked out in full.
#[inline(always)]
fn lin_to_byte(c: f64) -> u8 {
    let t = byte_thresholds();
    let k = t[1..].partition_point(|edge| *edge <= c);
    let near = |edge: f64| (c - edge).abs() <= edge.abs() * 1e-9 + 1e-15;
    if near(t[k]) || (k < 255 && near(t[k + 1])) {
        return lin_to_byte_exact(c);
    }
    k as u8
}

/// color-operations' `Saturation`: sRGB -> CIE XYZ (D65) -> Lab -> LCH, chroma times `sat`,
/// and back, in double precision; then rio-tiler's truncation to a byte (`scale_dtype`).
/// Scaling chroma at constant hue is scaling Lab's `(a, b)`, which is what is done here.
#[inline(always)]
fn apply_lch_saturation(r: u8, g: u8, b: u8, sat: f64) -> (u8, u8, u8) {
    const DELTA: f64 = 6.0 / 29.0;
    const T0: f64 = DELTA * DELTA * DELTA;
    const BINTERCEPT: f64 = 4.0 / 29.0;
    const XN: f64 = 0.95047;
    const ZN: f64 = 1.08883;
    let alpha: f64 = DELTA.powi(-2) / 3.0;
    let lin_lut = srgb_to_lin_lut();
    let (rl, gl, bl) = (lin_lut[usize::from(r)], lin_lut[usize::from(g)], lin_lut[usize::from(b)]);
    // `cbrt` and a product where color-operations takes powers: a bit or two apart at
    // most, which moves a truncated byte about once in 10^13 values.
    let lab_f = |t: f64| if t > T0 { t.cbrt() } else { alpha * t + BINTERCEPT };
    let lab_f_inv = |t: f64| if t > DELTA { t * t * t } else { 3.0 * DELTA * DELTA * (t - BINTERCEPT) };

    // sRGB -> CIE XYZ, relative to the white point
    let x = ((rl * 0.4124564) + (gl * 0.3575761) + (bl * 0.1804375)) / XN;
    let y = (rl * 0.2126729) + (gl * 0.7151522) + (bl * 0.0721750);
    let z = ((rl * 0.0193339) + (gl * 0.1191920) + (bl * 0.9503041)) / ZN;
    // XYZ -> Lab, chroma scaled
    let (fx, fy, fz) = (lab_f(x), lab_f(y), lab_f(z));
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy) * sat;
    let b_lab = 200.0 * (fy - fz) * sat;
    // Lab -> XYZ
    let ty = (l + 16.0) / 116.0;
    let x2 = lab_f_inv(ty + a / 500.0) * XN;
    let y2 = lab_f_inv(ty);
    let z2 = lab_f_inv(ty - b_lab / 200.0) * ZN;
    // XYZ -> linear sRGB -> sRGB
    let rl2 = (x2 * 3.2404542) + (y2 * -1.5371385) + (z2 * -0.4985314);
    let gl2 = (x2 * -0.9692660) + (y2 * 1.8760108) + (z2 * 0.0415560);
    let bl2 = (x2 * 0.0556434) + (y2 * -0.2040259) + (z2 * 1.0572252);
    (lin_to_byte(rl2), lin_to_byte(gl2), lin_to_byte(bl2))
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

#[cfg(test)]
mod rio_tests {
    use super::*;

    /// What color-operations makes of each formula, recorded with rio-tiler's own steps.
    const RIO: &str = include_str!("../tests/data/color_formula_rio.txt");

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    /// The formulas of one section of the recording, with their R, G and B bytes.
    fn section(name: &str) -> Vec<(String, [Vec<u8>; 3])> {
        let mut lines = RIO.lines().skip_while(|l| !l.starts_with(name)).skip(1);
        let mut out = Vec::new();
        for line in lines.by_ref() {
            let Some((formula, rest)) = line.split_once('|') else { break };
            let mut channels = rest.split('|').map(hex);
            out.push((formula.to_string(), [channels.next().unwrap(), channels.next().unwrap(), channels.next().unwrap()]));
        }
        out
    }

    #[test]
    fn every_byte_goes_where_rio_tiler_puts_it() {
        let cases = section("lut:");
        assert_eq!(cases.len(), 9);
        for (formula, [r, g, b]) in cases {
            let f = ColorFormula::parse(&formula).unwrap();
            for i in 0..=255u8 {
                let got = f.apply_rgb(i, i, i);
                assert_eq!(got, (r[usize::from(i)], g[usize::from(i)], b[usize::from(i)]), "{formula}: byte {i}");
            }
        }
    }

    #[test]
    fn saturation_lands_where_rio_tiler_puts_it() {
        let cases = section("grid:");
        assert_eq!(cases.len(), 4);
        let levels: Vec<u8> = (0..16).map(|i| (i * 17) as u8).collect();
        for (formula, [r, g, b]) in cases {
            let f = ColorFormula::parse(&formula).unwrap();
            let mut i = 0;
            for &red in &levels {
                for &green in &levels {
                    for &blue in &levels {
                        assert_eq!(f.apply_rgb(red, green, blue), (r[i], g[i], b[i]), "{formula}: ({red}, {green}, {blue})");
                        i += 1;
                    }
                }
            }
        }
    }

    #[test]
    fn the_threshold_search_gives_the_exact_byte() {
        let mut c = -0.01f64;
        while c < 1.02 {
            assert_eq!(lin_to_byte(c), lin_to_byte_exact(c), "{c}");
            c += 1.0 / 99_991.0;
        }
        for k in 0..256 {
            let edge = byte_thresholds()[k];
            for c in [edge, f64::from_bits(edge.to_bits().saturating_sub(1)), f64::from_bits(edge.to_bits() + 1)] {
                assert_eq!(lin_to_byte(c), lin_to_byte_exact(c), "byte {k} at {c}");
            }
        }
    }

    #[test]
    fn sigmoidal_rejects_out_of_range_bias_and_keeps_sub_epsilon_contrast_identity() {
        assert!(ColorFormula::parse("Sigmoidal RGB 10 1.5").is_err());
        assert!(ColorFormula::parse("Sigmoidal RGB 10 -0.1").is_err());
        assert!(ColorFormula::parse("Sigmoidal RGB 1000 5").is_err());
        for expr in ["Sigmoidal RGB 1e-20 0.5", "Sigmoidal RGB -1e-20 0.5"] {
            let f = ColorFormula::parse(expr).unwrap();
            assert_eq!(f.apply_rgb(10, 128, 250), (10, 128, 250), "{expr} must be identity");
        }
    }
}
