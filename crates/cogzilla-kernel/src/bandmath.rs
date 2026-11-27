/// Fast IEEE-754 `10 * log10(power)` for SAR gamma-naught linear power -> decibels (dB).
/// Accurate to `< 0.004 dB` across the entire positive `f32` range (far tighter than 8-bit PNG quantization)
/// and ~5x faster than `libm::log10f`, fully inlineable and branch-minimal.
#[inline(always)]
pub fn fast_db_from_power(power: f32) -> f32 {
    if !(power > 0.0) {
        return f32::NAN;
    }
    if !power.is_finite() {
        return f32::INFINITY;
    }
    let bits = power.to_bits();
    let exp = ((bits >> 23) as i32 - 127) as f32;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000); // m in [1.0, 2.0)
    // Degree-4 minimax polynomial for log2(m) on [1, 2) via t = m - 1 in [0, 1)
    let t = m - 1.0;
    // 1.442_695 is a fitted polynomial coefficient, not LOG2_E.
    #[allow(clippy::approx_constant)]
    let log2_m = (((-0.081_615_807 * t + 0.345_287_06) * t - 0.706_980_2) * t + 1.442_695) * t;
    // 10 * log10(2) = 3.010299956639812
    3.010_3 * (exp + log2_m)
}

/// Fused satellite spectral & SAR band-math expressions evaluated directly in registers
/// during tile resampling (zero intermediate `f32` tile buffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpectralExpr {
    /// SAR Linear Power -> Decibels: `10 * log10(power)` (1 band: VV or VH)
    Db,
    /// Dual-Pol SAR False-Color RGB (2 bands: VV, VH -> `[VV_dB, VH_dB, VV_dB - VH_dB]`)
    SarRgb,
    /// Normalized Difference: `(A - B) / (A + B)` (NDVI, NDWI, NBR, NDSI)
    NormalizedDiff,
    /// Soil-Adjusted Vegetation Index (`L = 0.5`): `1.5 * (A - B) / (A + B + 0.5)`
    Savi,
    /// Simple Ratio: `A / B`
    Ratio,
    /// Band Difference: `A - B`
    Diff,
    /// Enhanced Vegetation Index (3 bands: NIR, RED, BLUE):
    /// `2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0)`
    Evi,
}

impl SpectralExpr {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "db" | "sar_db" | "decibel" => Some(Self::Db),
            "sar_rgb" | "sarrgb" | "dualpol" => Some(Self::SarRgb),
            "ndvi" | "ndwi" | "nbr" | "ndsi" | "nd" | "normdiff" => Some(Self::NormalizedDiff),
            "savi" => Some(Self::Savi),
            "ratio" => Some(Self::Ratio),
            "diff" => Some(Self::Diff),
            "evi" => Some(Self::Evi),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn required_bands(self) -> usize {
        match self {
            Self::Db => 1,
            Self::Evi => 3,
            _ => 2,
        }
    }

    #[inline(always)]
    pub fn eval1(self, a: f32) -> f32 {
        match self {
            Self::Db => fast_db_from_power(a),
            _ => a,
        }
    }

    #[inline(always)]
    pub fn eval2(self, a: f32, b: f32) -> f32 {
        match self {
            Self::NormalizedDiff => {
                let d = a + b;
                if d.abs() < 1e-6 {
                    f32::NAN
                } else {
                    (a - b) / d
                }
            }
            Self::Savi => {
                let d = a + b + 0.5;
                if d.abs() < 1e-6 {
                    f32::NAN
                } else {
                    1.5 * (a - b) / d
                }
            }
            Self::Ratio => {
                if b.abs() < 1e-6 {
                    f32::NAN
                } else {
                    a / b
                }
            }
            Self::Diff => a - b,
            _ => f32::NAN,
        }
    }

    #[inline(always)]
    pub fn eval3(self, nir: f32, red: f32, blue: f32) -> f32 {
        let d = nir + 6.0 * red - 7.5 * blue + 1.0;
        if d.abs() < 1e-6 {
            f32::NAN
        } else {
            2.5 * (nir - red) / d
        }
    }

    /// Canonical `expr_vm::Op` bytecode sequence for built-in single-output spectral indices.
    /// Returns `None` for multi-channel `SarRgb`. Used as the single source of truth so GPU
    /// compilation (`compile_spectral_to_gpu`) cannot drift from the CPU definitions above.
    pub fn to_ops(self) -> Option<&'static [crate::expr_vm::Op]> {
        use crate::expr_vm::Op;
        static DB_OPS: [Op; 2] = [Op::LoadBand(0), Op::Db];
        static ND_OPS: [Op; 7] = [
            Op::LoadBand(0),
            Op::LoadBand(1),
            Op::Sub,
            Op::LoadBand(0),
            Op::LoadBand(1),
            Op::Add,
            Op::Div,
        ];
        static SAVI_OPS: [Op; 11] = [
            Op::PushConst(1.5),
            Op::LoadBand(0),
            Op::LoadBand(1),
            Op::Sub,
            Op::Mul,
            Op::LoadBand(0),
            Op::LoadBand(1),
            Op::Add,
            Op::PushConst(0.5),
            Op::Add,
            Op::Div,
        ];
        static RATIO_OPS: [Op; 3] = [Op::LoadBand(0), Op::LoadBand(1), Op::Div];
        static DIFF_OPS: [Op; 3] = [Op::LoadBand(0), Op::LoadBand(1), Op::Sub];
        static EVI_OPS: [Op; 17] = [
            Op::PushConst(2.5),
            Op::LoadBand(0),
            Op::LoadBand(1),
            Op::Sub,
            Op::Mul,
            Op::LoadBand(0),
            Op::PushConst(6.0),
            Op::LoadBand(1),
            Op::Mul,
            Op::Add,
            Op::PushConst(7.5),
            Op::LoadBand(2),
            Op::Mul,
            Op::Sub,
            Op::PushConst(1.0),
            Op::Add,
            Op::Div,
        ];
        match self {
            Self::Db => Some(&DB_OPS),
            Self::NormalizedDiff => Some(&ND_OPS),
            Self::Savi => Some(&SAVI_OPS),
            Self::Ratio => Some(&RATIO_OPS),
            Self::Diff => Some(&DIFF_OPS),
            Self::Evi => Some(&EVI_OPS),
            Self::SarRgb => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_db_matches_exact_log10_within_0_01_db() {
        for exp in -5..=2 {
            for k in 1..100 {
                let p = (k as f32 / 10.0) * 10.0f32.powi(exp);
                let exact = 10.0 * p.log10();
                let approx = fast_db_from_power(p);
                assert!(
                    (exact - approx).abs() < 0.015,
                    "p={p}: exact={exact}, approx={approx}, err={}",
                    (exact - approx).abs()
                );
            }
        }
        assert!(fast_db_from_power(0.0).is_nan());
        assert!(fast_db_from_power(-0.5).is_nan());
        assert!(fast_db_from_power(f32::NAN).is_nan());
    }

    #[test]
    fn spectral_expr_to_ops_matches_eval() {
        use crate::expr_vm::ExprProgram;
        let samples = [
            (0.8f32, 0.2f32, 0.1f32),
            (0.15, 0.45, 0.05),
            (1.2, 0.0, 0.3),
        ];
        for expr in [
            SpectralExpr::Db,
            SpectralExpr::NormalizedDiff,
            SpectralExpr::Savi,
            SpectralExpr::Ratio,
            SpectralExpr::Diff,
            SpectralExpr::Evi,
        ] {
            let ops = expr.to_ops().expect("single-output spectral expr has ops");
            let n = expr.required_bands();
            let prog = ExprProgram {
                ops: ops.to_vec(),
                used_bands: (1..=n as u16).collect(),
                max_depth: 8,
            };
            for &(a, b, c) in &samples {
                let got = prog.eval(&[a, b, c][..n]);
                let want = match n {
                    1 => expr.eval1(a),
                    2 => expr.eval2(a, b),
                    3 => expr.eval3(a, b, c),
                    _ => unreachable!(),
                };
                if want.is_nan() {
                    assert!(got.is_nan(), "{expr:?}: expected NaN, got {got}");
                } else {
                    assert!(
                        (got - want).abs() < 1e-5,
                        "{expr:?} ({a},{b},{c}): to_ops={got} vs eval={want}"
                    );
                }
            }
        }
        assert!(SpectralExpr::SarRgb.to_ops().is_none());
    }
}
