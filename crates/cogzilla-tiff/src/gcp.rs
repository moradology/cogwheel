use cogzilla_raster::Affine2D;

/// Maximum pixel residual (in pixels) below which a GCP set is represented as an exact
/// `Affine2D` with zero runtime polynomial overhead.
const AFFINE_RESIDUAL_THRESHOLD_PX: f64 = 0.05;

/// Normalized 2D polynomial warp (order 1, 2, or 3) fitted to Ground Control Points (GCPs).
///
/// Monomial basis (up to 10 terms for order 3):
/// `1, u, v, u^2, u*v, v^2, u^3, u^2*v, u*v^2, v^3`
#[derive(Debug, Clone)]
pub struct GcpPoly {
    pub order: u8,
    // Forward: pixel (col, row) -> CRS (x, y)
    u_mean: f64,
    u_scale: f64,
    v_mean: f64,
    v_scale: f64,
    fwd_x: [f64; 10],
    fwd_y: [f64; 10],
    // Inverse: CRS (x, y) -> pixel (col, row)
    x_mean: f64,
    x_scale: f64,
    y_mean: f64,
    y_scale: f64,
    inv_c: [f64; 10],
    inv_r: [f64; 10],
}

/// Result of fitting GCPs from TIFF `ModelTiepointTag` (33922).
#[derive(Debug, Clone)]
pub struct GcpFit {
    /// Best-fit order-1 affine (exact when `poly.is_none()`, approximate for resolution/overview
    /// selection when `poly.is_some()`).
    pub affine: Affine2D,
    /// `None` when the affine residual is `< 0.05 px` (zero overhead on hot loops);
    /// `Some(GcpPoly)` when an order-2 or order-3 polynomial warp is needed.
    pub poly: Option<GcpPoly>,
    /// Maximum pixel residual `hypot(pred_col - col, pred_row - row)` across all GCPs.
    pub residual_px: f64,
}

#[inline(always)]
fn term_count(order: u8) -> usize {
    match order {
        1 => 3,
        2 => 6,
        _ => 10,
    }
}

#[inline(always)]
fn eval_basis(u: f64, v: f64, order: u8) -> [f64; 10] {
    let mut b = [0.0f64; 10];
    b[0] = 1.0;
    b[1] = u;
    b[2] = v;
    if order >= 2 {
        let u2 = u * u;
        let uv = u * v;
        let v2 = v * v;
        b[3] = u2;
        b[4] = uv;
        b[5] = v2;
        if order >= 3 {
            b[6] = u2 * u;
            b[7] = u2 * v;
            b[8] = u * v2;
            b[9] = v2 * v;
        }
    }
    b
}

#[inline(always)]
fn dot_terms(coefs: &[f64; 10], basis: &[f64; 10], n: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..n {
        sum += coefs[i] * basis[i];
    }
    sum
}

impl GcpPoly {
    #[inline(always)]
    pub fn pixel_to_crs(&self, col: f64, row: f64) -> (f64, f64) {
        let u = (col - self.u_mean) / self.u_scale;
        let v = (row - self.v_mean) / self.v_scale;
        let b = eval_basis(u, v, self.order);
        let n = term_count(self.order);
        (dot_terms(&self.fwd_x, &b, n), dot_terms(&self.fwd_y, &b, n))
    }

    #[inline(always)]
    pub fn crs_to_pixel(&self, x: f64, y: f64) -> (f64, f64) {
        let xn = (x - self.x_mean) / self.x_scale;
        let yn = (y - self.y_mean) / self.y_scale;
        let b = eval_basis(xn, yn, self.order);
        let n = term_count(self.order);
        (dot_terms(&self.inv_c, &b, n), dot_terms(&self.inv_r, &b, n))
    }

    /// Scale this polynomial from full-resolution pixel coordinates to an overview IFD where
    /// `col_full = col_ov * scale_x` and `row_full = row_ov * scale_y`.
    pub fn scaled_for_overview(&self, scale_x: f64, scale_y: f64) -> Self {
        let mut out = self.clone();
        out.u_mean /= scale_x;
        out.u_scale /= scale_x;
        out.v_mean /= scale_y;
        out.v_scale /= scale_y;
        let inv_sx = 1.0 / scale_x;
        let inv_sy = 1.0 / scale_y;
        for c in &mut out.inv_c {
            *c *= inv_sx;
        }
        for r in &mut out.inv_r {
            *r *= inv_sy;
        }
        out
    }
}

/// Solve `n x n` linear system `A x = b` via Gaussian elimination with partial pivoting.
/// Returns `None` if the matrix is singular or ill-conditioned.
fn solve_linear(n: usize, a_in: &[[f64; 10]; 10], b_in: &[f64; 10]) -> Option<[f64; 10]> {
    let mut a = *a_in;
    let mut b = *b_in;

    for col in 0..n {
        let mut piv_row = col;
        let mut piv_val = a[col][col].abs();
        for (r, row) in a.iter().enumerate().take(n).skip(col + 1) {
            let v = row[col].abs();
            if v > piv_val {
                piv_val = v;
                piv_row = r;
            }
        }
        if !piv_val.is_finite() || piv_val < 1e-12 {
            return None;
        }
        if piv_row != col {
            a.swap(col, piv_row);
            b.swap(col, piv_row);
        }
        let diag = a[col][col];
        for r in (col + 1)..n {
            let factor = a[r][col] / diag;
            a[r][col] = 0.0;
            for c in (col + 1)..n {
                a[r][c] -= factor * a[col][c];
            }
            b[r] -= factor * b[col];
        }
    }

    let mut x = [0.0f64; 10];
    for r in (0..n).rev() {
        let mut s = b[r];
        for c in (r + 1)..n {
            s -= a[r][c] * x[c];
        }
        let val = s / a[r][r];
        if !val.is_finite() {
            return None;
        }
        x[r] = val;
    }
    Some(x)
}

fn mean_scale(vals: impl Iterator<Item = f64> + Clone) -> Option<(f64, f64)> {
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for v in vals.clone() {
        if !v.is_finite() {
            return None;
        }
        sum += v;
        cnt += 1;
    }
    if cnt == 0 {
        return None;
    }
    let mean = sum / (cnt as f64);
    let mut max_dev = 0.0f64;
    for v in vals {
        max_dev = max_dev.max((v - mean).abs());
    }
    let scale = if max_dev > 1e-12 { max_dev } else { 1.0 };
    Some((mean, scale))
}

/// Fit least-squares polynomial coefficients of `order` mapping normalized `(in_x, in_y)` to
/// `(out_x, out_y)`.
fn fit_poly_pair(
    pts: &[(f64, f64, f64, f64)],
    order: u8,
) -> Option<(f64, f64, f64, f64, [f64; 10], [f64; 10])> {
    let n = term_count(order);
    if pts.len() < n {
        return None;
    }
    let (mx, sx) = mean_scale(pts.iter().map(|p| p.0))?;
    let (my, sy) = mean_scale(pts.iter().map(|p| p.1))?;

    let mut ata = [[0.0f64; 10]; 10];
    let mut atbx = [0.0f64; 10];
    let mut atby = [0.0f64; 10];

    for &(ix, iy, ox, oy) in pts {
        let u = (ix - mx) / sx;
        let v = (iy - my) / sy;
        let b = eval_basis(u, v, order);
        for r in 0..n {
            let br = b[r];
            atbx[r] += br * ox;
            atby[r] += br * oy;
            for c in r..n {
                ata[r][c] += br * b[c];
            }
        }
    }
    for r in 0..n {
        for c in 0..r {
            ata[r][c] = ata[c][r];
        }
    }

    let cx = solve_linear(n, &ata, &atbx)?;
    let cy = solve_linear(n, &ata, &atby)?;
    Some((mx, sx, my, sy, cx, cy))
}

/// Fit an affine (or order 2–3 polynomial when residuals exceed 0.05 px) to TIFF `ModelTiepointTag`
/// 6-tuples `(col, row, z_px, x, y, z)`.
pub fn fit_gcps(tiepoints: &[f64], pixel_is_point: bool) -> Option<GcpFit> {
    let n_gcps = tiepoints.len() / 6;
    if n_gcps < 3 {
        return None;
    }
    let shift = if pixel_is_point { 0.5 } else { 0.0 };
    let mut fwd_pts = Vec::with_capacity(n_gcps);
    let mut inv_pts = Vec::with_capacity(n_gcps);
    for g in 0..n_gcps {
        let base = g * 6;
        let col = tiepoints[base] + shift;
        let row = tiepoints[base + 1] + shift;
        let x = tiepoints[base + 3];
        let y = tiepoints[base + 4];
        if !col.is_finite() || !row.is_finite() || !x.is_finite() || !y.is_finite() {
            return None;
        }
        fwd_pts.push((col, row, x, y));
        inv_pts.push((x, y, col, row));
    }

    // Stage 1: Order-1 least-squares affine fit.
    let (u_mean, u_scale, v_mean, v_scale, fx, fy) = fit_poly_pair(&fwd_pts, 1)?;
    let a = fx[1] / u_scale;
    let b = fx[2] / v_scale;
    let c = fx[0] - a * u_mean - b * v_mean;
    let d = fy[1] / u_scale;
    let e = fy[2] / v_scale;
    let f = fy[0] - d * u_mean - e * v_mean;
    let affine = Affine2D { a, b, c, d, e, f };
    let inv_affine = affine.invert()?;

    let mut affine_max_err = 0.0f64;
    for &(col, row, x, y) in &fwd_pts {
        let (pc, pr) = inv_affine.forward(x, y);
        affine_max_err = affine_max_err.max((pc - col).hypot(pr - row));
    }

    if affine_max_err < AFFINE_RESIDUAL_THRESHOLD_PX || n_gcps < 6 {
        return Some(GcpFit {
            affine,
            poly: None,
            residual_px: affine_max_err,
        });
    }

    // Stage 2: Try order 3 (n >= 10) then order 2 (n >= 6), keeping whichever reduces residual.
    let max_order = if n_gcps >= 10 { 3u8 } else { 2u8 };
    for order in (2..=max_order).rev() {
        if let (
            Some((um, us, vm, vs, fwd_x, fwd_y)),
            Some((xm, xs, ym, ys, inv_c, inv_r)),
        ) = (fit_poly_pair(&fwd_pts, order), fit_poly_pair(&inv_pts, order))
        {
            let poly = GcpPoly {
                order,
                u_mean: um,
                u_scale: us,
                v_mean: vm,
                v_scale: vs,
                fwd_x,
                fwd_y,
                x_mean: xm,
                x_scale: xs,
                y_mean: ym,
                y_scale: ys,
                inv_c,
                inv_r,
            };
            let mut poly_max_err = 0.0f64;
            for &(col, row, x, y) in &fwd_pts {
                let (pc, pr) = poly.crs_to_pixel(x, y);
                poly_max_err = poly_max_err.max((pc - col).hypot(pr - row));
            }
            if poly_max_err < affine_max_err {
                return Some(GcpFit {
                    affine,
                    poly: Some(poly),
                    residual_px: poly_max_err,
                });
            }
        }
    }

    Some(GcpFit {
        affine,
        poly: None,
        residual_px: affine_max_err,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_exact_affine_and_nonlinear_quadratic_gcps() {
        // 9 exact affine GCPs matching odd/gcp_affine.tif layout
        let mut tp = Vec::new();
        for &row in &[0.0, 256.0, 511.999] {
            for &col in &[0.0, 256.0, 511.999] {
                let x = -122.5 + col * 0.0005;
                let y = 37.9 - row * 0.0005;
                tp.extend_from_slice(&[col, row, 0.0, x, y, 0.0]);
            }
        }
        let fit = fit_gcps(&tp, false).unwrap();
        assert!(fit.poly.is_none(), "exact affine GCPs should use zero-overhead Affine2D");
        assert!(fit.residual_px < 1e-9, "residual={}", fit.residual_px);
        let (x, y) = fit.affine.forward(200.0, 200.0);
        assert!((x - (-122.4)).abs() < 1e-10);
        assert!((y - 37.8).abs() < 1e-10);

        // 9 nonlinear (quadratic bow) GCPs -> should select order-2 polynomial with near-zero residual
        let mut tp2 = Vec::new();
        for &row in &[0.0, 256.0, 512.0] {
            for &col in &[0.0, 256.0, 512.0] {
                let u = col / 512.0;
                let v = row / 512.0;
                let x = 10.0 + 0.5 * u + 0.02 * u * v + 0.01 * v * v;
                let y = 45.0 - 0.5 * v - 0.015 * u * u;
                tp2.extend_from_slice(&[col, row, 0.0, x, y, 0.0]);
            }
        }
        let fit2 = fit_gcps(&tp2, false).unwrap();
        let poly = fit2.poly.expect("nonlinear GCPs should fit order-2 GcpPoly");
        assert_eq!(poly.order, 2);
        assert!(fit2.residual_px < 0.2, "poly residual={}", fit2.residual_px);
        let (x, y) = poly.pixel_to_crs(128.0, 384.0);
        let (rc, rr) = poly.crs_to_pixel(x, y);
        assert!((rc - 128.0).abs() < 0.2 && (rr - 384.0).abs() < 0.2);

        // Overview scaling preserves exact CRS mapping
        let ov = poly.scaled_for_overview(2.0, 2.0);
        let (ox, oy) = ov.pixel_to_crs(64.0, 192.0);
        assert!((ox - x).abs() < 1e-12 && (oy - y).abs() < 1e-12);
        let (oc, or) = ov.crs_to_pixel(x, y);
        assert!((oc * 2.0 - rc).abs() < 1e-10 && (or * 2.0 - rr).abs() < 1e-10);
    }
}
