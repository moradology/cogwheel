fn eval_uv(px: u32, py: u32) -> vec2<f32> {
    if (params.coord_mode == 0u) {
        return vec2<f32>(u_coords[px], v_coords[py]);
    } else if (params.coord_mode == 1u) {
        let gx_f = f32(px) * 16.0 / f32(params.dst_w);
        let gy_f = f32(py) * 16.0 / f32(params.dst_h);
        let gx = min(u32(gx_f), 15u);
        let gy = min(u32(gy_f), 15u);
        let tx = gx_f - f32(gx);
        let ty = gy_f - f32(gy);
        let r0 = gy * 17u + gx;
        let r1 = (gy + 1u) * 17u + gx;
        let u_l = u_coords[r0] + (u_coords[r1] - u_coords[r0]) * ty;
        let u_r = u_coords[r0 + 1u] + (u_coords[r1 + 1u] - u_coords[r0 + 1u]) * ty;
        let v_l = v_coords[r0] + (v_coords[r1] - v_coords[r0]) * ty;
        let v_r = v_coords[r0 + 1u] + (v_coords[r1 + 1u] - v_coords[r0 + 1u]) * ty;
        return vec2<f32>(u_l + (u_r - u_l) * tx, v_l + (v_r - v_l) * tx);
    } else if (params.coord_mode == 2u) {
        let idx = py * params.dst_w + px;
        return vec2<f32>(u_coords[idx], v_coords[idx]);
    } else if (params.coord_mode == 3u) {
        // SeparableMasked: u_coords = [u_cols (dst_w), v_rows (dst_h)], v_coords = 1-bit mask
        let idx = py * params.dst_w + px;
        let word = bitcast<u32>(v_coords[idx >> 5u]);
        if ((word & (1u << (idx & 31u))) == 0u) {
            return vec2<f32>(nan_f32(), nan_f32());
        }
        return vec2<f32>(u_coords[px], u_coords[params.dst_w + py]);
    } else {
        // Grid2DMasked (coord_mode == 4u): u_coords = [grid_u (289), grid_v (289)], v_coords = 1-bit mask
        let idx = py * params.dst_w + px;
        let word = bitcast<u32>(v_coords[idx >> 5u]);
        if ((word & (1u << (idx & 31u))) == 0u) {
            return vec2<f32>(nan_f32(), nan_f32());
        }
        let gx_f = f32(px) * 16.0 / f32(params.dst_w);
        let gy_f = f32(py) * 16.0 / f32(params.dst_h);
        let gx = min(u32(gx_f), 15u);
        let gy = min(u32(gy_f), 15u);
        let tx = gx_f - f32(gx);
        let ty = gy_f - f32(gy);
        let r0 = gy * 17u + gx;
        let r1 = (gy + 1u) * 17u + gx;
        let u_l = u_coords[r0] + (u_coords[r1] - u_coords[r0]) * ty;
        let u_r = u_coords[r0 + 1u] + (u_coords[r1 + 1u] - u_coords[r0 + 1u]) * ty;
        let v_l = u_coords[289u + r0] + (u_coords[289u + r1] - u_coords[289u + r0]) * ty;
        let v_r = u_coords[289u + r0 + 1u] + (u_coords[289u + r1 + 1u] - u_coords[289u + r0 + 1u]) * ty;
        return vec2<f32>(u_l + (u_r - u_l) * tx, v_l + (v_r - v_l) * tx);
    }
}
