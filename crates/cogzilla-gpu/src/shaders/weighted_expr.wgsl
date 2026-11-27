// Direct 1:1 native-window GPU expression VM pass chained into `weighted_stats.wgsl`.
// Evaluates `GpuInstr` bytecode only on pixels with `cov[i] > cov_eps`, writing directly
// into a `WeightedBandSlot::src_buf` with zero CPU expression evaluation and zero extra readback.

struct WeightedExprParams {
    len: u32,
    /// Bitmask of which stacked `raw_bands` slots (`0..8`) this expression references.
    band_mask: u32,
    pc_start: u32,
    pc_end: u32,
    cov_eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: WeightedExprParams;
/// Stacked input raw bands: band `b` at `b * params.len + i` (up to 8 bands).
@group(0) @binding(1) var<storage, read> raw_bands: array<f32>;
@group(0) @binding(2) var<storage, read> cov: array<f32>;
@group(0) @binding(3) var<storage, read> instrs: array<GpuInstr>;
@group(0) @binding(4) var<storage, read_write> dst_expr: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn weighted_expr_eval(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * 256u;
    for (var i: u32 = gid.x; i < params.len; i = i + stride) {
        if (cov[i] <= params.cov_eps) {
            dst_expr[i] = nan_f32();
            continue;
        }

        var bvals: array<f32, 8>;
        var all_valid = true;
        let mask = params.band_mask;
        for (var b: u32 = 0u; b < 8u; b = b + 1u) {
            if ((mask & (1u << b)) != 0u) {
                let v = raw_bands[b * params.len + i];
                if (!is_finite_f32(v)) {
                    dst_expr[i] = nan_f32();
                    all_valid = false;
                    break;
                }
                bvals[b] = v;
            }
        }
        if (!all_valid) {
            continue;
        }

        dst_expr[i] = eval_expr_vm_range(&bvals, params.pc_start, params.pc_end);
    }
}
