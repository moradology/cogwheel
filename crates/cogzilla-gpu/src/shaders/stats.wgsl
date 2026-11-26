// GPU Raster Band Statistics, Shared-Memory Atomic Histogram & Percentile CDF Reducer.
//
// Pass 1 (`stats_reduce_pass1`):
//   Parallel grid-stride workgroup reduction (`@workgroup_size(256, 1, 1)`) computing
//   exact valid pixel count, min, max, and compensated sum per workgroup.
//
// Pass 2 (`stats_hist_pass2`):
//   Workgroup-privatized shared-memory atomic histogram (9 KB total shared memory:
//   4 KB user histogram + 4 KB 1024-bin percentile CDF + 1 KB centered variance) flushed
//   once per workgroup into global atomics — zero global atomic contention.

struct StatsParams {
    len: u32,
    has_nodata: u32,
    num_bins: u32,
    has_hist_range: u32,
    nodata_val: f32,
    mean: f32,
    p_min: f32,
    p_inv_span: f32,
    h_min: f32,
    h_max: f32,
    h_inv_span: f32,
    _pad: f32,
}

struct PartialStats {
    min_val: f32,
    max_val: f32,
    sum: f32,
    count: u32,
}

@group(0) @binding(0) var<uniform> params: StatsParams;
@group(0) @binding(1) var<storage, read> pixels: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<PartialStats>;

var<workgroup> sh_min: array<f32, 256>;
var<workgroup> sh_max: array<f32, 256>;
var<workgroup> sh_sum: array<f32, 256>;
var<workgroup> sh_cnt: array<u32, 256>;

fn is_valid_pixel(v: f32) -> bool {
    let bits = bitcast<u32>(v);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        return false;
    }
    if (params.has_nodata != 0u && v == params.nodata_val) {
        return false;
    }
    return true;
}

@compute @workgroup_size(256, 1, 1)
fn stats_reduce_pass1(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let stride = num_wg.x * 256u;
    var idx = wgid.x * 256u + tid;

    var local_min: f32 = 3.402823466e+38;
    var local_max: f32 = -3.402823466e+38;
    var local_sum: f32 = 0.0;
    var local_comp: f32 = 0.0;
    var local_cnt: u32 = 0u;

    while (idx < params.len) {
        let v = pixels[idx];
        if (is_valid_pixel(v)) {
            local_min = min(local_min, v);
            local_max = max(local_max, v);
            // Kahan compensated summation in thread register
            let y = v - local_comp;
            let t = local_sum + y;
            local_comp = (t - local_sum) - y;
            local_sum = t;
            local_cnt = local_cnt + 1u;
        }
        idx = idx + stride;
    }

    sh_min[tid] = local_min;
    sh_max[tid] = local_max;
    sh_sum[tid] = local_sum;
    sh_cnt[tid] = local_cnt;
    workgroupBarrier();

    var s: u32 = 128u;
    while (s > 0u) {
        if (tid < s) {
            let c_other = sh_cnt[tid + s];
            if (c_other > 0u) {
                if (sh_cnt[tid] == 0u) {
                    sh_min[tid] = sh_min[tid + s];
                    sh_max[tid] = sh_max[tid + s];
                } else {
                    sh_min[tid] = min(sh_min[tid], sh_min[tid + s]);
                    sh_max[tid] = max(sh_max[tid], sh_max[tid + s]);
                }
                sh_sum[tid] = sh_sum[tid] + sh_sum[tid + s];
                sh_cnt[tid] = sh_cnt[tid] + c_other;
            }
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        partials[wgid.x] = PartialStats(sh_min[0], sh_max[0], sh_sum[0], sh_cnt[0]);
    }
}

// Pass 2 bindings: 0=params, 1=pixels, 2=hist_out (2048 atomic<u32>), 3=var_partials (array<f32>)
@group(0) @binding(2) var<storage, read_write> hist_out: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> var_partials: array<f32>;

var<workgroup> sh_hist: array<atomic<u32>, 1024>;
var<workgroup> sh_cdf: array<atomic<u32>, 1024>;
var<workgroup> sh_var: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn stats_hist_pass2(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    // Zero 1024 bins per histogram (4 bins per thread)
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let b = tid + k * 256u;
        atomicStore(&sh_hist[b], 0u);
        atomicStore(&sh_cdf[b], 0u);
    }
    workgroupBarrier();

    let stride = num_wg.x * 256u;
    var idx = wgid.x * 256u + tid;
    var local_var: f32 = 0.0;
    let n_bins_f = f32(params.num_bins);
    let max_bin = params.num_bins - 1u;

    while (idx < params.len) {
        let v = pixels[idx];
        if (is_valid_pixel(v)) {
            let d = v - params.mean;
            local_var = local_var + d * d;

            // 1024-bin CDF histogram over [p_min, p_max] for sub-0.1% percentiles
            let cdf_f = floor((v - params.p_min) * params.p_inv_span * 1024.0);
            let cdf_bin = u32(clamp(cdf_f, 0.0, 1023.0));
            atomicAdd(&sh_cdf[cdf_bin], 1u);

            // User-requested N-bin histogram over [h_min, h_max]
            if (params.has_hist_range == 0u || (v >= params.h_min && v <= params.h_max)) {
                let hf = floor((v - params.h_min) * params.h_inv_span * n_bins_f);
                let hbin = u32(clamp(hf, 0.0, f32(max_bin)));
                atomicAdd(&sh_hist[hbin], 1u);
            }
        }
        idx = idx + stride;
    }

    sh_var[tid] = local_var;
    workgroupBarrier();

    // Flush privatized histograms into global atomics (4 bins per thread)
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let b = tid + k * 256u;
        if (b < params.num_bins) {
            let c = atomicLoad(&sh_hist[b]);
            if (c > 0u) {
                atomicAdd(&hist_out[b], c);
            }
        }
        let cc = atomicLoad(&sh_cdf[b]);
        if (cc > 0u) {
            atomicAdd(&hist_out[1024u + b], cc);
        }
    }

    // Reduce variance in shared memory
    var s: u32 = 128u;
    while (s > 0u) {
        if (tid < s) {
            sh_var[tid] = sh_var[tid] + sh_var[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (tid == 0u) {
        var_partials[wgid.x] = sh_var[0];
    }
}
