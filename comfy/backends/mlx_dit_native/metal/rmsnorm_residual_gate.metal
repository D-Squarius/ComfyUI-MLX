uint row = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint base = row * {{HIDDEN_SIZE}};

threadgroup float partial[{{THREADS}}];

float local_sum = 0.0f;
for (uint i = lid; i < {{HIDDEN_SIZE}}; i += {{THREADS}}) {
    float value = float(x[base + i]);
    local_sum += value * value;
}
partial[lid] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = {{THREADS}} >> 1; stride > 0; stride >>= 1) {
    if (lid < stride) {
        partial[lid] += partial[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

float inv_rms = metal::rsqrt((partial[0] / float({{HIDDEN_SIZE}})) + float({{EPS}}));

for (uint col = lid; col < {{HIDDEN_SIZE}}; col += {{THREADS}}) {
    uint elem = base + col;
    uint gate_idx = {{GATE_MODE}} == 0 ? col : ({{GATE_MODE}} == 1 ? (row / {{GATE_ROWS_PER_BATCH}}) * {{HIDDEN_SIZE}} + col : elem);
    float normed = float(x[elem]) * inv_rms * float(weight[col]);
    out[elem] = T(float(residual[elem]) + normed * float(gate[gate_idx]));
}
