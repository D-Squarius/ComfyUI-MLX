uint vector_index = thread_position_in_grid.x;
uint t = vector_index % {{TOKENS}};
uint h = (vector_index / {{TOKENS}}) % {{HEADS}};
uint b = vector_index / ({{TOKENS}} * {{HEADS}});
uint in_base = ((b * {{TOKENS}} + t) * {{HEADS}} + h) * {{HEAD_DIM}};
uint out_base = ((b * {{HEADS}} + h) * {{TOKENS}} + t) * {{HEAD_DIM}};

float q_sum_sq = 0.0f;
float k_sum_sq = 0.0f;
for (uint d = 0; d < {{HEAD_DIM}}; ++d) {
    float qv = float(q[in_base + d]);
    float kv = float(k[in_base + d]);
    q_sum_sq += qv * qv;
    k_sum_sq += kv * kv;
}

float q_inv_rms = metal::rsqrt((q_sum_sq / float({{HEAD_DIM}})) + float({{EPS}}));
float k_inv_rms = metal::rsqrt((k_sum_sq / float({{HEAD_DIM}})) + float({{EPS}}));
uint freq_b = {{FREQ_BATCH}} == 1 ? 0 : b;
uint freq_base = (freq_b * {{TOKENS}} + t) * ({{HEAD_DIM}} / 2) * 4;

for (uint pair = 0; pair < {{HEAD_DIM}} / 2; ++pair) {
    uint even = pair * 2;
    uint odd = even + 1;
    uint freq_offset = freq_base + pair * 4;
    float c0 = float(freqs[freq_offset]);
    float c1 = float(freqs[freq_offset + 1]);
    float c2 = float(freqs[freq_offset + 2]);
    float c3 = float(freqs[freq_offset + 3]);

    float q0 = float(q[in_base + even]) * q_inv_rms * float(q_weight[even]);
    float q1 = float(q[in_base + odd]) * q_inv_rms * float(q_weight[odd]);
    float k0 = float(k[in_base + even]) * k_inv_rms * float(k_weight[even]);
    float k1 = float(k[in_base + odd]) * k_inv_rms * float(k_weight[odd]);

    q_out[out_base + even] = T(q0 * c0 + q1 * c1);
    q_out[out_base + odd] = T(q0 * c2 + q1 * c3);
    k_out[out_base + even] = T(k0 * c0 + k1 * c1);
    k_out[out_base + odd] = T(k0 * c2 + k1 * c3);
    v_out[out_base + even] = v[in_base + even];
    v_out[out_base + odd] = v[in_base + odd];
}
