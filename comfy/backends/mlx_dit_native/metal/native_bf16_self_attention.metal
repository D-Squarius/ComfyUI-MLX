uint elem = thread_position_in_grid.x;
uint d = elem % {{HEAD_DIM}};
uint h = (elem / {{HEAD_DIM}}) % {{HEADS}};
uint t = (elem / ({{HEAD_DIM}} * {{HEADS}})) % {{TOKENS}};
uint b = elem / ({{HEAD_DIM}} * {{HEADS}} * {{TOKENS}});
uint q_base = ((b * {{TOKENS}} + t) * {{HEADS}} + h) * {{HEAD_DIM}};
uint freq_b_q = {{FREQ_BATCH}} == 1 ? 0 : b;
uint q_freq_base = (freq_b_q * {{TOKENS}} + t) * ({{HEAD_DIM}} / 2) * 4;

float q_sum_sq = 0.0f;
for (uint r = 0; r < {{HEAD_DIM}}; ++r) {
    float qv = float(q[q_base + r]);
    q_sum_sq += qv * qv;
}
float q_inv_rms = metal::rsqrt((q_sum_sq / float({{HEAD_DIM}})) + float({{EPS}}));

float max_score = -INFINITY;
for (uint key_t = 0; key_t < {{TOKENS}}; ++key_t) {
    uint k_base = ((b * {{TOKENS}} + key_t) * {{HEADS}} + h) * {{HEAD_DIM}};
    uint freq_b_k = {{FREQ_BATCH}} == 1 ? 0 : b;
    uint k_freq_base = (freq_b_k * {{TOKENS}} + key_t) * ({{HEAD_DIM}} / 2) * 4;

    float k_sum_sq = 0.0f;
    for (uint r = 0; r < {{HEAD_DIM}}; ++r) {
        float kv = float(k[k_base + r]);
        k_sum_sq += kv * kv;
    }
    float k_inv_rms = metal::rsqrt((k_sum_sq / float({{HEAD_DIM}})) + float({{EPS}}));

    float score = 0.0f;
    for (uint pair = 0; pair < {{HEAD_DIM}} / 2; ++pair) {
        uint even = pair * 2;
        uint odd = even + 1;

        uint q_freq_offset = q_freq_base + pair * 4;
        float qc0 = float(freqs[q_freq_offset]);
        float qc1 = float(freqs[q_freq_offset + 1]);
        float qc2 = float(freqs[q_freq_offset + 2]);
        float qc3 = float(freqs[q_freq_offset + 3]);
        float q0 = float(q[q_base + even]) * q_inv_rms * float(q_weight[even]);
        float q1 = float(q[q_base + odd]) * q_inv_rms * float(q_weight[odd]);
        float qr0 = q0 * qc0 + q1 * qc1;
        float qr1 = q0 * qc2 + q1 * qc3;

        uint k_freq_offset = k_freq_base + pair * 4;
        float kc0 = float(freqs[k_freq_offset]);
        float kc1 = float(freqs[k_freq_offset + 1]);
        float kc2 = float(freqs[k_freq_offset + 2]);
        float kc3 = float(freqs[k_freq_offset + 3]);
        float k0 = float(k[k_base + even]) * k_inv_rms * float(k_weight[even]);
        float k1 = float(k[k_base + odd]) * k_inv_rms * float(k_weight[odd]);
        float kr0 = k0 * kc0 + k1 * kc1;
        float kr1 = k0 * kc2 + k1 * kc3;

        score += qr0 * kr0 + qr1 * kr1;
    }
    score *= float({{SCALE}});
    max_score = metal::max(max_score, score);
}

float denom = 0.0f;
float acc = 0.0f;
for (uint key_t = 0; key_t < {{TOKENS}}; ++key_t) {
    uint k_base = ((b * {{TOKENS}} + key_t) * {{HEADS}} + h) * {{HEAD_DIM}};
    uint freq_b_k = {{FREQ_BATCH}} == 1 ? 0 : b;
    uint k_freq_base = (freq_b_k * {{TOKENS}} + key_t) * ({{HEAD_DIM}} / 2) * 4;

    float k_sum_sq = 0.0f;
    for (uint r = 0; r < {{HEAD_DIM}}; ++r) {
        float kv = float(k[k_base + r]);
        k_sum_sq += kv * kv;
    }
    float k_inv_rms = metal::rsqrt((k_sum_sq / float({{HEAD_DIM}})) + float({{EPS}}));

    float score = 0.0f;
    for (uint pair = 0; pair < {{HEAD_DIM}} / 2; ++pair) {
        uint even = pair * 2;
        uint odd = even + 1;

        uint q_freq_offset = q_freq_base + pair * 4;
        float qc0 = float(freqs[q_freq_offset]);
        float qc1 = float(freqs[q_freq_offset + 1]);
        float qc2 = float(freqs[q_freq_offset + 2]);
        float qc3 = float(freqs[q_freq_offset + 3]);
        float q0 = float(q[q_base + even]) * q_inv_rms * float(q_weight[even]);
        float q1 = float(q[q_base + odd]) * q_inv_rms * float(q_weight[odd]);
        float qr0 = q0 * qc0 + q1 * qc1;
        float qr1 = q0 * qc2 + q1 * qc3;

        uint k_freq_offset = k_freq_base + pair * 4;
        float kc0 = float(freqs[k_freq_offset]);
        float kc1 = float(freqs[k_freq_offset + 1]);
        float kc2 = float(freqs[k_freq_offset + 2]);
        float kc3 = float(freqs[k_freq_offset + 3]);
        float k0 = float(k[k_base + even]) * k_inv_rms * float(k_weight[even]);
        float k1 = float(k[k_base + odd]) * k_inv_rms * float(k_weight[odd]);
        float kr0 = k0 * kc0 + k1 * kc1;
        float kr1 = k0 * kc2 + k1 * kc3;

        score += qr0 * kr0 + qr1 * kr1;
    }
    score *= float({{SCALE}});
    float weight = metal::exp(score - max_score);
    denom += weight;
    acc += weight * float(v[k_base + d]);
}

out[elem] = T(acc / denom);
