uint row = thread_position_in_grid.x;
uint col = thread_position_in_grid.y;
uint base = row * {{HIDDEN_SIZE}};
float sum_sq = 0.0f;
for (uint i = 0; i < {{HIDDEN_SIZE}}; ++i) {
    float v = float(x[base + i]);
    sum_sq += v * v;
}
float inv_rms = metal::rsqrt((sum_sq / float({{HIDDEN_SIZE}})) + float({{EPS}}));
out[base + col] = T(float(x[base + col]) * inv_rms * float(weight[col]) + float(residual[base + col]));
