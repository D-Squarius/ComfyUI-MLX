uint elem = thread_position_in_grid.x;
uint col = elem % {{HIDDEN_SIZE}};
uint row = elem / {{HIDDEN_SIZE}};
uint scale_idx = {{SCALE_MODE}} == 0 ? col : ({{SCALE_MODE}} == 1 ? (row / {{SCALE_ROWS_PER_BATCH}}) * {{HIDDEN_SIZE}} + col : elem);
uint shift_idx = {{SHIFT_MODE}} == 0 ? col : ({{SHIFT_MODE}} == 1 ? (row / {{SHIFT_ROWS_PER_BATCH}}) * {{HIDDEN_SIZE}} + col : elem);
float x_f = float(x[elem]);
float scale_f = float(scale[scale_idx]);
float shift_f = float(shift[shift_idx]);
out[elem] = T(x_f * (1.0f + scale_f) + shift_f);
