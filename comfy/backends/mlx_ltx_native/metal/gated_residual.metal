uint elem = thread_position_in_grid.x;
uint col = elem % {{HIDDEN_SIZE}};
uint row = elem / {{HIDDEN_SIZE}};
uint gate_idx = {{GATE_MODE}} == 0 ? col : ({{GATE_MODE}} == 1 ? (row / {{GATE_ROWS_PER_BATCH}}) * {{HIDDEN_SIZE}} + col : elem);
float residual_f = float(residual[elem]);
float value_f = float(value[elem]);
float gate_f = float(gate[gate_idx]);
out[elem] = T(residual_f + value_f * gate_f);
