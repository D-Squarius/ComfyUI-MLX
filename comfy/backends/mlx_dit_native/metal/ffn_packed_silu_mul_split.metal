uint elem = thread_position_in_grid.x;
uint hidden = uint({{HIDDEN_DIM}});
uint col = elem % hidden;
uint row = elem / hidden;
uint in_base = row * hidden * 2 + col;
T gate = gu[in_base];
T up = gu[in_base + hidden];
out[elem] = (gate / (T(1.0) + metal::exp(-gate))) * up;
