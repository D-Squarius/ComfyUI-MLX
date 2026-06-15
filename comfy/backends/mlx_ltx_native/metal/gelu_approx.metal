uint elem = thread_position_in_grid.x;
float x_f = float(x[elem]);
float inner = 0.7978845608028654f * (x_f + 0.044715f * x_f * x_f * x_f);
out[elem] = T(0.5f * x_f * (1.0f + metal::tanh(inner)));
