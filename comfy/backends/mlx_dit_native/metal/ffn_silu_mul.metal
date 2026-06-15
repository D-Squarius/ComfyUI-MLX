uint elem = thread_position_in_grid.x;
T x = a[elem];
T y = b[elem];
out[elem] = (x / (T(1.0) + metal::exp(-x))) * y;
