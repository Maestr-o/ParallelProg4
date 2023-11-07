__kernel void krnl(long N, long r, __global long* X, __global long* Y,
                   __global long* Nmin) {
  long id = N + get_global_id(0);
  long M, a, b;
  a = r;
  b = r;
  int br = 0;

  while (true) {
    M = 0;
    for (long j = id; j > 0; j /= 10) {
      M *= 10;
      M += j % 10;
    }
    if (M == id) {
      a = r;
      b = r;
      for (; a * b <= M; a++) {
        for (; a * b <= M; b++) {
          if (a * b == M && a != b) {
            if (*Nmin == 0 || M < *Nmin) {
              *X = a;
              *Y = b;
              *Nmin = M;
            }

            br = 1;
            break;
          }
        }
        if (br == 1) break;
        b = a;
      }
      if (br == 1) break;
    }
    if (*Nmin != 0) break;
    if (br == 1) break;
    id = id + 200;
  }
}
