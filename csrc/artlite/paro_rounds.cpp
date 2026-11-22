#include "artlite/paro_rounds.h"

#include <cmath>
#include <cstddef>

namespace artlite {

void paro_rounds(int rounds, int groups, int group, const int64_t* pairs, const double* cs, const double* sn,
                 double* d, uint8_t* partner, double* coef, int64_t* pos, int64_t* origin) {
  const int half = group / 2;
  for (int G = 0; G < groups; G++) {
    double* dg = d + static_cast<size_t>(G) * group;
    int64_t* og = origin + static_cast<size_t>(G) * group;
    int64_t* pg = pos + static_cast<size_t>(G) * group;
    for (int c = 0; c < group; c++) og[c] = pg[c] = c;
    for (int r = 0; r < rounds; r++) {
      const size_t rg = (static_cast<size_t>(r) * groups + G);
      const int64_t* p = pairs + rg * group;
      const double* c_ = cs + rg * half;
      const double* s_ = sn + rg * half;
      uint8_t* pt = partner + rg * group;
      double* cf_ = coef + rg * group;
      for (int k = 0; k < half; k++) {  // (a round's pairs are disjoint: in turn as numpy's at once)
        const int64_t i = p[2 * k], j = p[2 * k + 1];
        const double c = c_[k], s = s_[k];
        const int64_t P = pg[i], Q = pg[j];
        const double di = dg[i], dj = dg[j];
        const int64_t oi = og[i], oj = og[j];
        const bool cf = std::fabs(c) >= std::fabs(s);  // (else the outputs swap storage: the tangent stays at most 1)
        const double t = cf ? s / c : c / s;
        pt[P] = static_cast<uint8_t>(Q);
        pt[Q] = static_cast<uint8_t>(P);
        cf_[P] = cf ? t * dj / di : -t * dj / di;
        cf_[Q] = cf ? -t * di / dj : t * di / dj;
        dg[i] = cf ? c * di : s * dj;
        dg[j] = cf ? c * dj : -s * di;
        og[i] = cf ? oi : oj;
        og[j] = cf ? oj : oi;
        pg[i] = cf ? P : Q;
        pg[j] = cf ? Q : P;
      }
    }
  }
}

}  // namespace artlite
