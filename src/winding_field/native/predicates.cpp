#include "predicates.h"
#include <algorithm>
#include <cfenv>
#include <limits>

bool wf_host_environment() {
    return std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53
        && std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits == 24
        && std::fegetround() == FE_TONEAREST && wf_numeric_probe();
}

namespace {
using Int = __int128;
struct Vec {
    Int x[3];
    bool operator==(const Vec& b) const { return x[0] == b.x[0] && x[1] == b.x[1] && x[2] == b.x[2]; }
};
Vec sub(const Vec& a, const Vec& b) { return {{a.x[0]-b.x[0], a.x[1]-b.x[1], a.x[2]-b.x[2]}}; }
Vec cross(const Vec& a, const Vec& b) {
    return {{a.x[1]*b.x[2]-a.x[2]*b.x[1], a.x[2]*b.x[0]-a.x[0]*b.x[2], a.x[0]*b.x[1]-a.x[1]*b.x[0]}};
}
Int dot(const Vec& a, const Vec& b) { return a.x[0]*b.x[0]+a.x[1]*b.x[1]+a.x[2]*b.x[2]; }
bool any(const Vec& v) { return v.x[0] || v.x[1] || v.x[2]; }
bool contains(const Vec* a, int n, const Vec& p) {
    for (int i = 0; i < n; ++i) if (a[i] == p) return true;
    return false;
}
bool on_allowed(const Vec& p, const Vec* shared, int count) {
    if (!count) return false;
    if (count == 1) return p == shared[0];
    Vec delta = sub(shared[1], shared[0]), offset = sub(p, shared[0]);
    if (!any(delta)) return p == shared[0];
    if (any(cross(delta, offset))) return false;
    Int t = dot(offset, delta);
    return t >= 0 && t <= dot(delta, delta);
}

int8_t contact(const double* row) {
    int scale = 0;
    bool nonzero = false;
    for (int i = 0; i < 18; ++i) {
        double x = row[i];
        if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<float>::max()
            || double(float(x)) != x) return -1;
        if (x != 0.) {
            int exponent;
            std::frexp(x, &exponent);
            if (!nonzero || exponent-24 < scale) scale = exponent-24;
            nonzero = true;
        }
    }
    int64_t scaled[18];
    for (int i = 0; i < 18; ++i) {
        double x = std::ldexp(row[i], -scale);
        if (!std::isfinite(x) || std::abs(x) > 0x1p52 || std::trunc(x) != x) return -1;
        scaled[i] = static_cast<int64_t>(x);
    }
    Vec points[6];
    for (int i = 0; i < 6; ++i) for (int k = 0; k < 3; ++k) {
        int64_t x = scaled[3*i+k]-scaled[k];
        if (x < -(int64_t(1) << 27) || x > (int64_t(1) << 27)) return -1;
        points[i].x[k] = x;
    }
    // Differences <=2^28 make all degree-four cross/dot/SAT expressions <2^117.
    Vec* a = points;
    Vec* b = points+3;
    Vec shared[2];
    int count = 0;
    for (int i = 0; i < 3; ++i) {
        if (row[18+i] != 0. && row[18+i] != 1.) return -1;
        if (row[18+i] == 1.) {
            if (count == 2 || !contains(b, 3, a[i])) return -1;
            shared[count++] = a[i];
        }
    }
    Vec na = cross(sub(a[1], a[0]), sub(a[2], a[0]));
    Vec nb = cross(sub(b[1], b[0]), sub(b[2], b[0]));
    if (!any(na) || !any(nb)) return -1;
    if (count == 2 && !(shared[0] == shared[1])) {
        if (any(cross(na, nb))) return 0;
        Vec edge = sub(shared[1], shared[0]);
        int ia = 0, ib = 0;
        while (contains(shared, count, a[ia])) ++ia;
        while (contains(shared, count, b[ib])) ++ib;
        Vec sa = cross(edge, sub(a[ia], shared[0]));
        Vec sb = cross(edge, sub(b[ib], shared[0]));
        return dot(sa, sb) > 0;
    }
    Int distances[2][3];
    for (int i = 0; i < 3; ++i) {
        distances[0][i] = dot(nb, sub(a[i], b[0]));
        distances[1][i] = dot(na, sub(b[i], a[0]));
    }
    for (int t = 0; t < 2; ++t) {
        Int* d = distances[t];
        if (!d[0] && !d[1] && !d[2]) continue;
        if (std::min({d[0], d[1], d[2]}) >= 0 || std::max({d[0], d[1], d[2]}) <= 0) {
            bool allowed = true;
            for (int i = 0; i < 3; ++i)
                if (!d[i] && !on_allowed(points[3*t+i], shared, count)) allowed = false;
            if (allowed) return 0;
        }
    }
    if (!distances[1][0] && !distances[1][1] && !distances[1][2]) {
        for (int t = 0; t < 2; ++t) {
            Vec* triangle = t ? b : a;
            Vec* other = t ? a : b;
            Vec normal = t ? nb : na;
            for (int i = 0; i < 3; ++i) {
                Vec u = triangle[i], v = triangle[(i+1)%3];
                Vec inward = cross(normal, sub(v, u));
                Int signs[3];
                for (int j = 0; j < 3; ++j) signs[j] = dot(inward, sub(other[j], u));
                if (std::max({signs[0], signs[1], signs[2]}) > 0) continue;
                Vec zeros[3];
                int nz = 0;
                for (int j = 0; j < 3; ++j) if (!signs[j]) zeros[nz++] = other[j];
                if (!nz) return 0;
                int axis = 0;
                while (u.x[axis] == v.x[axis]) ++axis;
                Int zlo = zeros[0].x[axis], zhi = zlo;
                for (int j = 1; j < nz; ++j) {
                    zlo = std::min(zlo, zeros[j].x[axis]);
                    zhi = std::max(zhi, zeros[j].x[axis]);
                }
                Int lo = std::max(std::min(u.x[axis], v.x[axis]), zlo);
                Int hi = std::min(std::max(u.x[axis], v.x[axis]), zhi);
                if (lo > hi) return 0;
                Vec endpoints[5] = {u, v, {}, {}, {}};
                for (int j = 0; j < nz; ++j) endpoints[2+j] = zeros[j];
                for (int j = 0; j < nz+2; ++j)
                    if ((endpoints[j].x[axis] == lo || endpoints[j].x[axis] == hi)
                        && !on_allowed(endpoints[j], shared, count)) return 1;
                return 0;
            }
        }
        return 1;
    }
    // shortcut: rational clipping stays in Python; extend only if measured contacts justify it.
    return -1;
}
}  // namespace

extern "C" int wf_accel_abi() { return 1; }
extern "C" int wf_accel_kind() {
#ifdef WF_WITH_CUDA
    return 1;
#else
    return 0;
#endif
}
extern "C" int wf_cpu_filter(int mode, int n, const double* rows, int8_t* output) {
    if (mode < 0 || mode > 2 || n < 1 || n > WF_MAX_ROWS || !rows || !output) return -1;
    if (!wf_host_environment()) return -3;
    for (int i = 0; i < n; ++i)
        output[i] = mode == 2 ? contact(rows+WF_COLUMNS*i) : wf_interval_filter(mode, rows+WF_COLUMNS*i);
    return 0;
}
