#ifndef WINDING_FIELD_PREDICATES_H
#define WINDING_FIELD_PREDICATES_H

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define WF_HD __host__ __device__ inline
#else
#define WF_HD inline
#endif

constexpr int WF_MAX_ROWS = 65536;
constexpr int WF_COLUMNS = 24;
// Both nextafter arguments must be double: CUDA's mixed overload can be host-only.
constexpr double WF_INFINITY = double(INFINITY);

struct WFInterval { double lo, hi; };
WF_HD WFInterval wf_point(double x) { return {x, x}; }
WF_HD WFInterval wf_add(WFInterval a, WFInterval b) {
    return {nextafter(a.lo+b.lo, -WF_INFINITY), nextafter(a.hi+b.hi, WF_INFINITY)};
}
WF_HD WFInterval wf_sub(WFInterval a, WFInterval b) {
    return {nextafter(a.lo-b.hi, -WF_INFINITY), nextafter(a.hi-b.lo, WF_INFINITY)};
}
WF_HD WFInterval wf_mul(WFInterval a, WFInterval b) {
    double p = a.lo*b.lo, q = a.lo*b.hi, r = a.hi*b.lo, s = a.hi*b.hi;
    return {nextafter(fmin(fmin(p, q), fmin(r, s)), -WF_INFINITY),
            nextafter(fmax(fmax(p, q), fmax(r, s)), WF_INFINITY)};
}
WF_HD WFInterval wf_div_positive(WFInterval a, double b) {
    return {nextafter(a.lo/b, -WF_INFINITY), nextafter(a.hi/b, WF_INFINITY)};
}
WF_HD bool wf_finite(WFInterval a) {
    return a.lo > -WF_INFINITY && a.hi < WF_INFINITY && a.lo <= a.hi;
}
WF_HD bool wf_scalar(double x) { return x >= 1e-20 && x <= 1e20; }
WF_HD bool wf_envelope(const double* x, int count) {
    for (int i = 0; i < count; ++i)
        if (!(x[i] >= -1e20 && x[i] <= 1e20)) return false;
    return true;
}

WF_HD bool wf_numeric_probe() {
    volatile double one = 1., half_ulp = 0x1p-53, tiny = 0x0.0000000000001p-1022;
    volatile double normal = 0x1p-1022, two = 2.;
    volatile float ftiny = 0x1p-149f, fnormal = 0x1p-126f, ftwo = 2.f;
    volatile double half_normal = normal/two;
    volatile float fhalf_normal = fnormal/ftwo;
    return one+half_ulp == 1. && one-half_ulp == 0x1.fffffffffffffp-1
        && nextafter(one, WF_INFINITY) == 0x1.0000000000001p0
        && nextafter(one, -WF_INFINITY) == 0x1.fffffffffffffp-1
        && nextafter(0., WF_INFINITY) == tiny
        && nextafter(tiny, -WF_INFINITY) == 0.
        // Compare normal results: DAZ can make two different subnormals compare equal.
        && half_normal*two == normal && fhalf_normal*ftwo == fnormal
        && normal/two == 0x0.8p-1022 && tiny*two == 0x0.0000000000002p-1022
        && tiny+tiny == 0x0.0000000000002p-1022
        && fnormal/ftwo == 0x1p-127f && ftiny*ftwo == 0x1p-148f;
}

WF_HD int8_t wf_coverage(const double* x) {
    if (!wf_envelope(x, 16) || !wf_scalar(x[6]) || !wf_scalar(x[15])
        || x[13] < 0. || x[13] > x[14]) return -1;
    WFInterval distance2 = wf_point(0.);
    for (int k = 0; k < 3; ++k) {
        if (x[7+k] > x[10+k]) return -1;
        WFInterval p = wf_add(wf_mul(wf_point(x[k]), wf_point(x[6])), wf_point(x[3+k]));
        WFInterval delta = wf_sub({x[7+k], x[10+k]}, p);
        distance2 = wf_add(distance2, wf_mul(delta, delta));
    }
    // Squared distances are nonnegative even when interval dependency straddles zero.
    distance2.lo = fmax(0., distance2.lo);
    WFInterval half2 = {x[13], x[14]};
    WFInterval r = wf_sub(wf_sub(wf_mul(wf_point(x[15]), wf_point(x[15])), distance2), half2);
    WFInterval test = wf_sub(wf_mul(r, r), wf_mul(wf_point(4.), wf_mul(distance2, half2)));
    if (!wf_finite(r) || !wf_finite(test)) return -1;
    if (r.lo >= 0. && test.lo >= 0.) return 1;
    if (r.hi < 0. || test.hi < 0.) return 0;
    return -1;
}

WF_HD int8_t wf_orientation(const double* x) {
    if (!wf_envelope(x, 21) || !wf_scalar(x[9])) return -1;
    for (int k = 10; k < 13; ++k) if (!wf_scalar(x[k])) return -1;
    WFInterval a[3], b[3], normal[3];
    for (int k = 0; k < 3; ++k) {
        a[k] = wf_mul(wf_sub(wf_point(x[3+k]), wf_point(x[k])), wf_point(x[9]));
        b[k] = wf_mul(wf_sub(wf_point(x[6+k]), wf_point(x[k])), wf_point(x[9]));
    }
    for (int k = 0; k < 3; ++k) {
        int u = (k+1)%3, v = (k+2)%3;
        normal[k] = wf_sub(wf_mul(a[u], b[v]), wf_mul(a[v], b[u]));
    }
    WFInterval lower = wf_point(0.);
    for (int axis = 0; axis < 3; ++axis) {
        int bit = 1 << axis;  // XYZ axes are the low-to-high bits of ZYX corners.
        WFInterval smallest = {WF_INFINITY, WF_INFINITY};
        for (int j = 0; j < 8; ++j) if (!(j & bit)) {
            WFInterval grad = wf_div_positive(wf_sub(wf_point(x[13+j+bit]), wf_point(x[13+j])), x[10+axis]);
            WFInterval term = wf_mul(grad, normal[axis]);
            smallest.lo = fmin(smallest.lo, term.lo);
            smallest.hi = fmin(smallest.hi, term.hi);
        }
        lower = wf_add(lower, smallest);
    }
    if (!wf_finite(lower)) return -1;
    if (lower.lo > 0.) return 1;
    if (lower.hi <= 0.) return 0;
    return -1;
}

WF_HD int8_t wf_interval_filter(int mode, const double* row) {
    return mode == 0 ? wf_coverage(row) : wf_orientation(row);
}

bool wf_host_environment();

extern "C" {
int wf_accel_abi();
int wf_accel_kind();
int wf_cpu_filter(int mode, int n, const double* rows, int8_t* output);
int wf_cuda_init(int capacity, uint64_t reserve_bytes);
int wf_cuda_filter(int mode, int n, const double* rows, int8_t* output);
int wf_cuda_close();
int wf_cuda_memory(uint64_t* free_bytes, uint64_t* total_bytes);
}
#endif
