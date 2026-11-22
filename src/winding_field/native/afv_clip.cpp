// Fused, exact float64 AFV support clipping (ABI 1).
//
// Mirrors winding_field.afv_clip.pieces() operation by operation: the same IEEE
// float64 operations in the same order, no fused multiply-add (build with
// -ffp-contract=off and without fast-math), the same NumPy/Python min/max/clip
// tie rules. Wherever the reference outcome could depend on an implementation-
// defined choice (a -0.0 clip input), or the reference itself switches to its
// streaming scalar path (interval budget, times > 1), the block is returned with
// status REFERENCE and the caller runs the Python reference for that block.
//
// Inputs: packed XYZ points (local grid coordinates), block offsets, a dense
// read-only Boolean support volume in ZYX order. Outputs: packed XYZ vertices,
// fragment vertex offsets and per-block fragment counts/status. No allocation
// escapes a call; scratch is bounded by the interval budget per block.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {

constexpr int32_t OK = 0;
constexpr int32_t REFERENCE = 1;
constexpr int32_t CAPACITY = 2;

struct Vec3 {
    double v[3];
};

// numpy.clip(x, 0, high) for finite x, high >= 0. Returns false when x is -0.0:
// the sign NumPy keeps for that tie is an implementation detail.
inline bool clip(double x, double high, double* out) {
    if (x == 0.0 && std::signbit(x)) return false;
    if (x < 0.0) {
        *out = 0.0;
    } else if (x > high) {
        *out = high;
    } else {
        *out = x;
    }
    return true;
}

// numpy.minimum / numpy.maximum for finite or infinite non-NaN operands. Equal
// operands of opposite sign only arise as signed zeros; callers make the result
// irrelevant (see enter/leave below).
inline double np_min(double a, double b) { return (a <= b) ? a : b; }
inline double np_max(double a, double b) { return (a >= b) ? a : b; }

struct Edge {
    bool none;
    double start[3];
    double end[3];
};

struct Output {
    double* vertices;
    int64_t vertex_capacity;
    int64_t vertices_used;
    int64_t* fragment_offsets;  // fragment_offsets[0] == 0 is written by the caller
    int64_t fragment_capacity;
    int64_t fragments_used;
};

class Piece {
   public:
    std::vector<Vec3> rows;
    void clear() { rows.clear(); }
};

// Append a finished fragment; false when the output buffers are full.
inline bool emit(const Piece& piece, Output* out, int64_t* block_fragments) {
    const int64_t n = static_cast<int64_t>(piece.rows.size());
    if (out->fragments_used + 1 > out->fragment_capacity ||
        out->vertices_used + n > out->vertex_capacity) {
        return false;
    }
    std::memcpy(out->vertices + 3 * out->vertices_used, piece.rows.data(),
                sizeof(double) * 3 * static_cast<size_t>(n));
    out->vertices_used += n;
    out->fragments_used += 1;
    out->fragment_offsets[out->fragments_used] = out->vertices_used;
    *block_fragments += 1;
    return true;
}

// One block. Returns OK, REFERENCE or CAPACITY; on anything but OK the caller
// rolls the output back to the block's start.
int32_t clip_block(const double* points, int64_t n, const uint8_t* support, const int64_t shape[3],
                   int64_t max_intervals, Output* out, int64_t* block_fragments) {
    // maps.shape is ZYX; the reference clips XYZ points against shape[::-1] - 1.
    const double high[3] = {static_cast<double>(shape[2] - 1), static_cast<double>(shape[1] - 1),
                            static_cast<double>(shape[0] - 1)};
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<Edge> edges;
    edges.reserve(n > 1 ? static_cast<size_t>(n - 1) : 0);
    int64_t count = 0;
    for (int64_t i = 0; i + 1 < n; ++i) {
        const double* a = points + 3 * i;
        const double* b = points + 3 * (i + 1);
        double delta[3], mn[3], mx[3];
        bool any = false, outside = false;
        for (int k = 0; k < 3; ++k) {
            delta[k] = b[k] - a[k];
            const bool moving = delta[k] != 0.0;
            any = any || moving;
            const double near = moving ? (-a[k]) / delta[k] : -inf;
            const double far = moving ? (high[k] - a[k]) / delta[k] : inf;
            mn[k] = np_min(near, far);
            mx[k] = np_max(near, far);
            if (!moving && (a[k] < 0.0 || a[k] > high[k])) outside = true;
        }
        // ndarray.max()/.min() over three values, first-wins on ties (a signed-zero
        // tie only changes M or m when they are zero, where it cannot matter).
        double M = mn[0];
        double m = mx[0];
        for (int k = 1; k < 3; ++k) {
            M = np_max(M, mn[k]);
            m = np_min(m, mx[k]);
        }
        // Python max(0.0, M) keeps 0.0 unless M > 0.0; min(1.0, m) keeps 1.0 unless m < 1.0.
        // A zero M yields enter 0.0; a zero m yields leave <= enter: the edge is dropped.
        const double enter = (M > 0.0) ? M : 0.0;
        const double leave = (m < 1.0) ? m : 1.0;
        Edge edge;
        edge.none = (leave <= enter) || outside || !any;
        if (edge.none) {
            edges.push_back(edge);
            continue;
        }
        for (int k = 0; k < 3; ++k) {
            const double s = a[k] + enter * delta[k];
            const double e = a[k] + leave * delta[k];
            if (!clip(s, high[k], &edge.start[k]) || !clip(e, high[k], &edge.end[k])) {
                return REFERENCE;
            }
        }
        int64_t crossings = 1;
        for (int k = 0; k < 3; ++k) {
            const double x = edge.start[k], y = edge.end[k];
            const double hi = (y > x) ? y : x;  // Python max(x, y)
            const double lo = (y < x) ? y : x;  // Python min(x, y)
            const int64_t c = static_cast<int64_t>(std::ceil(hi)) -
                              static_cast<int64_t>(std::floor(lo)) - 1;
            crossings += (c > 0) ? c : 0;
        }
        count += crossings;
        if (count > max_intervals) return REFERENCE;
        edges.push_back(edge);
    }

    Piece piece;
    std::vector<double> times;
    std::vector<Vec3> left, right;
    std::vector<uint8_t> supported;
    const int64_t stride_y = shape[2];
    const int64_t stride_z = shape[1] * shape[2];
    for (const Edge& edge : edges) {
        if (edge.none) {
            if (piece.rows.size() > 1 && !emit(piece, out, block_fragments)) return CAPACITY;
            piece.clear();
            continue;
        }
        double d[3];
        for (int k = 0; k < 3; ++k) d[k] = edge.end[k] - edge.start[k];
        times.clear();
        for (int k = 0; k < 3; ++k) {
            if (d[k] == 0.0) continue;
            const double s = edge.start[k], e = edge.end[k];
            const double lo = (e < s) ? e : s;
            const double hi = (e > s) ? e : s;
            const int64_t first = static_cast<int64_t>(std::floor(lo)) + 1;
            const int64_t stop = static_cast<int64_t>(std::ceil(hi));
            for (int64_t v = first; v < stop; ++v) {
                times.push_back((static_cast<double>(v) - s) / d[k]);
            }
        }
        times.push_back(1.0);
        std::sort(times.begin(), times.end());
        // numpy.unique then times[times > 0]; equal values are bit-identical except
        // signed zeros, which the > 0 filter removes.
        size_t kept = 0;
        for (size_t i = 0; i < times.size(); ++i) {
            if (kept && times[i] == times[kept - 1]) continue;
            times[kept++] = times[i];
        }
        times.resize(kept);
        times.erase(std::remove_if(times.begin(), times.end(), [](double t) { return !(t > 0.0); }),
                    times.end());
        for (double t : times) {
            if (t > 1.0) return REFERENCE;  // the reference streams these exactly
        }
        const size_t segments = times.size();
        left.resize(segments);
        right.resize(segments);
        supported.assign(segments, 1);
        for (size_t i = 0; i < segments; ++i) {
            const double previous = i ? times[i - 1] : 0.0;
            for (int k = 0; k < 3; ++k) {
                if (!clip(edge.start[k] + previous * d[k], high[k], &left[i].v[k]) ||
                    !clip(edge.start[k] + times[i] * d[k], high[k], &right[i].v[k])) {
                    return REFERENCE;
                }
            }
            // q = ((left + right) / 2)[::-1]: ZYX query of the segment midpoint.
            double q[3];
            for (int k = 0; k < 3; ++k) q[2 - k] = (left[i].v[k] + right[i].v[k]) / 2.0;
            int64_t low[3];
            double fraction[3];
            for (int j = 0; j < 3; ++j) {
                const double f = std::floor(q[j]);
                low[j] = static_cast<int64_t>(f);
                if (low[j] < 0 || low[j] > shape[j] - 1) return -1;  // internal invariant
                fraction[j] = q[j] - static_cast<double>(low[j]);
            }
            bool valid = true;
            for (int corner = 0; corner < 8 && valid; ++corner) {
                const int c[3] = {(corner >> 2) & 1, (corner >> 1) & 1, corner & 1};
                bool contributes = true;
                for (int j = 0; j < 3; ++j) {
                    const double w = c[j] ? fraction[j] : 1.0 - fraction[j];
                    if (!(w > 0.0)) contributes = false;
                }
                if (!contributes) continue;
                int64_t ij[3];
                for (int j = 0; j < 3; ++j) {
                    const int64_t value = low[j] + c[j];
                    ij[j] = (value < shape[j] - 1) ? value : shape[j] - 1;
                }
                if (!support[ij[0] * stride_z + ij[1] * stride_y + ij[2]]) valid = false;
            }
            supported[i] = valid ? 1 : 0;
        }
        // Runs of supported segments, as np.diff over [False, s, False].
        bool any_supported = false;
        size_t i = 0;
        while (i < segments) {
            if (!supported[i]) {
                ++i;
                continue;
            }
            any_supported = true;
            const size_t first = i;
            while (i < segments && supported[i]) ++i;
            const size_t last = i;  // exclusive
            if (!piece.rows.empty()) {
                bool close = first == 0;
                if (close) {
                    const Vec3& tail = piece.rows.back();
                    for (int k = 0; k < 3; ++k) {
                        // np.allclose(a, b, rtol=0, atol=1e-9): |a - b| <= 1e-9 + 0 * |b|
                        if (!(std::fabs(tail.v[k] - left[first].v[k]) <= 1e-9 + 0.0 * std::fabs(left[first].v[k]))) {
                            close = false;
                        }
                    }
                }
                if (!close) {
                    if (piece.rows.size() > 1 && !emit(piece, out, block_fragments)) return CAPACITY;
                    piece.clear();
                }
            }
            if (piece.rows.empty()) {
                piece.rows.push_back(left[first]);
                piece.rows.push_back(right[last - 1]);
            } else {
                piece.rows.push_back(right[last - 1]);
            }
            if (last < segments) {
                if (!emit(piece, out, block_fragments)) return CAPACITY;
                piece.clear();
            }
        }
        if (!any_supported && !piece.rows.empty()) {
            if (piece.rows.size() > 1 && !emit(piece, out, block_fragments)) return CAPACITY;
            piece.clear();
        }
    }
    if (piece.rows.size() > 1 && !emit(piece, out, block_fragments)) return CAPACITY;
    return OK;
}

}  // namespace

extern "C" {

int32_t afv_clip_abi(void) { return 1; }

// Returns 0 on success (per-block status in block_status), or a negative code
// for invalid arguments / violated invariants (never a partial success).
// Blocks after the first CAPACITY block are left with status CAPACITY.
int32_t afv_clip_blocks(const double* points, int64_t point_count, const int64_t* block_offsets,
                        int64_t block_count, const uint8_t* support, int64_t sz, int64_t sy,
                        int64_t sx, int64_t max_intervals, double* vertices,
                        int64_t vertex_capacity, int64_t* fragment_offsets,
                        int64_t fragment_capacity, int64_t* block_fragments, int32_t* block_status,
                        int64_t* totals) {
    if (!points || !block_offsets || !support || !vertices || !fragment_offsets ||
        !block_fragments || !block_status || !totals) {
        return -2;
    }
    if (point_count < 0 || block_count < 0 || sz < 1 || sy < 1 || sx < 1 || max_intervals < 1 ||
        vertex_capacity < 0 || fragment_capacity < 0) {
        return -3;
    }
    if (block_offsets[0] != 0 || block_offsets[block_count] != point_count) return -4;
    for (int64_t b = 0; b < block_count; ++b) {
        if (block_offsets[b + 1] < block_offsets[b]) return -4;
    }
    for (int64_t i = 0; i < 3 * point_count; ++i) {
        if (!std::isfinite(points[i])) return -5;
    }
    const int64_t shape[3] = {sz, sy, sx};
    Output out{vertices, vertex_capacity, 0, fragment_offsets, fragment_capacity, 0};
    fragment_offsets[0] = 0;
    bool full = false;
    try {
        for (int64_t b = 0; b < block_count; ++b) {
            block_fragments[b] = 0;
            if (full) {
                block_status[b] = CAPACITY;
                continue;
            }
            const int64_t vertices_before = out.vertices_used;
            const int64_t fragments_before = out.fragments_used;
            const int32_t status =
                clip_block(points + 3 * block_offsets[b], block_offsets[b + 1] - block_offsets[b],
                           support, shape, max_intervals, &out, &block_fragments[b]);
            if (status < 0) return status;
            if (status != OK) {
                // Discard this block's partial fragments; it is rerun elsewhere.
                out.vertices_used = vertices_before;
                out.fragments_used = fragments_before;
                block_fragments[b] = 0;
            }
            block_status[b] = status;
            if (status == CAPACITY) full = true;
        }
    } catch (const std::bad_alloc&) {
        return -6;
    }
    totals[0] = out.vertices_used;
    totals[1] = out.fragments_used;
    return 0;
}

}  // extern "C"
