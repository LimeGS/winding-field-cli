#!/usr/bin/env python3
"""Apply the Flatboi recovery patch to a villa checkout at the recorded revision.

Every edit is an exact-string replacement. A missing anchor aborts the run, so a
different upstream revision can never be silently patched into something else.
Usage: apply_patch.py <villa-checkout>
"""
import sys
from pathlib import Path

INCLUDES_BEFORE = '''#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
'''
INCLUDES_AFTER = '''#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
'''

HELPERS_BEFORE = '''namespace igl {

class Pastix6LLT {
'''
HELPERS_AFTER = '''namespace igl {

// PaStiX stage failures used to be ignored: a failed numeric factorization left the
// factor holding garbage that the solve then published. Every stage is checked.
inline void pastix_require(int code, const char* stage) {
    if (code == 0) return;
    std::ostringstream message;
    message << "PaStiX " << stage << " failed with code " << code;
    throw std::runtime_error(message.str());
}

// FNV-1a over the complete CSC/CSR pattern. nnz alone is not a pattern identity:
// two different graphs can share a sparsity count, and a stale value map then
// scatters coefficients into the wrong rows.
inline std::uint64_t pastix_pattern_hash(const pastix_int_t* colptr, pastix_int_t n,
                                         const pastix_int_t* rowptr, pastix_int_t nnz) {
    std::uint64_t hash = 1469598103934665603ull;
    auto mix = [&hash](const void* data, std::size_t bytes) {
        const auto* bytes_ptr = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < bytes; ++i) {
            hash ^= bytes_ptr[i];
            hash *= 1099511628211ull;
        }
    };
    mix(colptr, sizeof(pastix_int_t) * static_cast<std::size_t>(n + 1));
    mix(rowptr, sizeof(pastix_int_t) * static_cast<std::size_t>(nnz));
    return hash;
}

// Lossless dump of the exact system that failed: dimensions, endianness marker, CSC
// pattern, values and (when available) the right-hand side. Reconstructible offline.
inline void pastix_dump_system(const char* directory, const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& L,
                               const Eigen::VectorXd* rhs, const char* label, int iteration) {
    if (!directory || !*directory) return;
    std::ostringstream name;
    name << directory << "/pastix-" << label << "-" << iteration << ".bin";
    std::FILE* out = std::fopen(name.str().c_str(), "wb");
    if (!out) {
        std::cerr << "[pastix] cannot write " << name.str() << std::endl;
        return;
    }
    const std::uint64_t n = static_cast<std::uint64_t>(L.rows());
    const std::uint64_t nnz = static_cast<std::uint64_t>(L.nonZeros());
    const std::uint32_t endian = 0x01020304u;
    std::fwrite("FBPASTIX", 1, 8, out);
    std::fwrite(&endian, sizeof(endian), 1, out);
    std::fwrite(&n, sizeof(n), 1, out);
    std::fwrite(&nnz, sizeof(nnz), 1, out);
    for (Eigen::Index j = 0; j < L.outerSize() + 1; ++j) {
        const std::uint64_t start = j < L.outerSize()
            ? static_cast<std::uint64_t>(L.outerIndexPtr()[j]) : nnz;
        std::fwrite(&start, sizeof(start), 1, out);
    }
    for (Eigen::Index j = 0; j < L.outerSize(); ++j) {
        for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator it(L, j); it; ++it) {
            const std::uint64_t row = static_cast<std::uint64_t>(it.index());
            std::fwrite(&row, sizeof(row), 1, out);
        }
    }
    std::fwrite(L.valuePtr(), sizeof(double), static_cast<std::size_t>(nnz), out);
    const std::uint64_t rhs_size = rhs ? static_cast<std::uint64_t>(rhs->size()) : 0;
    std::fwrite(&rhs_size, sizeof(rhs_size), 1, out);
    if (rhs_size) std::fwrite(rhs->data(), sizeof(double), static_cast<std::size_t>(rhs_size), out);
    std::fclose(out);
    std::cerr << "[pastix] dumped " << name.str() << " (" << label << ")" << std::endl;
}

class Pastix6LLT {
'''

MEMBERS_BEFORE = '''    std::vector<int> bindtab_;
'''
MEMBERS_AFTER = '''    std::vector<int> bindtab_;
    std::uint64_t   pattern_hash_ = 0;
    pastix_int_t    pattern_n_ = 0;
    pastix_int_t    pattern_nnz_ = 0;
    const char*     dump_dir_ = std::getenv("FLATBOI_DIAG_DIR");
    int             iteration_ = 0;
    // Symmetric equilibration of the assembled system: (D L D) y = D b, x = D y.
    // Same equations, same solution; it only stops a 1e18 coefficient range from
    // destroying the Cholesky pivots.
    std::vector<double> equilibration_;

    void check_coefficients(const SparseMatrix& L) const {
        if (!L.valuePtr()) throw std::runtime_error("PaStiX: empty coefficient array");
        const auto nnz = static_cast<std::size_t>(L.nonZeros());
        for (std::size_t k = 0; k < nnz; ++k) {
            if (!std::isfinite(L.valuePtr()[k]))
                throw std::runtime_error("PaStiX: non-finite coefficient at " + std::to_string(k));
        }
        double low = 0.0, high = 0.0;
        for (Eigen::Index j = 0; j < L.outerSize(); ++j) {
            const double value = L.coeff(j, j);
            if (!std::isfinite(value))
                throw std::runtime_error("PaStiX: non-finite diagonal at row " + std::to_string(j));
            if (j == 0) { low = high = value; }
            else { low = std::min(low, value); high = std::max(high, value); }
            if (!(value > 0.0))
                throw std::runtime_error("PaStiX: nonpositive diagonal " + std::to_string(value) +
                                         " at row " + std::to_string(j));
        }
        std::cerr << "[pastix] diagonal range " << low << " .. " << high << std::endl;
    }
'''

ANALYZE_TAIL_BEFORE = '''        pastix_task_analyze(data_, &spm_);
        analyzed_ = true;'''
ANALYZE_TAIL_AFTER = '''        // The identity is taken from the caller's matrix before PaStiX may reorder or
        // rewrite its internal copy, so a later call compares like with like.
        {
            const SparseMatrix lower = lower_triangle(L);
            const pastix_int_t rows_n = static_cast<pastix_int_t>(lower.rows());
            const pastix_int_t nnz = static_cast<pastix_int_t>(lower.nonZeros());
            std::vector<pastix_int_t> colptr(static_cast<std::size_t>(rows_n + 1)), rowptr(
                static_cast<std::size_t>(nnz));
            pastix_int_t out = 0;
            colptr[0] = 0;
            for (pastix_int_t j = 0; j < rows_n; ++j) {
                for (SparseMatrix::InnerIterator it(lower, j); it; ++it) {
                    rowptr[out++] = static_cast<pastix_int_t>(it.index());
                }
                colptr[j + 1] = out;
            }
            pattern_hash_ = pastix_pattern_hash(colptr.data(), rows_n, rowptr.data(), nnz);
            pattern_n_ = rows_n;
            pattern_nnz_ = nnz;
        }
        pastix_require(pastix_task_analyze(data_, &spm_), "analyze");
        analyzed_ = true;'''

FACTORIZE_BEFORE = '''    void factorize(const SparseMatrix& L) {
        if (!analyzed_) {
            throw std::runtime_error("Pastix6LLT: factorize() before analyzePattern()");
        }
        // Sparsity is fixed across SLIM iters — just refresh the lower-tri
        // values using the index map captured in analyzePattern().
        const double* src = L.valuePtr();
        double* dst = static_cast<double*>(spm_.values);
        const auto n = value_src_idx_.size();
        for (std::size_t k = 0; k < n; ++k) {
            dst[k] = src[value_src_idx_[k]];
        }
        pastix_task_numfact(data_, &spm_);
    }'''
FACTORIZE_AFTER = '''    void factorize(const SparseMatrix& L) {
        ++iteration_;
        try {
            check_coefficients(L);
        } catch (const std::exception&) {
            pastix_dump_system(dump_dir_, L, nullptr, "coefficient", iteration_);
            throw;
        }
        // The symbolic factorization is valid only for the pattern it analysed. A
        // changed pattern with the same nnz used to reuse the stale value map.
        const SparseMatrix lower = lower_triangle(L);
        const pastix_int_t nnz = static_cast<pastix_int_t>(lower.nonZeros());
        const pastix_int_t n = static_cast<pastix_int_t>(lower.rows());
        std::uint64_t hash = 0;
        {
            std::vector<pastix_int_t> colptr(static_cast<std::size_t>(n + 1)), rowptr(
                static_cast<std::size_t>(nnz));
            pastix_int_t out = 0;
            colptr[0] = 0;
            for (pastix_int_t j = 0; j < n; ++j) {
                for (SparseMatrix::InnerIterator it(lower, j); it; ++it) {
                    rowptr[out++] = static_cast<pastix_int_t>(it.index());
                }
                colptr[j + 1] = out;
            }
            hash = pastix_pattern_hash(colptr.data(), n, rowptr.data(), nnz);
        }
        if (!analyzed_ || hash != pattern_hash_ || n != pattern_n_ || nnz != pattern_nnz_) {
            if (analyzed_)
                std::cerr << "[pastix] re-analyzing a changed sparsity pattern" << std::endl;
            analyzePattern(L);
        }
        // Refresh the lower-tri values through the analysed index map, symmetrically
        // equilibrated so the factor sees a unit diagonal.
        equilibration_.resize(static_cast<std::size_t>(n));
        for (pastix_int_t j = 0; j < n; ++j) {
            equilibration_[static_cast<std::size_t>(j)] =
                1.0 / std::sqrt(L.coeff(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(j)));
        }
        const double* src = L.valuePtr();
        double* dst = static_cast<double*>(spm_.values);
        const pastix_int_t* rows = static_cast<const pastix_int_t*>(spm_.rowptr);
        const pastix_int_t* cols = static_cast<const pastix_int_t*>(spm_.colptr);
        const auto count = value_src_idx_.size();
        pastix_int_t column = 0;
        for (std::size_t k = 0; k < count; ++k) {
            while (column + 1 < n && cols[column + 1] <= static_cast<pastix_int_t>(k)) ++column;
            const double scale = equilibration_[static_cast<std::size_t>(rows[k])] *
                                 equilibration_[static_cast<std::size_t>(column)];
            dst[k] = src[value_src_idx_[k]] * scale;
        }
        if (std::getenv("FLATBOI_DIAG_ALL"))
            pastix_dump_system(dump_dir_, L, nullptr, "iter", iteration_);
        const int code = pastix_task_numfact(data_, &spm_);
        if (code != 0) {
            pastix_dump_system(dump_dir_, L, nullptr, "numfact", iteration_);
            pastix_require(code, "numeric factorization");
        }
    }'''

SOLVE_BEFORE = '''        VectorXd x = rhs;
        const pastix_int_t nrhs = static_cast<pastix_int_t>(rhs.size() / n);
        // pastix_task_solve writes the answer in place over the rhs buffer.
        pastix_task_solve(data_, n, nrhs, x.data(), n);
        return x;'''
SOLVE_AFTER = '''        VectorXd x = rhs;
        const pastix_int_t nrhs = static_cast<pastix_int_t>(rhs.size() / n);
        if (equilibration_.size() != static_cast<std::size_t>(n))
            throw std::runtime_error("Pastix6LLT: solve() before factorize()");
        // (D L D) y = D b, then x = D y returns the solution of the original system.
        for (pastix_int_t r = 0; r < nrhs; ++r)
            for (pastix_int_t i = 0; i < n; ++i)
                x(static_cast<Eigen::Index>(r * n + i)) *= equilibration_[static_cast<std::size_t>(i)];
        // pastix_task_solve writes the answer in place over the rhs buffer.
        pastix_require(pastix_task_solve(data_, n, nrhs, x.data(), n), "solve");
        for (pastix_int_t r = 0; r < nrhs; ++r)
            for (pastix_int_t i = 0; i < n; ++i)
                x(static_cast<Eigen::Index>(r * n + i)) *= equilibration_[static_cast<std::size_t>(i)];
        if (!x.allFinite()) throw std::runtime_error("PaStiX: solve produced non-finite values");
        return x;'''

LOWER_HELPER_BEFORE = '''    void setThreads(int n) { iparm_[IPARM_THREAD_NBR] = n; }
'''
LOWER_HELPER_AFTER = '''    void setThreads(int n) { iparm_[IPARM_THREAD_NBR] = n; }

    // PaStiX SpmSymmetric expects exactly one triangle, and so does the pattern hash.
    static SparseMatrix lower_triangle(const SparseMatrix& L) {
        SparseMatrix lower = L.triangularView<Eigen::Lower>();
        lower.makeCompressed();
        return lower;
    }
'''

SLIM_INCLUDES_BEFORE = '''#include <unordered_map>
'''
SLIM_INCLUDES_AFTER = '''#include <algorithm>
#include <atomic>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
'''

SLIM_BEFORE = '''      struct PastixCache {
        igl::Pastix6LLT solver;
        Eigen::Index rows = -1;
        Eigen::Index cols = -1;
        Eigen::Index nnz  = -1;
      };
      static std::unordered_map<const igl::SLIMData*, PastixCache> cache;
      PastixCache& entry = cache[&s];
      const Eigen::Index L_nnz = L.nonZeros();
      if (entry.rows != L.rows() || entry.cols != L.cols() || entry.nnz != L_nnz) {
        entry.solver.analyzePattern(L);
        entry.rows = L.rows();
        entry.cols = L.cols();
        entry.nnz  = L_nnz;
      }
      entry.solver.factorize(L);
      Uc = entry.solver.solve(s.rhs);'''
SLIM_AFTER = '''      // The solver owns the pattern identity now: factorize() re-analyses whenever the
      // full sparsity pattern changes, not merely when the nnz count does.
      static std::unordered_map<const igl::SLIMData*, std::unique_ptr<igl::Pastix6LLT>> cache;
      std::unique_ptr<igl::Pastix6LLT>& entry = cache[&s];
      if (!entry) entry = std::make_unique<igl::Pastix6LLT>();
      entry->factorize(L);
      Uc = entry->solve(s.rhs);
      // Backward error of the original equations, not the factor's internal estimate.
      // A plain relative residual is meaningless when the operator spans ~1e18; the
      // scale-invariant form is ||Lx-b|| / (||L|| ||x|| + ||b||).
      const double backward = (L * Uc - s.rhs).norm() /
          (L.norm() * Uc.norm() + s.rhs.norm() + 1.0);
      std::cerr << "[slim] backward error " << backward << std::endl;
      if (!std::isfinite(backward) || backward > 1e-10) {
        static std::atomic<int> failures{0};
        const int count = failures.fetch_add(1);
        igl::pastix_dump_system(std::getenv("FLATBOI_DIAG_DIR"), L, &s.rhs, "residual", count);
        std::ostringstream message;
        message << "SLIM linear solve backward error " << backward << " exceeds 1e-10";
        throw std::runtime_error(message.str());
      }'''

FLATBOI_CHECK_BEFORE = '''    // Persist
    save_energies(obj_path, energies);
    fb.save_obj(uv_out);
'''
FLATBOI_CHECK_AFTER = '''    // A published UV must be a valid map: reject a final step with zero/inverted faces
    // before it reaches the energies file or the checkpoint.
    {
      int invalid = 0;
      double worst = 0.0;
      for (int fi = 0; fi < fb.F.rows(); ++fi) {
        const Eigen::RowVector2d a = uv_out.row(fb.F(fi, 1)) - uv_out.row(fb.F(fi, 0));
        const Eigen::RowVector2d b = uv_out.row(fb.F(fi, 2)) - uv_out.row(fb.F(fi, 0));
        const double area = a.x() * b.y() - a.y() * b.x();
        if (!(area > 0.0)) { ++invalid; worst = std::min(worst, area); }
      }
      if (invalid) {
        std::cerr << "Error: final UV has " << invalid
                  << " nonpositive faces (worst signed area " << worst
                  << "); refusing to publish an invalid map.\\n";
        return 4;
      }
    }

    // Persist
    save_energies(obj_path, energies);
    fb.save_obj(uv_out);
'''


def replace(text, before, after, name):
    if text.count(before) != 1:
        raise SystemExit(f'anchor not found exactly once in {name}: {before.splitlines()[0]}')
    return text.replace(before, after)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1]) / 'volume-cartographer/libs'
    solver = root / 'libigl_changes/include/igl/pastix6_solver.h'
    slim = root / 'libigl_changes/include/igl/slim.cpp'
    flatboi = root / 'flatboi/flatboi.cpp'
    text = solver.read_text()
    text = replace(text, INCLUDES_BEFORE, INCLUDES_AFTER, solver.name)
    text = replace(text, HELPERS_BEFORE, HELPERS_AFTER, solver.name)
    text = replace(text, MEMBERS_BEFORE, MEMBERS_AFTER, solver.name)
    text = replace(text, LOWER_HELPER_BEFORE, LOWER_HELPER_AFTER, solver.name)
    text = replace(text, ANALYZE_TAIL_BEFORE, ANALYZE_TAIL_AFTER, solver.name)
    text = replace(text, FACTORIZE_BEFORE, FACTORIZE_AFTER, solver.name)
    text = replace(text, SOLVE_BEFORE, SOLVE_AFTER, solver.name)
    solver.write_text(text)
    text = slim.read_text()
    text = replace(text, SLIM_INCLUDES_BEFORE, SLIM_INCLUDES_AFTER, slim.name)
    text = replace(text, SLIM_BEFORE, SLIM_AFTER, slim.name)
    slim.write_text(text)
    text = flatboi.read_text()
    text = replace(text, FLATBOI_CHECK_BEFORE, FLATBOI_CHECK_AFTER, flatboi.name)
    flatboi.write_text(text)
    print('patched', solver, slim, flatboi)


if __name__ == '__main__':
    main()
