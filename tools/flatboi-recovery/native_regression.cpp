// Native regressions for the corrected PaStiX shim. Built by CMakeLists.txt as the
// `native_regression` target; exits non-zero on the first violated expectation.
//
//  1. A non-SPD coefficient set must raise instead of printing "Negative diagonal
//     term" and letting the solve publish garbage.
//  2. Two distinct sparse patterns with identical n and nnz must re-analyse: the
//     stale symbolic/value map used to scatter coefficients into wrong rows.
//  3. Answers must satisfy the original equations to the factor's tolerance.
#include <igl/pastix6_solver.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Sparse = igl::Pastix6LLT::SparseMatrix;

Sparse laplacian(int size, const std::vector<std::pair<int, int>>& extra) {
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < size; ++i) triplets.emplace_back(i, i, 4.0);
    for (int i = 0; i + 1 < size; ++i) {
        triplets.emplace_back(i, i + 1, -1.0);
        triplets.emplace_back(i + 1, i, -1.0);
    }
    for (const auto& [i, j] : extra) {
        triplets.emplace_back(i, j, -1.0);
        triplets.emplace_back(j, i, -1.0);
    }
    Sparse matrix(size, size);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return matrix;
}

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (condition) {
        std::cout << "ok   " << what << "\n";
    } else {
        std::cout << "FAIL " << what << "\n";
        ++failures;
    }
}

}  // namespace

int main() {
    // 1. Non-SPD coefficients are rejected before PaStiX can factorize them.
    {
        Sparse bad = laplacian(8, {{0, 7}});
        bad.coeffRef(3, 3) = -1.0;
        bad.makeCompressed();
        igl::Pastix6LLT solver;
        bool rejected = false;
        std::string message;
        try {
            solver.factorize(bad);
        } catch (const std::exception& error) {
            rejected = true;
            message = error.what();
        }
        expect(rejected && message.find("nonpositive diagonal") != std::string::npos,
               "non-SPD diagonal rejected with an explicit cause: " + message);
    }
    // 2. A different pattern with the same n/nnz forces a new symbolic analysis.
    {
        Sparse first = laplacian(12, {{0, 6}});
        Sparse second = laplacian(12, {{2, 9}});
        expect(first.rows() == second.rows() && first.nonZeros() == second.nonZeros(),
               "the two regression patterns share n and nnz");
        igl::Pastix6LLT solver;
        Eigen::VectorXd rhs(12);
        for (int i = 0; i < 12; ++i) rhs(i) = 1.0 + i;
        std::vector<double> errors;
        for (const Sparse* matrix : {&first, &second, &first}) {
            solver.factorize(*matrix);
            const Eigen::VectorXd x = solver.solve(rhs);
            errors.push_back(((*matrix) * x - rhs).norm() / rhs.norm());
        }
        expect(errors[0] < 1e-10 && errors[1] < 1e-10 && errors[2] < 1e-10,
               "each pattern solved on its own equations (relative residuals recorded)");
        std::cout << "     residuals " << errors[0] << " " << errors[1] << " " << errors[2] << "\n";
    }
    // 3. Repeated factorization of the same pattern keeps the original equations.
    {
        Sparse matrix = laplacian(40, {{0, 20}, {5, 33}});
        Eigen::VectorXd rhs = Eigen::VectorXd::Ones(40);
        igl::Pastix6LLT solver;
        solver.factorize(matrix);
        const Eigen::VectorXd first = solver.solve(rhs);
        solver.factorize(matrix);
        const Eigen::VectorXd second = solver.solve(rhs);
        expect((first - second).norm() <= 1e-12 * first.norm(),
               "repeated factorization of one pattern is stable");
        expect((matrix * second - rhs).norm() / rhs.norm() < 1e-10,
               "solved vector satisfies the original equations");
    }
    // 4. A badly scaled but exactly SPD system must still be solved accurately:
    //    its diagonal spans 1e18..1e-18 while the equilibrated operator is the
    //    original unit-scale Laplacian, so symmetric equilibration is what makes
    //    the pivots meaningful.
    {
        Sparse matrix = laplacian(60, {{0, 30}, {7, 44}, {12, 51}});
        Eigen::VectorXd weights(60);
        for (int i = 0; i < 60; ++i) weights(i) = std::pow(10.0, -18.0 + 36.0 * i / 59.0);
        Eigen::VectorXd root = weights.array().sqrt();
        Eigen::VectorXd inverse_root = weights.array().rsqrt();
        Sparse diagonal(60, 60);
        diagonal.setIdentity();
        for (int i = 0; i < 60; ++i) diagonal.coeffRef(i, i) = root(i);
        Sparse scaled = (diagonal * matrix * diagonal).eval();
        scaled.makeCompressed();
        Eigen::VectorXd rhs(60);
        for (int i = 0; i < 60; ++i) rhs(i) = 1.0 + 0.01 * i;
        igl::Pastix6LLT solver;
        solver.factorize(scaled);
        const Eigen::VectorXd solution = solver.solve(rhs);
        Eigen::MatrixXd dense = Eigen::MatrixXd(matrix);
        const Eigen::VectorXd expected = inverse_root.cwiseProduct(
            Eigen::VectorXd(dense.partialPivLu().solve(inverse_root.cwiseProduct(rhs))));
        const double relative = (solution - expected).norm() / expected.norm();
        const double residual = (scaled * solution - rhs).norm();
        const double backward = residual /
            (scaled.norm() * solution.norm() + rhs.norm() + 1.0);
        expect(backward < 1e-14 && relative < 1e-8,
               "equilibrated solve keeps a 1e18 diagonal range accurate");
        std::cout << "     backward error " << backward << " relative error " << relative << "\n";
    }
    std::cout << (failures ? "NATIVE REGRESSION FAILED\n" : "NATIVE REGRESSION PASSED\n");
    return failures ? 1 : 0;
}
