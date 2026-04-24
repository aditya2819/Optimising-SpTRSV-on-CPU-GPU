#include "sptrsv_kernel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <mkl.h>
#include <mkl_spblas.h>

namespace {

template <typename Func>
double timeMs(Func&& func)
{
    const auto start = std::chrono::steady_clock::now();
    func();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double speedupOverMkl(double kernelMs, double mklMs)
{
    return (kernelMs > 0.0) ? (mklMs / kernelMs) : 0.0;
}

CsrMatrix generateLowerTriangular(int n, double density, unsigned seed)
{
    CsrMatrix matrix;
    matrix.n = n;
    matrix.rowPtr.resize(static_cast<std::size_t>(n + 1), 0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_real_distribution<double> val(0.1, 2.0);

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < row; ++col) {
            if (prob(rng) < density) {
                matrix.colIdx.push_back(col);
                matrix.values.push_back(val(rng));
            }
        }
        matrix.colIdx.push_back(row);
        matrix.values.push_back(static_cast<double>(n) + 1.0);
        matrix.rowPtr[static_cast<std::size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
    }

    matrix.nnz = static_cast<int>(matrix.colIdx.size());
    return matrix;
}

CsrMatrix generateBandedLowerTriangular(int n, int bandwidth)
{
    CsrMatrix matrix;
    matrix.n = n;
    matrix.rowPtr.resize(static_cast<std::size_t>(n + 1), 0);

    for (int row = 0; row < n; ++row) {
        const int startCol = std::max(0, row - bandwidth);
        for (int col = startCol; col < row; ++col) {
            matrix.colIdx.push_back(col);
            matrix.values.push_back(0.25 + 0.01 * static_cast<double>((row - col) % 13));
        }
        matrix.colIdx.push_back(row);
        matrix.values.push_back(static_cast<double>(bandwidth) + 2.0);
        matrix.rowPtr[static_cast<std::size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
    }

    matrix.nnz = static_cast<int>(matrix.colIdx.size());
    return matrix;
}

CsrMatrix generateChainLowerTriangular(int n)
{
    CsrMatrix matrix;
    matrix.n = n;
    matrix.rowPtr.resize(static_cast<std::size_t>(n + 1), 0);

    for (int row = 0; row < n; ++row) {
        if (row > 0) {
            matrix.colIdx.push_back(row - 1);
            matrix.values.push_back(0.5);
        }
        matrix.colIdx.push_back(row);
        matrix.values.push_back(2.0);
        matrix.rowPtr[static_cast<std::size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
    }

    matrix.nnz = static_cast<int>(matrix.colIdx.size());
    return matrix;
}

CsrMatrix generateWideLevelLowerTriangular(int n, int prefixWidth)
{
    CsrMatrix matrix;
    matrix.n = n;
    matrix.rowPtr.resize(static_cast<std::size_t>(n + 1), 0);

    const int frontier = std::min(prefixWidth, n);
    for (int row = 0; row < n; ++row) {
        if (row < frontier) {
            for (int col = 0; col < row; ++col) {
                matrix.colIdx.push_back(col);
                matrix.values.push_back(0.2);
            }
        } else {
            for (int col = 0; col < frontier; ++col) {
                matrix.colIdx.push_back(col);
                matrix.values.push_back(0.2 + 0.01 * static_cast<double>(col % 7));
            }
        }
        matrix.colIdx.push_back(row);
        matrix.values.push_back(static_cast<double>(frontier) + 3.0);
        matrix.rowPtr[static_cast<std::size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
    }

    matrix.nnz = static_cast<int>(matrix.colIdx.size());
    return matrix;
}

CsrMatrix generateBlockLowerTriangular(int n, int blockSize, double density, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> probabilityDist(0.0, 1.0);

    CsrMatrix matrix;
    matrix.n = n;
    matrix.rowPtr.resize(static_cast<std::size_t>(n + 1), 0);

    for (int row = 0; row < n; ++row) {
        const int rowBlock = row / blockSize;
        for (int col = 0; col < row; ++col) {
            const int colBlock = col / blockSize;
            const double threshold = (colBlock == rowBlock) ? 0.65 : density;
            if (probabilityDist(rng) < threshold) {
                matrix.colIdx.push_back(col);
                matrix.values.push_back(0.15 + 0.02 * static_cast<double>((row + col) % 17));
            }
        }
        matrix.colIdx.push_back(row);
        matrix.values.push_back(static_cast<double>(blockSize) + 4.0);
        matrix.rowPtr[static_cast<std::size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
    }

    matrix.nnz = static_cast<int>(matrix.colIdx.size());
    return matrix;
}

CsrMatrix generatePowerLawLowerTriangular(int n, double baseDensity, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> probabilityDist(0.0, 1.0);

    CsrMatrix matrix;
    matrix.n = n;
    matrix.rowPtr.resize(static_cast<std::size_t>(n + 1), 0);

    for (int row = 0; row < n; ++row) {
        const double scale = 1.0 / std::sqrt(static_cast<double>(row + 1));
        const double rowDensity = std::min(1.0, baseDensity * 12.0 * scale);
        for (int col = 0; col < row; ++col) {
            if (probabilityDist(rng) < rowDensity) {
                matrix.colIdx.push_back(col);
                matrix.values.push_back(0.1 + 0.03 * static_cast<double>((row - col) % 19));
            }
        }
        matrix.colIdx.push_back(row);
        matrix.values.push_back(static_cast<double>(n) + 5.0);
        matrix.rowPtr[static_cast<std::size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
    }

    matrix.nnz = static_cast<int>(matrix.colIdx.size());
    return matrix;
}

std::vector<double> generateRandomRhs(int n, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> rhs(static_cast<std::size_t>(n));
    for (double& value : rhs) {
        value = dist(rng);
    }
    return rhs;
}

double computeResidual(const CsrMatrix& matrix,
                       const std::vector<double>& solution,
                       const std::vector<double>& rhs)
{
    double residual = 0.0;
    for (int row = 0; row < matrix.n; ++row) {
        double dot = 0.0;
        for (int entry = matrix.rowPtr[static_cast<std::size_t>(row)];
             entry < matrix.rowPtr[static_cast<std::size_t>(row + 1)];
             ++entry) {
            dot += matrix.values[static_cast<std::size_t>(entry)] *
                   solution[static_cast<std::size_t>(matrix.colIdx[static_cast<std::size_t>(entry)])];
        }
        const double diff = dot - rhs[static_cast<std::size_t>(row)];
        residual += diff * diff;
    }
    return std::sqrt(residual);
}

struct MklSptrsvContext {
    sparse_matrix_t handle = nullptr;
    matrix_descr descr{};

    ~MklSptrsvContext()
    {
        if (handle != nullptr) {
            mkl_sparse_destroy(handle);
        }
    }
};

MklSptrsvContext makeMklContext(const CsrMatrix& matrix)
{
    MklSptrsvContext ctx;
    ctx.descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    ctx.descr.mode = SPARSE_FILL_MODE_LOWER;
    ctx.descr.diag = SPARSE_DIAG_NON_UNIT;

    sparse_status_t status = mkl_sparse_d_create_csr(
        &ctx.handle,
        SPARSE_INDEX_BASE_ZERO,
        matrix.n,
        matrix.n,
        const_cast<int*>(matrix.rowPtr.data()),
        const_cast<int*>(matrix.rowPtr.data() + 1),
        const_cast<int*>(matrix.colIdx.data()),
        const_cast<double*>(matrix.values.data()));
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_d_create_csr failed");
    }

    status = mkl_sparse_set_sv_hint(ctx.handle, SPARSE_OPERATION_NON_TRANSPOSE, ctx.descr, 1000);
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_set_sv_hint failed");
    }

    status = mkl_sparse_optimize(ctx.handle);
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_optimize failed");
    }

    return ctx;
}

void sptrsvMkl(const MklSptrsvContext& ctx,
               const std::vector<double>& rhs,
               std::vector<double>& solution)
{
    solution.assign(rhs.size(), 0.0);
    const sparse_status_t status = mkl_sparse_d_trsv(
        SPARSE_OPERATION_NON_TRANSPOSE,
        1.0,
        ctx.handle,
        ctx.descr,
        rhs.data(),
        solution.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_d_trsv failed");
    }
}

void sptrsvMklEndToEnd(const CsrMatrix& matrix,
                       const std::vector<double>& rhs,
                       std::vector<double>& solution)
{
    const MklSptrsvContext ctx = makeMklContext(matrix);
    sptrsvMkl(ctx, rhs, solution);
}

CsrMatrix loadMatrixMarketAsLowerTriangular(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open Matrix Market file: " + path);
    }

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("Empty Matrix Market file: " + path);
    }

    std::istringstream header(line);
    std::string banner;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;
    header >> banner >> object >> format >> field >> symmetry;
    if (banner != "%%MatrixMarket" || object != "matrix" || format != "coordinate") {
        throw std::runtime_error("Unsupported Matrix Market header: " + path);
    }
    if (field != "real" && field != "integer" && field != "pattern") {
        throw std::runtime_error("Unsupported Matrix Market field '" + field + "': " + path);
    }
    if (symmetry != "general" && symmetry != "symmetric") {
        throw std::runtime_error("Unsupported Matrix Market symmetry '" + symmetry + "': " + path);
    }

    do {
        if (!std::getline(in, line)) {
            throw std::runtime_error("Missing size line in Matrix Market file: " + path);
        }
    } while (!line.empty() && line[0] == '%');

    std::istringstream dims(line);
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    dims >> rows >> cols >> nnz;
    if (!dims || rows <= 0 || cols <= 0 || rows != cols) {
        throw std::runtime_error("Matrix must be square: " + path);
    }

    std::vector<std::tuple<int, int, double>> entries;
    entries.reserve(static_cast<std::size_t>(nnz));
    for (int i = 0; i < nnz; ++i) {
        if (!std::getline(in, line)) {
            throw std::runtime_error("Unexpected EOF while reading " + path);
        }
        if (line.empty() || line[0] == '%') {
            --i;
            continue;
        }
        std::istringstream entry(line);
        int row = 0;
        int col = 0;
        double value = 1.0;
        entry >> row >> col;
        if (!entry) {
            throw std::runtime_error("Invalid Matrix Market entry in " + path);
        }
        if (field == "real" || field == "integer") {
            entry >> value;
            if (!entry) {
                throw std::runtime_error("Invalid Matrix Market value in " + path);
            }
        }
        --row;
        --col;
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::runtime_error("Out-of-range Matrix Market entry in " + path);
        }
        entries.emplace_back(row, col, value);
        if (symmetry == "symmetric" && row != col) {
            entries.emplace_back(col, row, value);
        }
    }

    std::vector<std::vector<std::pair<int, double>>> rowEntries(static_cast<std::size_t>(rows));
    for (const auto& [row, col, value] : entries) {
        if (col <= row) {
            rowEntries[static_cast<std::size_t>(row)].push_back({col, value});
        }
    }

    CsrMatrix matrix;
    matrix.n = rows;
    matrix.rowPtr.resize(static_cast<std::size_t>(rows + 1), 0);
    for (int row = 0; row < rows; ++row) {
        auto& entriesForRow = rowEntries[static_cast<std::size_t>(row)];
        std::sort(entriesForRow.begin(), entriesForRow.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        bool hasDiagonal = false;
        for (const auto& [col, value] : entriesForRow) {
            if (col == row) {
                hasDiagonal = true;
            }
            matrix.colIdx.push_back(col);
            matrix.values.push_back(value);
        }
        if (!hasDiagonal) {
            matrix.colIdx.push_back(row);
            matrix.values.push_back(static_cast<double>(rows) + 1.0);
        } else {
            for (std::size_t idx = static_cast<std::size_t>(matrix.rowPtr[static_cast<std::size_t>(row)]);
                 idx < matrix.colIdx.size();
                 ++idx) {
                if (matrix.colIdx[idx] == row) {
                    matrix.values[idx] += static_cast<double>(rows) + 1.0;
                    break;
                }
            }
        }
        matrix.rowPtr[static_cast<std::size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
    }

    matrix.nnz = static_cast<int>(matrix.colIdx.size());
    return matrix;
}

struct BenchmarkCase {
    std::string name;
    std::function<CsrMatrix(int)> factory;
};

struct RealMatrixCase {
    std::string family;
    std::string name;
    std::string path;
};

std::vector<BenchmarkCase> makeBenchmarkCases()
{
    return {
        {"random", [](int n) { return generateLowerTriangular(n, 0.01, 42); }},
        {"banded", [](int n) { return generateBandedLowerTriangular(n, 64); }},
        {"chain", [](int n) { return generateChainLowerTriangular(n); }},
        {"wide-level", [](int n) { return generateWideLevelLowerTriangular(n, 32); }},
        {"block", [](int n) { return generateBlockLowerTriangular(n, 64, 0.0025, 42); }},
        {"power-law", [](int n) { return generatePowerLawLowerTriangular(n, 0.002, 42); }}
    };
}

std::vector<RealMatrixCase> makeRealMatrixCases()
{
    return {
        {"random_irregular", "ASIC_100k", "data/matrices/random_irregular/ASIC_100k/ASIC_100k.mtx"},
        {"banded", "LFAT5000", "data/matrices/banded/LFAT5000/LFAT5000.mtx"},
        {"chain_like", "poli", "data/matrices/chain_like/poli/poli.mtx"},
        {"wide_level", "web-Stanford", "data/matrices/wide_level/web-Stanford/web-Stanford.mtx"},
        {"block_structured", "circuit204", "data/matrices/block_structured/circuit204/circuit204.mtx"},
        {"power_law", "soc-sign-epinions", "data/matrices/power_law/soc-sign-epinions/soc-sign-epinions.mtx"},
        {"extra", "msc04515", "data/matrices/more/Boeing__msc04515/msc04515/msc04515.mtx"},
        {"extra", "pwtk", "data/matrices/more/Boeing__pwtk/pwtk/pwtk.mtx"},
        {"extra", "helm2d03", "data/matrices/more/GHS_indef__helm2d03/helm2d03/helm2d03.mtx"},
        {"extra", "apache1", "data/matrices/more/GHS_psdef__apache1/apache1/apache1.mtx"},
        {"extra", "bcsstk13", "data/matrices/more/HB__bcsstk13/bcsstk13/bcsstk13.mtx"},
        {"extra", "bcsstk14", "data/matrices/more/HB__bcsstk14/bcsstk14/bcsstk14.mtx"},
        {"extra", "bcsstk17", "data/matrices/more/HB__bcsstk17/bcsstk17/bcsstk17.mtx"},
        {"extra", "bcsstk18", "data/matrices/more/HB__bcsstk18/bcsstk18/bcsstk18.mtx"},
        {"extra", "bcsstk27", "data/matrices/more/HB__bcsstk27/bcsstk27/bcsstk27.mtx"},
        {"extra", "bcsstm22", "data/matrices/more/HB__bcsstm22/bcsstm22/bcsstm22.mtx"},
        {"extra", "scircuit", "data/matrices/more/Hamm__scircuit/scircuit/scircuit.mtx"},
        {"extra", "fv1", "data/matrices/more/Norris__fv1/fv1/fv1.mtx"},
        {"extra", "fv2", "data/matrices/more/Norris__fv2/fv2/fv2.mtx"},
        {"extra", "fv3", "data/matrices/more/Norris__fv3/fv3/fv3.mtx"},
        {"extra", "EPA", "data/matrices/more/Pajek__EPA/EPA/EPA.mtx"},
        {"extra", "ca-GrQc", "data/matrices/more/SNAP__ca-GrQc/ca-GrQc/ca-GrQc.mtx"},
        {"extra", "email-Enron", "data/matrices/more/SNAP__email-Enron/email-Enron/email-Enron.mtx"},
        {"extra", "soc-Epinions1", "data/matrices/more/SNAP__soc-Epinions1/soc-Epinions1/soc-Epinions1.mtx"},
        {"extra", "roadNet-CA", "data/matrices/more/SNAP__roadNet-CA/roadNet-CA/roadNet-CA.mtx"},
        {"extra", "wiki-Vote", "data/matrices/more/SNAP__wiki-Vote/wiki-Vote/wiki-Vote.mtx"},
        {"extra", "ASIC_680k", "data/matrices/more/Sandia__ASIC_680k/ASIC_680k/ASIC_680k.mtx"},
        {"very_large", "web-BerkStan", "data/matrices/very_large/web-BerkStan/web-BerkStan.mtx"},
        {"very_large", "roadNet-TX", "data/matrices/very_large/roadNet-TX/roadNet-TX.mtx"},
        {"very_large", "wiki-Talk", "data/matrices/very_large/wiki-Talk/wiki-Talk.mtx"}
    };
}

void printSingleResult(const std::string& label,
                       const CsrMatrix& matrix,
                       const std::vector<double>& rhs,
                       std::function<void(std::vector<double>&)> solve)
{
    std::vector<double> solution;
    const double ms = timeMs([&]() { solve(solution); });
    const double residual = computeResidual(matrix, solution, rhs);
    std::cout << std::setw(14) << label
              << std::setw(14) << std::fixed << std::setprecision(4) << ms
              << std::setw(14) << std::scientific << std::setprecision(4) << residual
              << "\n";
}

void runSyntheticSweep(const std::string& familyFilter = "", int onlyN = 0)
{
    const std::vector<int> sizes = {1024, 4096, 16384, 32768};
    const auto cases = makeBenchmarkCases();

    for (const auto& benchmarkCase : cases) {
        if (!familyFilter.empty() && benchmarkCase.name != familyFilter) {
            continue;
        }
        std::cout << "\nSynthetic Family: " << benchmarkCase.name << "\n";
        std::cout << std::setw(12) << "N"
                  << std::setw(14) << "NNZ"
                  << std::setw(16) << "Basic (ms)"
                  << std::setw(12) << "B xMKL"
                  << std::setw(16) << "BranchRed (ms)"
                  << std::setw(12) << "R xMKL"
                  << std::setw(16) << "Unrolled (ms)"
                  << std::setw(12) << "U xMKL"
                  << std::setw(16) << "ThreshOmp"
                  << std::setw(12) << "T xMKL"
                  << std::setw(16) << "ChunkedOmp"
                  << std::setw(12) << "C xMKL"
                  << std::setw(16) << "CapChunkO"
                  << std::setw(12) << "CC xMKL"
                  << std::setw(16) << "MKL (ms)"
                  << std::setw(16) << "Residual B"
                  << std::setw(16) << "Residual R"
                  << std::setw(16) << "Residual U"
                  << std::setw(16) << "Residual T"
                  << std::setw(16) << "Residual C"
                  << std::setw(16) << "Residual CC"
                  << std::setw(16) << "Residual M"
                  << "\n";
        std::cout << std::string(322, '-') << "\n";

        for (int n : sizes) {
            if (onlyN > 0 && n != onlyN) {
                continue;
            }
            const CsrMatrix matrix = benchmarkCase.factory(n);
            const std::vector<double> rhs = generateRandomRhs(n, 99);
            const LevelSchedule schedule = buildLevelSchedule(matrix);
            std::vector<double> basicSolution;
            std::vector<double> branchReducedSolution;
            std::vector<double> unrolledSolution;
            std::vector<double> thresholdOmpSolution;
            std::vector<double> chunkedOmpSolution;
            std::vector<double> cappedChunkOmpSolution;
            std::vector<double> mklSolution;
            const MklSptrsvContext mkl = makeMklContext(matrix);

            const double basicMs = timeMs([&]() { sptrsvCpuBasic(matrix, rhs, basicSolution); });
            const double branchReducedMs =
                timeMs([&]() { sptrsvCpuBranchReduced(matrix, rhs, branchReducedSolution); });
            const double unrolledMs =
                timeMs([&]() { sptrsvCpuUnrolled(matrix, rhs, unrolledSolution); });
            const double thresholdOmpMs =
                timeMs([&]() { sptrsvCpuThresholdOmp(matrix, schedule, rhs, thresholdOmpSolution); });
            const double chunkedOmpMs =
                timeMs([&]() { sptrsvCpuChunkedOmp(matrix, schedule, rhs, chunkedOmpSolution); });
            const double cappedChunkOmpMs =
                timeMs([&]() { sptrsvCpuCappedChunkOmp(matrix, schedule, rhs, cappedChunkOmpSolution); });
            const double mklMs = timeMs([&]() { sptrsvMkl(mkl, rhs, mklSolution); });
            const double basicResidual = computeResidual(matrix, basicSolution, rhs);
            const double branchReducedResidual = computeResidual(matrix, branchReducedSolution, rhs);
            const double unrolledResidual = computeResidual(matrix, unrolledSolution, rhs);
            const double thresholdOmpResidual = computeResidual(matrix, thresholdOmpSolution, rhs);
            const double chunkedOmpResidual = computeResidual(matrix, chunkedOmpSolution, rhs);
            const double cappedChunkOmpResidual = computeResidual(matrix, cappedChunkOmpSolution, rhs);
            const double mklResidual = computeResidual(matrix, mklSolution, rhs);
            const double basicSpeedup = speedupOverMkl(basicMs, mklMs);
            const double branchReducedSpeedup = speedupOverMkl(branchReducedMs, mklMs);
            const double unrolledSpeedup = speedupOverMkl(unrolledMs, mklMs);
            const double thresholdOmpSpeedup = speedupOverMkl(thresholdOmpMs, mklMs);
            const double chunkedOmpSpeedup = speedupOverMkl(chunkedOmpMs, mklMs);
            const double cappedChunkOmpSpeedup = speedupOverMkl(cappedChunkOmpMs, mklMs);

            std::cout << std::setw(12) << n
                      << std::setw(14) << matrix.nnz
                      << std::setw(16) << std::fixed << std::setprecision(4) << basicMs
                      << std::setw(12) << std::fixed << std::setprecision(2) << basicSpeedup
                      << std::setw(16) << std::fixed << std::setprecision(4) << branchReducedMs
                      << std::setw(12) << std::fixed << std::setprecision(2) << branchReducedSpeedup
                      << std::setw(16) << std::fixed << std::setprecision(4) << unrolledMs
                      << std::setw(12) << std::fixed << std::setprecision(2) << unrolledSpeedup
                      << std::setw(16) << std::fixed << std::setprecision(4) << thresholdOmpMs
                      << std::setw(12) << std::fixed << std::setprecision(2) << thresholdOmpSpeedup
                      << std::setw(16) << std::fixed << std::setprecision(4) << chunkedOmpMs
                      << std::setw(12) << std::fixed << std::setprecision(2) << chunkedOmpSpeedup
                      << std::setw(16) << std::fixed << std::setprecision(4) << cappedChunkOmpMs
                      << std::setw(12) << std::fixed << std::setprecision(2) << cappedChunkOmpSpeedup
                      << std::setw(16) << std::fixed << std::setprecision(4) << mklMs
                      << std::setw(16) << std::scientific << std::setprecision(4) << basicResidual
                      << std::setw(16) << std::scientific << std::setprecision(4) << branchReducedResidual
                      << std::setw(16) << std::scientific << std::setprecision(4) << unrolledResidual
                      << std::setw(16) << std::scientific << std::setprecision(4) << thresholdOmpResidual
                      << std::setw(16) << std::scientific << std::setprecision(4) << chunkedOmpResidual
                      << std::setw(16) << std::scientific << std::setprecision(4) << cappedChunkOmpResidual
                      << std::setw(16) << std::scientific << std::setprecision(4) << mklResidual
                      << std::endl;
        }
    }
}

void runRealSweep()
{
    const auto cases = makeRealMatrixCases();
    std::cout << std::setw(18) << "Family"
              << std::setw(18) << "Matrix"
              << std::setw(14) << "N"
              << std::setw(14) << "NNZ"
              << std::setw(16) << "Basic (ms)"
              << std::setw(12) << "B xMKL"
              << std::setw(16) << "BranchRed (ms)"
              << std::setw(12) << "R xMKL"
              << std::setw(16) << "Unrolled (ms)"
              << std::setw(12) << "U xMKL"
              << std::setw(16) << "ThreshOmp"
              << std::setw(12) << "T xMKL"
              << std::setw(16) << "ChunkedOmp"
              << std::setw(12) << "C xMKL"
              << std::setw(16) << "CapChunkO"
              << std::setw(12) << "CC xMKL"
              << std::setw(16) << "MKL (ms)"
              << std::setw(16) << "Residual B"
              << std::setw(16) << "Residual R"
              << std::setw(16) << "Residual U"
              << std::setw(16) << "Residual T"
              << std::setw(16) << "Residual C"
              << std::setw(16) << "Residual CC"
              << std::setw(16) << "Residual M"
              << "\n";
    std::cout << std::string(372, '-') << "\n";

    for (const auto& realCase : cases) {
        try {
            const CsrMatrix matrix = loadMatrixMarketAsLowerTriangular(realCase.path);
            const std::vector<double> rhs = generateRandomRhs(matrix.n, 99);
            const LevelSchedule schedule = buildLevelSchedule(matrix);
            std::vector<double> basicSolution;
            std::vector<double> branchReducedSolution;
            std::vector<double> unrolledSolution;
            std::vector<double> thresholdOmpSolution;
            std::vector<double> chunkedOmpSolution;
            std::vector<double> cappedChunkOmpSolution;
            std::vector<double> mklSolution;
            const MklSptrsvContext mkl = makeMklContext(matrix);

            const double basicMs = timeMs([&]() { sptrsvCpuBasic(matrix, rhs, basicSolution); });
            const double branchReducedMs =
                timeMs([&]() { sptrsvCpuBranchReduced(matrix, rhs, branchReducedSolution); });
            const double unrolledMs =
                timeMs([&]() { sptrsvCpuUnrolled(matrix, rhs, unrolledSolution); });
            const double thresholdOmpMs =
                timeMs([&]() { sptrsvCpuThresholdOmp(matrix, schedule, rhs, thresholdOmpSolution); });
            const double chunkedOmpMs =
                timeMs([&]() { sptrsvCpuChunkedOmp(matrix, schedule, rhs, chunkedOmpSolution); });
            const double cappedChunkOmpMs =
                timeMs([&]() { sptrsvCpuCappedChunkOmp(matrix, schedule, rhs, cappedChunkOmpSolution); });
            const double mklMs = timeMs([&]() { sptrsvMkl(mkl, rhs, mklSolution); });
            const double basicResidual = computeResidual(matrix, basicSolution, rhs);
            const double branchReducedResidual = computeResidual(matrix, branchReducedSolution, rhs);
            const double unrolledResidual = computeResidual(matrix, unrolledSolution, rhs);
            const double thresholdOmpResidual = computeResidual(matrix, thresholdOmpSolution, rhs);
            const double chunkedOmpResidual = computeResidual(matrix, chunkedOmpSolution, rhs);
            const double cappedChunkOmpResidual = computeResidual(matrix, cappedChunkOmpSolution, rhs);
            const double mklResidual = computeResidual(matrix, mklSolution, rhs);
            const double basicSpeedup = speedupOverMkl(basicMs, mklMs);
            const double branchReducedSpeedup = speedupOverMkl(branchReducedMs, mklMs);
            const double unrolledSpeedup = speedupOverMkl(unrolledMs, mklMs);
            const double thresholdOmpSpeedup = speedupOverMkl(thresholdOmpMs, mklMs);
            const double chunkedOmpSpeedup = speedupOverMkl(chunkedOmpMs, mklMs);
            const double cappedChunkOmpSpeedup = speedupOverMkl(cappedChunkOmpMs, mklMs);

            std::cout << std::setw(18) << realCase.family
                      << std::setw(18) << realCase.name
                      << std::setw(14) << matrix.n
                      << std::setw(14) << matrix.nnz
                      << std::setw(16) << std::fixed << std::setprecision(4) << basicMs
                      << std::setw(12) << std::fixed << std::setprecision(2) << basicSpeedup
                      << std::setw(16) << std::fixed << std::setprecision(4) << branchReducedMs
                      << std::setw(12) << std::fixed << std::setprecision(2) << branchReducedSpeedup
                      << std::setw(16) << std::fixed << std::setprecision(4) << unrolledMs
                      << std::setw(12) << std::fixed << std::setprecision(2) << unrolledSpeedup
                      << std::setw(16) << std::fixed << std::setprecision(4) << thresholdOmpMs
                      << std::setw(12) << std::fixed << std::setprecision(2) << thresholdOmpSpeedup
                      << std::setw(16) << std::fixed << std::setprecision(4) << chunkedOmpMs
                      << std::setw(12) << std::fixed << std::setprecision(2) << chunkedOmpSpeedup
                      << std::setw(16) << std::fixed << std::setprecision(4) << cappedChunkOmpMs
                      << std::setw(12) << std::fixed << std::setprecision(2) << cappedChunkOmpSpeedup
                      << std::setw(16) << std::fixed << std::setprecision(4) << mklMs
                      << std::setw(16) << std::scientific << std::setprecision(4) << basicResidual
                      << std::setw(16) << std::scientific << std::setprecision(4) << branchReducedResidual
                      << std::setw(16) << std::scientific << std::setprecision(4) << unrolledResidual
                      << std::setw(16) << std::scientific << std::setprecision(4) << thresholdOmpResidual
                      << std::setw(16) << std::scientific << std::setprecision(4) << chunkedOmpResidual
                      << std::setw(16) << std::scientific << std::setprecision(4) << cappedChunkOmpResidual
                      << std::setw(16) << std::scientific << std::setprecision(4) << mklResidual
                      << std::endl;
        } catch (const std::exception& ex) {
            std::cout << std::setw(18) << realCase.family
                      << std::setw(18) << realCase.name
                      << "  ERROR: " << ex.what() << std::endl;
        }
    }
}

int findRealMatrixIndex(const std::string& name)
{
    const auto cases = makeRealMatrixCases();
    for (std::size_t i = 0; i < cases.size(); ++i) {
        if (cases[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void runRealOne(const std::string& name, const std::string& oneMode)
{
    const auto cases = makeRealMatrixCases();
    const int index = findRealMatrixIndex(name);
    if (index < 0) {
        throw std::runtime_error("Unknown real matrix name: " + name);
    }

    const auto& realCase = cases[static_cast<std::size_t>(index)];
    const CsrMatrix matrix = loadMatrixMarketAsLowerTriangular(realCase.path);
    const std::vector<double> rhs = generateRandomRhs(matrix.n, 99);
    const LevelSchedule schedule = buildLevelSchedule(matrix);
    const bool useMklSolveOnly = (oneMode == "all" || oneMode == "mkl" || oneMode == "mkl-solve");
    const MklSptrsvContext mkl = useMklSolveOnly ? makeMklContext(matrix) : MklSptrsvContext{};
    std::cout << "Real Matrix: " << realCase.name << "\n";
    std::cout << "Family     : " << realCase.family << "\n";
    std::cout << "Path       : " << realCase.path << "\n";
    std::cout << "N          : " << matrix.n << "\n";
    std::cout << "NNZ        : " << matrix.nnz << "\n\n";

    std::cout << std::setw(14) << "Kernel"
              << std::setw(14) << "Time (ms)"
              << std::setw(14) << "Residual"
              << "\n";
    std::cout << std::string(42, '-') << "\n";

    if (oneMode == "all" || oneMode == "basic") {
        printSingleResult("Basic", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvCpuBasic(matrix, rhs, solution); });
    }
    if (oneMode == "all" || oneMode == "branchred") {
        printSingleResult("BranchRed", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvCpuBranchReduced(matrix, rhs, solution); });
    }
    if (oneMode == "all" || oneMode == "unrolled") {
        printSingleResult("Unrolled", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvCpuUnrolled(matrix, rhs, solution); });
    }
    if (oneMode == "all" || oneMode == "thresholdomp") {
        printSingleResult("ThreshOmp", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvCpuThresholdOmp(matrix, schedule, rhs, solution); });
    }
    if (oneMode == "all" || oneMode == "chunkedomp") {
        printSingleResult("ChunkOmp", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvCpuChunkedOmp(matrix, schedule, rhs, solution); });
    }
    if (oneMode == "all" || oneMode == "capchunkomp") {
        printSingleResult("CapChunkO", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvCpuCappedChunkOmp(matrix, schedule, rhs, solution); });
    }
    if (oneMode == "all" || oneMode == "mkl") {
        printSingleResult("MKL", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvMkl(mkl, rhs, solution); });
    }
    if (oneMode == "mkl-solve") {
        printSingleResult("MKL", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvMkl(mkl, rhs, solution); });
    }
}

void runRealProfile(const std::string& name, const std::string& kernel, int iters)
{
    if (iters <= 0) {
        throw std::runtime_error("Profile iterations must be positive");
    }

    const auto cases = makeRealMatrixCases();
    const int index = findRealMatrixIndex(name);
    if (index < 0) {
        throw std::runtime_error("Unknown real matrix name: " + name);
    }

    const auto& realCase = cases[static_cast<std::size_t>(index)];
    const CsrMatrix matrix = loadMatrixMarketAsLowerTriangular(realCase.path);
    const std::vector<double> rhs = generateRandomRhs(matrix.n, 99);
    const LevelSchedule schedule = buildLevelSchedule(matrix);
    const bool useMkl = (kernel == "mkl" || kernel == "mkl-solve");
    const MklSptrsvContext mkl = useMkl ? makeMklContext(matrix) : MklSptrsvContext{};
    std::vector<double> solution;

    auto solveOnce = [&]() {
        if (kernel == "basic") {
            sptrsvCpuBasic(matrix, rhs, solution);
        } else if (kernel == "branchred") {
            sptrsvCpuBranchReduced(matrix, rhs, solution);
        } else if (kernel == "unrolled") {
            sptrsvCpuUnrolled(matrix, rhs, solution);
        } else if (kernel == "thresholdomp") {
            sptrsvCpuThresholdOmp(matrix, schedule, rhs, solution);
        } else if (kernel == "chunkedomp") {
            sptrsvCpuChunkedOmp(matrix, schedule, rhs, solution);
        } else if (kernel == "capchunkomp") {
            sptrsvCpuCappedChunkOmp(matrix, schedule, rhs, solution);
        } else if (useMkl) {
            sptrsvMkl(mkl, rhs, solution);
        } else {
            throw std::runtime_error("Unknown profile kernel: " + kernel);
        }
    };

    // Warm up once so VTune spends less time on cold-start effects.
    solveOnce();

    for (int iter = 0; iter < iters; ++iter) {
        solveOnce();
    }

    const double residual = computeResidual(matrix, solution, rhs);
    std::cout << "Profile Matrix: " << realCase.name << "\n";
    std::cout << "Kernel       : " << kernel << "\n";
    std::cout << "Iterations   : " << iters << "\n";
    std::cout << "N            : " << matrix.n << "\n";
    std::cout << "NNZ          : " << matrix.nnz << "\n";
    std::cout << "Residual     : " << std::scientific << std::setprecision(4) << residual << "\n";
}

void runFamilyOne(const std::string& name, int n, const std::string& mode)
{
    const auto cases = makeBenchmarkCases();
    auto it = std::find_if(
        cases.begin(),
        cases.end(),
        [&](const BenchmarkCase& c) { return c.name == name; });
    if (it == cases.end()) {
        throw std::runtime_error("Unknown family name: " + name);
    }

    const CsrMatrix matrix = it->factory(n);
    const std::vector<double> rhs = generateRandomRhs(matrix.n, 99);
    const LevelSchedule schedule = buildLevelSchedule(matrix);
    const bool useMklSolveOnly = (mode == "all" || mode == "mkl");
    const MklSptrsvContext mkl = useMklSolveOnly ? makeMklContext(matrix) : MklSptrsvContext{};

    std::cout << "Synthetic Family: " << name << "\n";
    std::cout << "N               : " << matrix.n << "\n";
    std::cout << "NNZ             : " << matrix.nnz << "\n\n";

    std::cout << std::setw(14) << "Kernel"
              << std::setw(14) << "Time (ms)"
              << std::setw(14) << "Residual"
              << "\n";
    std::cout << std::string(42, '-') << "\n";

    if (mode == "all" || mode == "basic") {
        printSingleResult("Basic", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvCpuBasic(matrix, rhs, solution); });
    }
    if (mode == "all" || mode == "branchred") {
        printSingleResult("BranchRed", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvCpuBranchReduced(matrix, rhs, solution); });
    }
    if (mode == "all" || mode == "unrolled") {
        printSingleResult("Unrolled", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvCpuUnrolled(matrix, rhs, solution); });
    }
    if (mode == "all" || mode == "thresholdomp") {
        printSingleResult("ThreshOmp", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvCpuThresholdOmp(matrix, schedule, rhs, solution); });
    }
    if (mode == "all" || mode == "chunkedomp") {
        printSingleResult("ChunkOmp", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvCpuChunkedOmp(matrix, schedule, rhs, solution); });
    }
    if (mode == "all" || mode == "capchunkomp") {
        printSingleResult("CapChunkO", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvCpuCappedChunkOmp(matrix, schedule, rhs, solution); });
    }
    if (mode == "all" || mode == "mkl") {
        printSingleResult("MKL", matrix, rhs,
                          [&](std::vector<double>& solution) { sptrsvMkl(mkl, rhs, solution); });
    }
}

}  // namespace

int main(int argc, char* argv[])
{
    try {
        mkl_set_dynamic(0);
        mkl_set_num_threads(1);

        if (argc == 1) {
            runSyntheticSweep();
            return 0;
        }

        const std::string mode = argv[1];
        if (mode == "families") {
            const std::string family = (argc >= 3) ? argv[2] : "";
            const int onlyN = (argc >= 4) ? std::atoi(argv[3]) : 0;
            runSyntheticSweep(family, onlyN);
            return 0;
        }
        if (mode == "family-one") {
            if (argc != 4 && argc != 5) {
                throw std::runtime_error("Usage: ./sptrsv_bench family-one <family-name> <n> [basic|branchred|unrolled|thresholdomp|chunkedomp|capchunkomp|mkl|all]");
            }
            runFamilyOne(argv[2], std::atoi(argv[3]), (argc == 5) ? argv[4] : "all");
            return 0;
        }
        if (mode == "real") {
            runRealSweep();
            return 0;
        }
        if (mode == "real-one") {
            if (argc != 3 && argc != 4) {
                throw std::runtime_error("Usage: ./sptrsv_bench real-one <matrix-name> [basic|branchred|unrolled|thresholdomp|chunkedomp|capchunkomp|mkl|mkl-solve|all]");
            }
            runRealOne(argv[2], (argc == 4) ? argv[3] : "all");
            return 0;
        }
        if (mode == "real-profile") {
            if (argc != 4 && argc != 5) {
                throw std::runtime_error("Usage: ./sptrsv_bench real-profile <matrix-name> <basic|branchred|unrolled|thresholdomp|chunkedomp|capchunkomp|mkl|mkl-solve> [iters]");
            }
            runRealProfile(argv[2], argv[3], (argc == 5) ? std::atoi(argv[4]) : 500);
            return 0;
        }

        throw std::runtime_error("Usage: ./sptrsv_bench [families [family [n]] | family-one <family-name> <n> [basic|branchred|unrolled|thresholdomp|chunkedomp|capchunkomp|mkl|all] | real | real-one <matrix-name> [basic|branchred|unrolled|thresholdomp|chunkedomp|capchunkomp|mkl|mkl-solve|all] | real-profile <matrix-name> <basic|branchred|unrolled|thresholdomp|chunkedomp|capchunkomp|mkl|mkl-solve> [iters]]");
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
