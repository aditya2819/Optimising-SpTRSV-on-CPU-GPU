#include "sptrsv_kernel.h"
#include "sptrsv_kernel.cuh"
#include "learned_selector_weights.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <cstdint>
#include <array>
#include <limits>

template <typename Func>
double timeMs(Func&& func);

std::vector<int> computeRowLevels(const CsrMatrix& matrix);
std::string chooseLearnedKernel(const CsrMatrix& matrix);

namespace {

void checkCuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << what << ": " << cudaGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

void checkCusparse(cusparseStatus_t status, const char* what)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << what << ": cuSPARSE status " << static_cast<int>(status);
        throw std::runtime_error(oss.str());
    }
}

void setMatrixAttributes(cusparseSpMatDescr_t mat)
{
    cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
    checkCusparse(
        cusparseSpMatSetAttribute(mat, CUSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)),
        "cusparseSpMatSetAttribute(fill mode)");
    checkCusparse(
        cusparseSpMatSetAttribute(mat, CUSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)),
        "cusparseSpMatSetAttribute(diag type)");
}

bool useCoalescedForBandedAware(const CsrMatrix& matrix)
{
    long long adjacent = 0;
    long long possibleAdjacent = 0;
    long long totalRowNnz = 0;
    int maxRowNnz = 0;

    for (int row = 0; row < matrix.n; ++row) {
        const int rowStart = matrix.rowPtr[static_cast<size_t>(row)];
        const int rowEnd = matrix.rowPtr[static_cast<size_t>(row + 1)];
        const int rowNnz = rowEnd - rowStart;
        totalRowNnz += rowNnz;
        maxRowNnz = std::max(maxRowNnz, rowNnz);
        for (int idx = rowStart + 1; idx < rowEnd; ++idx) {
            ++possibleAdjacent;
            if (matrix.colIdx[static_cast<size_t>(idx)] == matrix.colIdx[static_cast<size_t>(idx - 1)] + 1) {
                ++adjacent;
            }
        }
    }

    const double avgRowNnz = matrix.n > 0 ? static_cast<double>(totalRowNnz) / static_cast<double>(matrix.n) : 0.0;
    const double adjacentFrac = possibleAdjacent > 0 ? static_cast<double>(adjacent) / static_cast<double>(possibleAdjacent) : 0.0;
    return adjacentFrac >= 0.6 && avgRowNnz <= 64.0 && maxRowNnz <= 256;
}

struct RandomSolveOnlyContext {
    int n = 0;
    int maxLevel = 0;
    size_t sharedBytes = 0;

    std::vector<int> threadOffsets;
    std::vector<int> threadCounts;
    std::vector<int> warpOffsets;
    std::vector<int> warpCounts;
    std::vector<int> blockOffsets;
    std::vector<int> blockCounts;

    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;
    int* dThreadRows = nullptr;
    int* dWarpRows = nullptr;
    int* dBlockRows = nullptr;
    cudaStream_t threadStream = nullptr;
    cudaStream_t warpStream = nullptr;
    cudaStream_t blockStream = nullptr;
    cudaEvent_t threadDone = nullptr;
    cudaEvent_t warpDone = nullptr;
    cudaEvent_t blockDone = nullptr;
};

void destroyRandomSolveOnlyContext(RandomSolveOnlyContext& ctx)
{
    if (ctx.threadDone) cudaEventDestroy(ctx.threadDone);
    if (ctx.warpDone) cudaEventDestroy(ctx.warpDone);
    if (ctx.blockDone) cudaEventDestroy(ctx.blockDone);
    if (ctx.threadStream) cudaStreamDestroy(ctx.threadStream);
    if (ctx.warpStream) cudaStreamDestroy(ctx.warpStream);
    if (ctx.blockStream) cudaStreamDestroy(ctx.blockStream);
    if (ctx.dBlockRows) cudaFree(ctx.dBlockRows);
    if (ctx.dWarpRows) cudaFree(ctx.dWarpRows);
    if (ctx.dThreadRows) cudaFree(ctx.dThreadRows);
    if (ctx.dSolution) cudaFree(ctx.dSolution);
    if (ctx.dRhs) cudaFree(ctx.dRhs);
    if (ctx.dValues) cudaFree(ctx.dValues);
    if (ctx.dColIdx) cudaFree(ctx.dColIdx);
    if (ctx.dRowPtr) cudaFree(ctx.dRowPtr);
    ctx = {};
}

RandomSolveOnlyContext makeRandomSolveOnlyContext(const CsrMatrix& matrix, const std::vector<double>& rhs)
{
    constexpr int kThreadThreshold = 2;
    constexpr int kWarpThreshold = 64;
    constexpr int kBlockSize = 256;

    RandomSolveOnlyContext ctx;
    ctx.n = matrix.n;
    ctx.sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);

    const std::vector<int> levels = computeRowLevels(matrix);
    for (int level : levels) {
        ctx.maxLevel = std::max(ctx.maxLevel, level);
    }

    std::vector<int> hostThreadRows;
    std::vector<int> hostWarpRows;
    std::vector<int> hostBlockRows;
    ctx.threadOffsets.resize(static_cast<size_t>(ctx.maxLevel + 1));
    ctx.threadCounts.resize(static_cast<size_t>(ctx.maxLevel + 1));
    ctx.warpOffsets.resize(static_cast<size_t>(ctx.maxLevel + 1));
    ctx.warpCounts.resize(static_cast<size_t>(ctx.maxLevel + 1));
    ctx.blockOffsets.resize(static_cast<size_t>(ctx.maxLevel + 1));
    ctx.blockCounts.resize(static_cast<size_t>(ctx.maxLevel + 1));

    for (int level = 0; level <= ctx.maxLevel; ++level) {
        ctx.threadOffsets[static_cast<size_t>(level)] = static_cast<int>(hostThreadRows.size());
        ctx.warpOffsets[static_cast<size_t>(level)] = static_cast<int>(hostWarpRows.size());
        ctx.blockOffsets[static_cast<size_t>(level)] = static_cast<int>(hostBlockRows.size());
        for (int row = 0; row < matrix.n; ++row) {
            if (levels[static_cast<size_t>(row)] != level) {
                continue;
            }
            const int rowNnz = matrix.rowPtr[static_cast<size_t>(row + 1)] - matrix.rowPtr[static_cast<size_t>(row)];
            if (rowNnz <= kThreadThreshold) {
                hostThreadRows.push_back(row);
            } else if (rowNnz <= kWarpThreshold) {
                hostWarpRows.push_back(row);
            } else {
                hostBlockRows.push_back(row);
            }
        }
        ctx.threadCounts[static_cast<size_t>(level)] = static_cast<int>(hostThreadRows.size()) - ctx.threadOffsets[static_cast<size_t>(level)];
        ctx.warpCounts[static_cast<size_t>(level)] = static_cast<int>(hostWarpRows.size()) - ctx.warpOffsets[static_cast<size_t>(level)];
        ctx.blockCounts[static_cast<size_t>(level)] = static_cast<int>(hostBlockRows.size()) - ctx.blockOffsets[static_cast<size_t>(level)];
    }

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(random persist rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(random persist colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(random persist values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(random persist rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(random persist solution)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dThreadRows), std::max<size_t>(1, hostThreadRows.size()) * sizeof(int)), "cudaMalloc(random persist thread rows)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dWarpRows), std::max<size_t>(1, hostWarpRows.size()) * sizeof(int)), "cudaMalloc(random persist warp rows)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dBlockRows), std::max<size_t>(1, hostBlockRows.size()) * sizeof(int)), "cudaMalloc(random persist block rows)");

        checkCuda(cudaStreamCreate(&ctx.threadStream), "cudaStreamCreate(random persist thread)");
        checkCuda(cudaStreamCreate(&ctx.warpStream), "cudaStreamCreate(random persist warp)");
        checkCuda(cudaStreamCreate(&ctx.blockStream), "cudaStreamCreate(random persist block)");
        checkCuda(cudaEventCreateWithFlags(&ctx.threadDone, cudaEventDisableTiming), "cudaEventCreate(random persist thread)");
        checkCuda(cudaEventCreateWithFlags(&ctx.warpDone, cudaEventDisableTiming), "cudaEventCreate(random persist warp)");
        checkCuda(cudaEventCreateWithFlags(&ctx.blockDone, cudaEventDisableTiming), "cudaEventCreate(random persist block)");

        checkCuda(cudaMemcpy(ctx.dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(random persist rowPtr)");
        checkCuda(cudaMemcpy(ctx.dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(random persist colIdx)");
        checkCuda(cudaMemcpy(ctx.dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(random persist values)");
        checkCuda(cudaMemcpy(ctx.dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(random persist rhs)");
        if (!hostThreadRows.empty()) {
            checkCuda(cudaMemcpy(ctx.dThreadRows, hostThreadRows.data(), hostThreadRows.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(random persist thread rows)");
        }
        if (!hostWarpRows.empty()) {
            checkCuda(cudaMemcpy(ctx.dWarpRows, hostWarpRows.data(), hostWarpRows.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(random persist warp rows)");
        }
        if (!hostBlockRows.empty()) {
            checkCuda(cudaMemcpy(ctx.dBlockRows, hostBlockRows.data(), hostBlockRows.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(random persist block rows)");
        }
        return ctx;
    } catch (...) {
        destroyRandomSolveOnlyContext(ctx);
        throw;
    }
}

double runRandomSolveOnlyContext(RandomSolveOnlyContext& ctx, std::vector<double>& solution)
{
    constexpr int kThreadBlockSize = 128;
    constexpr int kWarpBlockSize = 128;
    constexpr int kBlockSize = 256;

    const double ms = timeMs([&]() {
        checkCuda(cudaMemset(ctx.dSolution, 0, static_cast<size_t>(ctx.n) * sizeof(double)), "cudaMemset(random persist solution)");
        bool havePreviousLevelWork = false;
        bool prevThreadActive = false;
        bool prevWarpActive = false;
        bool prevBlockActive = false;

        for (int level = 0; level <= ctx.maxLevel; ++level) {
            const int threadCount = ctx.threadCounts[static_cast<size_t>(level)];
            const int warpCount = ctx.warpCounts[static_cast<size_t>(level)];
            const int blockCount = ctx.blockCounts[static_cast<size_t>(level)];

            const bool threadActive = threadCount > 0;
            const bool warpActive = warpCount > 0;
            const bool blockActive = blockCount > 0;

            if (havePreviousLevelWork) {
                if (threadActive) {
                    if (prevThreadActive) checkCuda(cudaStreamWaitEvent(ctx.threadStream, ctx.threadDone, 0), "cudaStreamWaitEvent(random persist thread/thread)");
                    if (prevWarpActive) checkCuda(cudaStreamWaitEvent(ctx.threadStream, ctx.warpDone, 0), "cudaStreamWaitEvent(random persist thread/warp)");
                    if (prevBlockActive) checkCuda(cudaStreamWaitEvent(ctx.threadStream, ctx.blockDone, 0), "cudaStreamWaitEvent(random persist thread/block)");
                }
                if (warpActive) {
                    if (prevThreadActive) checkCuda(cudaStreamWaitEvent(ctx.warpStream, ctx.threadDone, 0), "cudaStreamWaitEvent(random persist warp/thread)");
                    if (prevWarpActive) checkCuda(cudaStreamWaitEvent(ctx.warpStream, ctx.warpDone, 0), "cudaStreamWaitEvent(random persist warp/warp)");
                    if (prevBlockActive) checkCuda(cudaStreamWaitEvent(ctx.warpStream, ctx.blockDone, 0), "cudaStreamWaitEvent(random persist warp/block)");
                }
                if (blockActive) {
                    if (prevThreadActive) checkCuda(cudaStreamWaitEvent(ctx.blockStream, ctx.threadDone, 0), "cudaStreamWaitEvent(random persist block/thread)");
                    if (prevWarpActive) checkCuda(cudaStreamWaitEvent(ctx.blockStream, ctx.warpDone, 0), "cudaStreamWaitEvent(random persist block/warp)");
                    if (prevBlockActive) checkCuda(cudaStreamWaitEvent(ctx.blockStream, ctx.blockDone, 0), "cudaStreamWaitEvent(random persist block/block)");
                }
            }

            if (threadActive) {
                const int blocks = (threadCount + kThreadBlockSize - 1) / kThreadBlockSize;
                sptrsvThreadRowListKernel<<<blocks, kThreadBlockSize, 0, ctx.threadStream>>>(
                    threadCount,
                    ctx.dThreadRows + ctx.threadOffsets[static_cast<size_t>(level)],
                    ctx.dRowPtr,
                    ctx.dColIdx,
                    ctx.dValues,
                    ctx.dRhs,
                    ctx.dSolution);
                checkCuda(cudaEventRecord(ctx.threadDone, ctx.threadStream), "cudaEventRecord(random persist thread)");
            }

            if (warpActive) {
                const int warpsPerBlock = kWarpBlockSize / 32;
                const int blocks = (warpCount + warpsPerBlock - 1) / warpsPerBlock;
                sptrsvWarpRowListKernel<<<blocks, kWarpBlockSize, 0, ctx.warpStream>>>(
                    warpCount,
                    ctx.dWarpRows + ctx.warpOffsets[static_cast<size_t>(level)],
                    ctx.dRowPtr,
                    ctx.dColIdx,
                    ctx.dValues,
                    ctx.dRhs,
                    ctx.dSolution);
                checkCuda(cudaEventRecord(ctx.warpDone, ctx.warpStream), "cudaEventRecord(random persist warp)");
            }

            if (blockActive) {
                sptrsvBlockRowListKernel<<<blockCount, kBlockSize, ctx.sharedBytes, ctx.blockStream>>>(
                    blockCount,
                    ctx.dBlockRows + ctx.blockOffsets[static_cast<size_t>(level)],
                    ctx.dRowPtr,
                    ctx.dColIdx,
                    ctx.dValues,
                    ctx.dRhs,
                    ctx.dSolution);
                checkCuda(cudaEventRecord(ctx.blockDone, ctx.blockStream), "cudaEventRecord(random persist block)");
            }

            havePreviousLevelWork = threadActive || warpActive || blockActive;
            prevThreadActive = threadActive;
            prevWarpActive = warpActive;
            prevBlockActive = blockActive;
        }

        if (prevThreadActive) checkCuda(cudaStreamSynchronize(ctx.threadStream), "cudaStreamSynchronize(random persist thread)");
        if (prevWarpActive) checkCuda(cudaStreamSynchronize(ctx.warpStream), "cudaStreamSynchronize(random persist warp)");
        if (prevBlockActive) checkCuda(cudaStreamSynchronize(ctx.blockStream), "cudaStreamSynchronize(random persist block)");
        checkCuda(cudaGetLastError(), "runRandomSolveOnlyContext dispatch");
    });

    solution.resize(static_cast<size_t>(ctx.n));
    checkCuda(cudaMemcpy(solution.data(), ctx.dSolution, static_cast<size_t>(ctx.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(random persist solution)");
    return ms;
}

struct PowerSolveOnlyContext {
    int n = 0;
    int maxLevel = 0;
    size_t sharedBytes = 0;

    std::vector<int> warpOffsets;
    std::vector<int> warpCounts;
    std::vector<int> blockOffsets;
    std::vector<int> blockCounts;

    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;
    int* dWarpRows = nullptr;
    int* dBlockRows = nullptr;
    cudaStream_t workStream = nullptr;
};

void destroyPowerSolveOnlyContext(PowerSolveOnlyContext& ctx)
{
    if (ctx.workStream) cudaStreamDestroy(ctx.workStream);
    if (ctx.dBlockRows) cudaFree(ctx.dBlockRows);
    if (ctx.dWarpRows) cudaFree(ctx.dWarpRows);
    if (ctx.dSolution) cudaFree(ctx.dSolution);
    if (ctx.dRhs) cudaFree(ctx.dRhs);
    if (ctx.dValues) cudaFree(ctx.dValues);
    if (ctx.dColIdx) cudaFree(ctx.dColIdx);
    if (ctx.dRowPtr) cudaFree(ctx.dRowPtr);
    ctx = {};
}

PowerSolveOnlyContext makePowerSolveOnlyContext(const CsrMatrix& matrix, const std::vector<double>& rhs)
{
    constexpr int kWarpThreshold = 24;
    constexpr int kBlockSize = 256;

    PowerSolveOnlyContext ctx;
    ctx.n = matrix.n;
    ctx.sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);

    const std::vector<int> levels = computeRowLevels(matrix);
    for (int level : levels) {
        ctx.maxLevel = std::max(ctx.maxLevel, level);
    }

    std::vector<int> hostWarpRows;
    std::vector<int> hostBlockRows;
    ctx.warpOffsets.resize(static_cast<size_t>(ctx.maxLevel + 1));
    ctx.warpCounts.resize(static_cast<size_t>(ctx.maxLevel + 1));
    ctx.blockOffsets.resize(static_cast<size_t>(ctx.maxLevel + 1));
    ctx.blockCounts.resize(static_cast<size_t>(ctx.maxLevel + 1));

    for (int level = 0; level <= ctx.maxLevel; ++level) {
        ctx.warpOffsets[static_cast<size_t>(level)] = static_cast<int>(hostWarpRows.size());
        ctx.blockOffsets[static_cast<size_t>(level)] = static_cast<int>(hostBlockRows.size());
        for (int row = 0; row < matrix.n; ++row) {
            if (levels[static_cast<size_t>(row)] != level) {
                continue;
            }
            const int rowNnz = matrix.rowPtr[static_cast<size_t>(row + 1)] - matrix.rowPtr[static_cast<size_t>(row)];
            if (rowNnz <= kWarpThreshold) {
                hostWarpRows.push_back(row);
            } else {
                hostBlockRows.push_back(row);
            }
        }
        ctx.warpCounts[static_cast<size_t>(level)] = static_cast<int>(hostWarpRows.size()) - ctx.warpOffsets[static_cast<size_t>(level)];
        ctx.blockCounts[static_cast<size_t>(level)] = static_cast<int>(hostBlockRows.size()) - ctx.blockOffsets[static_cast<size_t>(level)];
    }

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(power persist rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(power persist colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(power persist values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(power persist rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(power persist solution)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dWarpRows), std::max<size_t>(1, hostWarpRows.size()) * sizeof(int)), "cudaMalloc(power persist warp rows)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dBlockRows), std::max<size_t>(1, hostBlockRows.size()) * sizeof(int)), "cudaMalloc(power persist block rows)");
        checkCuda(cudaStreamCreate(&ctx.workStream), "cudaStreamCreate(power persist work)");

        checkCuda(cudaMemcpy(ctx.dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(power persist rowPtr)");
        checkCuda(cudaMemcpy(ctx.dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(power persist colIdx)");
        checkCuda(cudaMemcpy(ctx.dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(power persist values)");
        checkCuda(cudaMemcpy(ctx.dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(power persist rhs)");
        if (!hostWarpRows.empty()) {
            checkCuda(cudaMemcpy(ctx.dWarpRows, hostWarpRows.data(), hostWarpRows.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(power persist warp rows)");
        }
        if (!hostBlockRows.empty()) {
            checkCuda(cudaMemcpy(ctx.dBlockRows, hostBlockRows.data(), hostBlockRows.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(power persist block rows)");
        }
        return ctx;
    } catch (...) {
        destroyPowerSolveOnlyContext(ctx);
        throw;
    }
}

double runPowerSolveOnlyContext(PowerSolveOnlyContext& ctx, std::vector<double>& solution)
{
    constexpr int kWarpBlockSize = 128;
    constexpr int kBlockSize = 256;
    constexpr int kMaxLevelsPerBatch = 32;
    constexpr int kMinRowsPerBatch = 2048;

    const double ms = timeMs([&]() {
        checkCuda(cudaMemset(ctx.dSolution, 0, static_cast<size_t>(ctx.n) * sizeof(double)), "cudaMemset(power persist solution)");
        int level = 0;
        while (level <= ctx.maxLevel) {
            int levelsInBatch = 0;
            int rowsInBatch = 0;
            bool launchedBatch = false;
            while (level <= ctx.maxLevel && levelsInBatch < kMaxLevelsPerBatch && (rowsInBatch < kMinRowsPerBatch || levelsInBatch == 0)) {
                const int warpCount = ctx.warpCounts[static_cast<size_t>(level)];
                const int blockCount = ctx.blockCounts[static_cast<size_t>(level)];
                if (warpCount > 0) {
                    const int warpsPerBlock = kWarpBlockSize / 32;
                    const int blocks = (warpCount + warpsPerBlock - 1) / warpsPerBlock;
                    sptrsvWarpRowListKernel<<<blocks, kWarpBlockSize, 0, ctx.workStream>>>(
                        warpCount,
                        ctx.dWarpRows + ctx.warpOffsets[static_cast<size_t>(level)],
                        ctx.dRowPtr,
                        ctx.dColIdx,
                        ctx.dValues,
                        ctx.dRhs,
                        ctx.dSolution);
                    launchedBatch = true;
                }
                if (blockCount > 0) {
                    sptrsvBlockRowListKernel<<<blockCount, kBlockSize, ctx.sharedBytes, ctx.workStream>>>(
                        blockCount,
                        ctx.dBlockRows + ctx.blockOffsets[static_cast<size_t>(level)],
                        ctx.dRowPtr,
                        ctx.dColIdx,
                        ctx.dValues,
                        ctx.dRhs,
                        ctx.dSolution);
                    launchedBatch = true;
                }
                rowsInBatch += warpCount + blockCount;
                ++levelsInBatch;
                ++level;
            }
            if (launchedBatch) {
                checkCuda(cudaStreamSynchronize(ctx.workStream), "cudaStreamSynchronize(power persist superlevel)");
            }
        }
        checkCuda(cudaGetLastError(), "runPowerSolveOnlyContext dispatch");
    });

    solution.resize(static_cast<size_t>(ctx.n));
    checkCuda(cudaMemcpy(solution.data(), ctx.dSolution, static_cast<size_t>(ctx.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(power persist solution)");
    return ms;
}

struct CuSparseSolveOnlyContext {
    int n = 0;
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr;
    cusparseDnVecDescr_t vecY = nullptr;
    cusparseSpSVDescr_t spsvDescr = nullptr;
    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;
    void* dBuffer = nullptr;
};

void destroyCuSparseSolveOnlyContext(CuSparseSolveOnlyContext& ctx)
{
    if (ctx.dBuffer) cudaFree(ctx.dBuffer);
    if (ctx.spsvDescr) cusparseSpSV_destroyDescr(ctx.spsvDescr);
    if (ctx.vecY) cusparseDestroyDnVec(ctx.vecY);
    if (ctx.vecX) cusparseDestroyDnVec(ctx.vecX);
    if (ctx.matA) cusparseDestroySpMat(ctx.matA);
    if (ctx.handle) cusparseDestroy(ctx.handle);
    if (ctx.dSolution) cudaFree(ctx.dSolution);
    if (ctx.dRhs) cudaFree(ctx.dRhs);
    if (ctx.dValues) cudaFree(ctx.dValues);
    if (ctx.dColIdx) cudaFree(ctx.dColIdx);
    if (ctx.dRowPtr) cudaFree(ctx.dRowPtr);
    ctx = {};
}

CuSparseSolveOnlyContext makeCuSparseSolveOnlyContext(const CsrMatrix& matrix, const std::vector<double>& rhs)
{
    CuSparseSolveOnlyContext ctx;
    ctx.n = matrix.n;
    size_t bufferSize = 0;
    const double alpha = 1.0;

    try {
        checkCusparse(cusparseCreate(&ctx.handle), "cusparseCreate(persist)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(cusparse persist rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(cusparse persist colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(cusparse persist values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(cusparse persist rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(cusparse persist solution)");

        checkCuda(cudaMemcpy(ctx.dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(cusparse persist rowPtr)");
        checkCuda(cudaMemcpy(ctx.dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(cusparse persist colIdx)");
        checkCuda(cudaMemcpy(ctx.dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(cusparse persist values)");
        checkCuda(cudaMemcpy(ctx.dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(cusparse persist rhs)");

        checkCusparse(cusparseCreateCsr(
            &ctx.matA,
            matrix.n,
            matrix.n,
            matrix.nnz,
            ctx.dRowPtr,
            ctx.dColIdx,
            ctx.dValues,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F), "cusparseCreateCsr(persist)");
        setMatrixAttributes(ctx.matA);
        checkCusparse(cusparseCreateDnVec(&ctx.vecX, matrix.n, ctx.dRhs, CUDA_R_64F), "cusparseCreateDnVec(x persist)");
        checkCusparse(cusparseCreateDnVec(&ctx.vecY, matrix.n, ctx.dSolution, CUDA_R_64F), "cusparseCreateDnVec(y persist)");
        checkCusparse(cusparseSpSV_createDescr(&ctx.spsvDescr), "cusparseSpSV_createDescr(persist)");
        checkCusparse(cusparseSpSV_bufferSize(
            ctx.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            ctx.matA,
            ctx.vecX,
            ctx.vecY,
            CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            ctx.spsvDescr,
            &bufferSize), "cusparseSpSV_bufferSize(persist)");
        checkCuda(cudaMalloc(&ctx.dBuffer, bufferSize), "cudaMalloc(cusparse persist buffer)");
        checkCusparse(cusparseSpSV_analysis(
            ctx.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            ctx.matA,
            ctx.vecX,
            ctx.vecY,
            CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            ctx.spsvDescr,
            ctx.dBuffer), "cusparseSpSV_analysis(persist)");
        return ctx;
    } catch (...) {
        destroyCuSparseSolveOnlyContext(ctx);
        throw;
    }
}

double runCuSparseSolveOnlyContext(CuSparseSolveOnlyContext& ctx, std::vector<double>& solution)
{
    const double alpha = 1.0;
    const double ms = timeMs([&]() {
        checkCuda(cudaMemset(ctx.dSolution, 0, static_cast<size_t>(ctx.n) * sizeof(double)), "cudaMemset(cusparse persist solution)");
        checkCusparse(cusparseSpSV_solve(
            ctx.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            ctx.matA,
            ctx.vecX,
            ctx.vecY,
            CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            ctx.spsvDescr), "cusparseSpSV_solve(persist)");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(cusparse persist)");
    });

    solution.resize(static_cast<size_t>(ctx.n));
    checkCuda(cudaMemcpy(solution.data(), ctx.dSolution, static_cast<size_t>(ctx.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(cusparse persist solution)");
    return ms;
}

struct RegularSolveOnlyContext {
    int n = 0;
    bool useWarp = true;
    size_t sharedBytes = 0;
    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;
};

void destroyRegularSolveOnlyContext(RegularSolveOnlyContext& ctx)
{
    if (ctx.dSolution) cudaFree(ctx.dSolution);
    if (ctx.dRhs) cudaFree(ctx.dRhs);
    if (ctx.dValues) cudaFree(ctx.dValues);
    if (ctx.dColIdx) cudaFree(ctx.dColIdx);
    if (ctx.dRowPtr) cudaFree(ctx.dRowPtr);
    ctx = {};
}

RegularSolveOnlyContext makeRegularSolveOnlyContext(const CsrMatrix& matrix, const std::vector<double>& rhs)
{
    constexpr int kBlockSize = 256;
    RegularSolveOnlyContext ctx;
    ctx.n = matrix.n;
    ctx.sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);
    int maxRowNnz = 0;
    for (int row = 0; row < matrix.n; ++row) {
        const int rowNnz = matrix.rowPtr[static_cast<size_t>(row + 1)] - matrix.rowPtr[static_cast<size_t>(row)];
        maxRowNnz = std::max(maxRowNnz, rowNnz);
    }
    ctx.useWarp = (maxRowNnz <= 32);

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(regular persist rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(regular persist colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(regular persist values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(regular persist rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(regular persist solution)");

        checkCuda(cudaMemcpy(ctx.dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(regular persist rowPtr)");
        checkCuda(cudaMemcpy(ctx.dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(regular persist colIdx)");
        checkCuda(cudaMemcpy(ctx.dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(regular persist values)");
        checkCuda(cudaMemcpy(ctx.dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(regular persist rhs)");
        return ctx;
    } catch (...) {
        destroyRegularSolveOnlyContext(ctx);
        throw;
    }
}

double runRegularSolveOnlyContext(RegularSolveOnlyContext& ctx, std::vector<double>& solution)
{
    constexpr int kBlockSize = 256;
    const double ms = timeMs([&]() {
        checkCuda(cudaMemset(ctx.dSolution, 0, static_cast<size_t>(ctx.n) * sizeof(double)), "cudaMemset(regular persist solution)");
        if (ctx.useWarp) {
            sptrsvWarpBucketedKernel<<<1, 32>>>(0, ctx.n, ctx.dRowPtr, ctx.dColIdx, ctx.dValues, ctx.dRhs, ctx.dSolution);
        } else {
            sptrsvBlockBucketedKernel<<<1, kBlockSize, ctx.sharedBytes>>>(0, ctx.n, ctx.dRowPtr, ctx.dColIdx, ctx.dValues, ctx.dRhs, ctx.dSolution);
        }
        checkCuda(cudaGetLastError(), "runRegularSolveOnlyContext dispatch");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(regular persist)");
    });

    solution.resize(static_cast<size_t>(ctx.n));
    checkCuda(cudaMemcpy(solution.data(), ctx.dSolution, static_cast<size_t>(ctx.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(regular persist solution)");
    return ms;
}

struct CoalescedSolveOnlyContext {
    int n = 0;
    size_t sharedBytes = 0;
    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;
};

void destroyCoalescedSolveOnlyContext(CoalescedSolveOnlyContext& ctx)
{
    if (ctx.dSolution) cudaFree(ctx.dSolution);
    if (ctx.dRhs) cudaFree(ctx.dRhs);
    if (ctx.dValues) cudaFree(ctx.dValues);
    if (ctx.dColIdx) cudaFree(ctx.dColIdx);
    if (ctx.dRowPtr) cudaFree(ctx.dRowPtr);
    ctx = {};
}

CoalescedSolveOnlyContext makeCoalescedSolveOnlyContext(const CsrMatrix& matrix, const std::vector<double>& rhs)
{
    constexpr int kBlockSize = 256;
    CoalescedSolveOnlyContext ctx;
    ctx.n = matrix.n;
    ctx.sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);
    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(coalesced persist rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(coalesced persist colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(coalesced persist values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(coalesced persist rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(coalesced persist solution)");
        checkCuda(cudaMemcpy(ctx.dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(coalesced persist rowPtr)");
        checkCuda(cudaMemcpy(ctx.dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(coalesced persist colIdx)");
        checkCuda(cudaMemcpy(ctx.dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(coalesced persist values)");
        checkCuda(cudaMemcpy(ctx.dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(coalesced persist rhs)");
        return ctx;
    } catch (...) {
        destroyCoalescedSolveOnlyContext(ctx);
        throw;
    }
}

double runCoalescedSolveOnlyContext(CoalescedSolveOnlyContext& ctx, std::vector<double>& solution)
{
    constexpr int kBlockSize = 256;
    const double ms = timeMs([&]() {
        checkCuda(cudaMemset(ctx.dSolution, 0, static_cast<size_t>(ctx.n) * sizeof(double)), "cudaMemset(coalesced persist solution)");
        sptrsvCoalescedKernel<<<1, kBlockSize, ctx.sharedBytes>>>(ctx.n, ctx.dRowPtr, ctx.dColIdx, ctx.dValues, ctx.dRhs, ctx.dSolution);
        checkCuda(cudaGetLastError(), "runCoalescedSolveOnlyContext dispatch");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(coalesced persist)");
    });
    solution.resize(static_cast<size_t>(ctx.n));
    checkCuda(cudaMemcpy(solution.data(), ctx.dSolution, static_cast<size_t>(ctx.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(coalesced persist solution)");
    return ms;
}

struct RowBucketedSolveOnlyContext {
    int n = 0;
    size_t sharedBytes = 0;
    std::vector<std::tuple<int, int, int>> segments;
    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;
};

void destroyRowBucketedSolveOnlyContext(RowBucketedSolveOnlyContext& ctx)
{
    if (ctx.dSolution) cudaFree(ctx.dSolution);
    if (ctx.dRhs) cudaFree(ctx.dRhs);
    if (ctx.dValues) cudaFree(ctx.dValues);
    if (ctx.dColIdx) cudaFree(ctx.dColIdx);
    if (ctx.dRowPtr) cudaFree(ctx.dRowPtr);
    ctx = {};
}

RowBucketedSolveOnlyContext makeRowBucketedSolveOnlyContext(const CsrMatrix& matrix, const std::vector<double>& rhs)
{
    constexpr int kThreadThreshold = 4;
    constexpr int kWarpThreshold = 32;
    constexpr int kBlockSize = 256;

    RowBucketedSolveOnlyContext ctx;
    ctx.n = matrix.n;
    ctx.sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);

    auto bucketForRow = [&](int row) {
        const int rowNnz = matrix.rowPtr[static_cast<size_t>(row + 1)] - matrix.rowPtr[static_cast<size_t>(row)];
        if (rowNnz <= kThreadThreshold) return 0;
        if (rowNnz <= kWarpThreshold) return 1;
        return 2;
    };

    int segmentBegin = 0;
    int currentBucket = bucketForRow(0);
    for (int row = 1; row < matrix.n; ++row) {
        const int rowBucket = bucketForRow(row);
        if (rowBucket != currentBucket) {
            ctx.segments.emplace_back(currentBucket, segmentBegin, row);
            segmentBegin = row;
            currentBucket = rowBucket;
        }
    }
    ctx.segments.emplace_back(currentBucket, segmentBegin, matrix.n);

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(bucketed persist rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(bucketed persist colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(bucketed persist values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(bucketed persist rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ctx.dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(bucketed persist solution)");
        checkCuda(cudaMemcpy(ctx.dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(bucketed persist rowPtr)");
        checkCuda(cudaMemcpy(ctx.dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(bucketed persist colIdx)");
        checkCuda(cudaMemcpy(ctx.dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(bucketed persist values)");
        checkCuda(cudaMemcpy(ctx.dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(bucketed persist rhs)");
        return ctx;
    } catch (...) {
        destroyRowBucketedSolveOnlyContext(ctx);
        throw;
    }
}

double runRowBucketedSolveOnlyContext(RowBucketedSolveOnlyContext& ctx, std::vector<double>& solution)
{
    constexpr int kBlockSize = 256;
    const double ms = timeMs([&]() {
        checkCuda(cudaMemset(ctx.dSolution, 0, static_cast<size_t>(ctx.n) * sizeof(double)), "cudaMemset(bucketed persist solution)");
        for (const auto& [bucket, rowBegin, rowEnd] : ctx.segments) {
            if (bucket == 0) {
                sptrsvThreadBucketedKernel<<<1, 1>>>(rowBegin, rowEnd, ctx.dRowPtr, ctx.dColIdx, ctx.dValues, ctx.dRhs, ctx.dSolution);
            } else if (bucket == 1) {
                sptrsvWarpBucketedKernel<<<1, 32>>>(rowBegin, rowEnd, ctx.dRowPtr, ctx.dColIdx, ctx.dValues, ctx.dRhs, ctx.dSolution);
            } else {
                sptrsvBlockBucketedKernel<<<1, kBlockSize, ctx.sharedBytes>>>(rowBegin, rowEnd, ctx.dRowPtr, ctx.dColIdx, ctx.dValues, ctx.dRhs, ctx.dSolution);
            }
        }
        checkCuda(cudaGetLastError(), "runRowBucketedSolveOnlyContext dispatch");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(bucketed persist)");
    });
    solution.resize(static_cast<size_t>(ctx.n));
    checkCuda(cudaMemcpy(solution.data(), ctx.dSolution, static_cast<size_t>(ctx.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(bucketed persist solution)");
    return ms;
}

double sptrsvGpuNamedPersistSolveOnly(const std::string& kernel, const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    if (kernel == "RandomAware") {
        RandomSolveOnlyContext ctx = makeRandomSolveOnlyContext(matrix, rhs);
        try {
            const double ms = runRandomSolveOnlyContext(ctx, solution);
            destroyRandomSolveOnlyContext(ctx);
            return ms;
        } catch (...) {
            destroyRandomSolveOnlyContext(ctx);
            throw;
        }
    }
    if (kernel == "PowerLawAware") {
        PowerSolveOnlyContext ctx = makePowerSolveOnlyContext(matrix, rhs);
        try {
            const double ms = runPowerSolveOnlyContext(ctx, solution);
            destroyPowerSolveOnlyContext(ctx);
            return ms;
        } catch (...) {
            destroyPowerSolveOnlyContext(ctx);
            throw;
        }
    }
    if (kernel == "RegularAware") {
        RegularSolveOnlyContext ctx = makeRegularSolveOnlyContext(matrix, rhs);
        try {
            const double ms = runRegularSolveOnlyContext(ctx, solution);
            destroyRegularSolveOnlyContext(ctx);
            return ms;
        } catch (...) {
            destroyRegularSolveOnlyContext(ctx);
            throw;
        }
    }
    if (kernel == "Coalesced") {
        CoalescedSolveOnlyContext ctx = makeCoalescedSolveOnlyContext(matrix, rhs);
        try {
            const double ms = runCoalescedSolveOnlyContext(ctx, solution);
            destroyCoalescedSolveOnlyContext(ctx);
            return ms;
        } catch (...) {
            destroyCoalescedSolveOnlyContext(ctx);
            throw;
        }
    }
    if (kernel == "BandedAware") {
        if (useCoalescedForBandedAware(matrix)) {
            CoalescedSolveOnlyContext ctx = makeCoalescedSolveOnlyContext(matrix, rhs);
            try {
                const double ms = runCoalescedSolveOnlyContext(ctx, solution);
                destroyCoalescedSolveOnlyContext(ctx);
                return ms;
            } catch (...) {
                destroyCoalescedSolveOnlyContext(ctx);
                throw;
            }
        }
        RowBucketedSolveOnlyContext ctx = makeRowBucketedSolveOnlyContext(matrix, rhs);
        try {
            const double ms = runRowBucketedSolveOnlyContext(ctx, solution);
            destroyRowBucketedSolveOnlyContext(ctx);
            return ms;
        } catch (...) {
            destroyRowBucketedSolveOnlyContext(ctx);
            throw;
        }
    }
    if (kernel == "RowBucketed") {
        RowBucketedSolveOnlyContext ctx = makeRowBucketedSolveOnlyContext(matrix, rhs);
        try {
            const double ms = runRowBucketedSolveOnlyContext(ctx, solution);
            destroyRowBucketedSolveOnlyContext(ctx);
            return ms;
        } catch (...) {
            destroyRowBucketedSolveOnlyContext(ctx);
            throw;
        }
    }
    throw std::runtime_error("Learned persistent solve-only mode currently supports RandomAware, PowerLawAware, RegularAware, Coalesced, BandedAware, and RowBucketed only");
}

}  // namespace

using LearnedFeatureVector = std::array<double, kLearnedFeatureCount>;

CsrMatrix generateLowerTriangular(int n, double density, unsigned seed)
{
    CsrMatrix matrix;
    matrix.n = n;
    matrix.rowPtr.resize(static_cast<size_t>(n + 1), 0);

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
        matrix.rowPtr[static_cast<size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
    }

    matrix.nnz = static_cast<int>(matrix.colIdx.size());
    return matrix;
}

CsrMatrix generateBandedLowerTriangular(int n, int bandwidth)
{
    CsrMatrix matrix;
    matrix.n = n;
    matrix.rowPtr.resize(static_cast<size_t>(n + 1), 0);

    for (int row = 0; row < n; ++row) {
        const int startCol = std::max(0, row - bandwidth);
        for (int col = startCol; col < row; ++col) {
            matrix.colIdx.push_back(col);
            matrix.values.push_back(0.25 + 0.01 * static_cast<double>((row - col) % 13));
        }
        matrix.colIdx.push_back(row);
        matrix.values.push_back(static_cast<double>(bandwidth) + 2.0);
        matrix.rowPtr[static_cast<size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
    }

    matrix.nnz = static_cast<int>(matrix.colIdx.size());
    return matrix;
}

CsrMatrix generateChainLowerTriangular(int n)
{
    CsrMatrix matrix;
    matrix.n = n;
    matrix.rowPtr.resize(static_cast<size_t>(n + 1), 0);

    for (int row = 0; row < n; ++row) {
        if (row > 0) {
            matrix.colIdx.push_back(row - 1);
            matrix.values.push_back(0.5);
        }
        matrix.colIdx.push_back(row);
        matrix.values.push_back(2.0);
        matrix.rowPtr[static_cast<size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
    }

    matrix.nnz = static_cast<int>(matrix.colIdx.size());
    return matrix;
}

CsrMatrix generateWideLevelLowerTriangular(int n, int prefixWidth)
{
    CsrMatrix matrix;
    matrix.n = n;
    matrix.rowPtr.resize(static_cast<size_t>(n + 1), 0);

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
        matrix.rowPtr[static_cast<size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
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
    matrix.rowPtr.resize(static_cast<size_t>(n + 1), 0);

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
        matrix.rowPtr[static_cast<size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
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
    matrix.rowPtr.resize(static_cast<size_t>(n + 1), 0);

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
        matrix.rowPtr[static_cast<size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
    }

    matrix.nnz = static_cast<int>(matrix.colIdx.size());
    return matrix;
}

std::vector<double> generateRandomRhs(int n, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> rhs(static_cast<size_t>(n));
    for (double& value : rhs) {
        value = dist(rng);
    }
    return rhs;
}

void sptrsvCpuReference(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    solution.assign(static_cast<size_t>(matrix.n), 0.0);

    for (int row = 0; row < matrix.n; ++row) {
        double rowRhs = rhs[static_cast<size_t>(row)];
        double diagonal = 1.0;

        for (int entry = matrix.rowPtr[static_cast<size_t>(row)];
             entry < matrix.rowPtr[static_cast<size_t>(row + 1)];
             ++entry) {
            const int col = matrix.colIdx[static_cast<size_t>(entry)];
            const double value = matrix.values[static_cast<size_t>(entry)];
            if (col < row) {
                rowRhs -= value * solution[static_cast<size_t>(col)];
            } else if (col == row) {
                diagonal = value;
            }
        }

        solution[static_cast<size_t>(row)] = rowRhs / diagonal;
    }
}

double computeResidual(const CsrMatrix& matrix, const std::vector<double>& solution, const std::vector<double>& rhs)
{
    double residual = 0.0;
    for (int row = 0; row < matrix.n; ++row) {
        double dot = 0.0;
        for (int entry = matrix.rowPtr[static_cast<size_t>(row)];
             entry < matrix.rowPtr[static_cast<size_t>(row + 1)];
             ++entry) {
            dot += matrix.values[static_cast<size_t>(entry)] *
                   solution[static_cast<size_t>(matrix.colIdx[static_cast<size_t>(entry)])];
        }
        const double diff = dot - rhs[static_cast<size_t>(row)];
        residual += diff * diff;
    }
    return std::sqrt(residual);
}

std::vector<int> computeRowLevels(const CsrMatrix& matrix)
{
    std::vector<int> levels(static_cast<size_t>(matrix.n), 0);
    for (int row = 0; row < matrix.n; ++row) {
        int level = 0;
        for (int entry = matrix.rowPtr[static_cast<size_t>(row)];
             entry < matrix.rowPtr[static_cast<size_t>(row + 1)];
             ++entry) {
            const int col = matrix.colIdx[static_cast<size_t>(entry)];
            if (col < row) {
                level = std::max(level, levels[static_cast<size_t>(col)] + 1);
            }
        }
        levels[static_cast<size_t>(row)] = level;
    }
    return levels;
}

int estimateBandwidth(const CsrMatrix& matrix)
{
    int bandwidth = 0;
    for (int row = 0; row < matrix.n; ++row) {
        for (int entry = matrix.rowPtr[static_cast<size_t>(row)];
             entry < matrix.rowPtr[static_cast<size_t>(row + 1)];
             ++entry) {
            const int col = matrix.colIdx[static_cast<size_t>(entry)];
            if (col < row) {
                bandwidth = std::max(bandwidth, row - col);
            }
        }
    }
    return bandwidth;
}

struct BcsrMatrix {
    int n = 0;
    int blockSize = 0;
    int numBlockRows = 0;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<double> values;
};

struct BlockAnalysis {
    int blockSize = 0;
    int numBlocks = 0;
    int storedEntries = 0;
    int realEntries = 0;
    double occupancy = 0.0;
};

std::uint64_t makeBlockKey(int blockRow, int blockCol)
{
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(blockRow)) << 32) |
           static_cast<std::uint32_t>(blockCol);
}

BlockAnalysis analyzeBlockSize(const CsrMatrix& matrix, int blockSize)
{
    std::unordered_map<std::uint64_t, std::uint64_t> blockMasks;
    blockMasks.reserve(static_cast<size_t>(matrix.nnz));

    for (int row = 0; row < matrix.n; ++row) {
        for (int entry = matrix.rowPtr[static_cast<size_t>(row)];
             entry < matrix.rowPtr[static_cast<size_t>(row + 1)];
             ++entry) {
            const int col = matrix.colIdx[static_cast<size_t>(entry)];
            const int blockRow = row / blockSize;
            const int blockCol = col / blockSize;
            const int localRow = row % blockSize;
            const int localCol = col % blockSize;
            const int bit = localRow * blockSize + localCol;
            blockMasks[makeBlockKey(blockRow, blockCol)] |= (std::uint64_t{1} << bit);
        }
    }

    int realEntries = 0;
    for (const auto& [_, mask] : blockMasks) {
        realEntries += __builtin_popcountll(mask);
    }

    BlockAnalysis analysis;
    analysis.blockSize = blockSize;
    analysis.numBlocks = static_cast<int>(blockMasks.size());
    analysis.realEntries = realEntries;
    analysis.storedEntries = analysis.numBlocks * blockSize * blockSize;
    analysis.occupancy = (analysis.storedEntries > 0)
        ? static_cast<double>(analysis.realEntries) / static_cast<double>(analysis.storedEntries)
        : 0.0;
    return analysis;
}

BlockAnalysis chooseBlockSize(const CsrMatrix& matrix)
{
    const BlockAnalysis bs2 = analyzeBlockSize(matrix, 2);
    const BlockAnalysis bs4 = analyzeBlockSize(matrix, 4);
    const BlockAnalysis bs8 = analyzeBlockSize(matrix, 8);

    BlockAnalysis best = bs2;
    for (const BlockAnalysis candidate : {bs4, bs8}) {
        if (candidate.occupancy > best.occupancy + 0.03 ||
            (std::abs(candidate.occupancy - best.occupancy) <= 0.03 &&
             candidate.blockSize > best.blockSize)) {
            best = candidate;
        }
    }
    return best;
}

BcsrMatrix convertToBcsr(const CsrMatrix& matrix, int blockSize)
{
    const int numBlockRows = (matrix.n + blockSize - 1) / blockSize;
    std::vector<std::unordered_map<int, std::vector<double>>> rows(static_cast<size_t>(numBlockRows));

    for (int row = 0; row < matrix.n; ++row) {
        const int blockRow = row / blockSize;
        const int localRow = row % blockSize;
        for (int entry = matrix.rowPtr[static_cast<size_t>(row)];
             entry < matrix.rowPtr[static_cast<size_t>(row + 1)];
             ++entry) {
            const int col = matrix.colIdx[static_cast<size_t>(entry)];
            const int blockCol = col / blockSize;
            const int localCol = col % blockSize;
            auto& block = rows[static_cast<size_t>(blockRow)][blockCol];
            if (block.empty()) {
                block.assign(static_cast<size_t>(blockSize * blockSize), 0.0);
            }
            block[static_cast<size_t>(localRow * blockSize + localCol)] = matrix.values[static_cast<size_t>(entry)];
        }
    }

    BcsrMatrix bcsr;
    bcsr.n = matrix.n;
    bcsr.blockSize = blockSize;
    bcsr.numBlockRows = numBlockRows;
    bcsr.rowPtr.resize(static_cast<size_t>(numBlockRows + 1), 0);

    for (int blockRow = 0; blockRow < numBlockRows; ++blockRow) {
        auto cols = std::vector<int>();
        cols.reserve(rows[static_cast<size_t>(blockRow)].size());
        for (const auto& [blockCol, _] : rows[static_cast<size_t>(blockRow)]) {
            cols.push_back(blockCol);
        }
        std::sort(cols.begin(), cols.end());

        for (int blockCol : cols) {
            bcsr.colIdx.push_back(blockCol);
            const auto& block = rows[static_cast<size_t>(blockRow)][blockCol];
            bcsr.values.insert(bcsr.values.end(), block.begin(), block.end());
        }
        bcsr.rowPtr[static_cast<size_t>(blockRow + 1)] = static_cast<int>(bcsr.colIdx.size());
    }

    return bcsr;
}

std::vector<int> computeBcsrBlockLevels(const BcsrMatrix& bcsr)
{
    std::vector<int> levels(static_cast<size_t>(bcsr.numBlockRows), 0);
    for (int blockRow = 0; blockRow < bcsr.numBlockRows; ++blockRow) {
        int level = 0;
        for (int entry = bcsr.rowPtr[static_cast<size_t>(blockRow)];
             entry < bcsr.rowPtr[static_cast<size_t>(blockRow + 1)];
             ++entry) {
            const int blockCol = bcsr.colIdx[static_cast<size_t>(entry)];
            if (blockCol < blockRow) {
                level = std::max(level, levels[static_cast<size_t>(blockCol)] + 1);
            }
        }
        levels[static_cast<size_t>(blockRow)] = level;
    }
    return levels;
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

struct MatrixFeatures {
    double avgRowNnz = 0.0;
    double coeffVarRowNnz = 0.0;
    double adjacentFrac = 0.0;
    int maxRowNnz = 0;
    int bandwidth = 0;
};

struct FamilyChoice {
    std::string family;
    std::string kernel;
};

MatrixFeatures analyzeMatrixFeatures(const CsrMatrix& matrix)
{
    MatrixFeatures features;
    if (matrix.n <= 0) {
        return features;
    }

    std::vector<int> rowNnz(static_cast<size_t>(matrix.n), 0);
    double sum = 0.0;
    std::uint64_t adjacentPairs = 0;
    std::uint64_t comparablePairs = 0;
    for (int row = 0; row < matrix.n; ++row) {
        const int rowBegin = matrix.rowPtr[static_cast<size_t>(row)];
        const int rowEnd = matrix.rowPtr[static_cast<size_t>(row + 1)];
        const int nnz = rowEnd - rowBegin;
        rowNnz[static_cast<size_t>(row)] = nnz;
        features.maxRowNnz = std::max(features.maxRowNnz, nnz);
        sum += static_cast<double>(nnz);

        int prevOffdiagCol = -2;
        bool havePrevOffdiag = false;
        for (int entry = rowBegin; entry < rowEnd; ++entry) {
            const int col = matrix.colIdx[static_cast<size_t>(entry)];
            if (col >= row) {
                continue;
            }
            if (havePrevOffdiag) {
                ++comparablePairs;
                if (col == prevOffdiagCol + 1) {
                    ++adjacentPairs;
                }
            }
            prevOffdiagCol = col;
            havePrevOffdiag = true;
        }
    }
    features.avgRowNnz = sum / static_cast<double>(matrix.n);

    double variance = 0.0;
    for (int nnz : rowNnz) {
        const double diff = static_cast<double>(nnz) - features.avgRowNnz;
        variance += diff * diff;
    }
    variance /= static_cast<double>(matrix.n);
    const double stddev = std::sqrt(variance);
    features.coeffVarRowNnz = (features.avgRowNnz > 0.0) ? (stddev / features.avgRowNnz) : 0.0;
    features.adjacentFrac = (comparablePairs > 0)
        ? static_cast<double>(adjacentPairs) / static_cast<double>(comparablePairs)
        : 0.0;

    features.bandwidth = estimateBandwidth(matrix);
    return features;
}

FamilyChoice chooseFamilyKernel(const CsrMatrix& matrix)
{
    const MatrixFeatures f = analyzeMatrixFeatures(matrix);

    if (f.maxRowNnz <= 2 && f.bandwidth <= 1) {
        return {"chain", "RowBucketed"};
    }
    if (f.adjacentFrac >= 0.95 && f.bandwidth <= 16 && f.maxRowNnz <= 8) {
        return {"banded", "RegularAware"};
    }
    if (f.avgRowNnz <= 6.0 && f.coeffVarRowNnz <= 0.12 && f.maxRowNnz <= 8) {
        return {"regular", "RegularAware"};
    }
    if (f.bandwidth <= 96 && f.avgRowNnz >= 32.0 && f.coeffVarRowNnz < 0.20) {
        return {"banded", "BandedAware"};
    }
    if (f.coeffVarRowNnz >= 0.35 && f.avgRowNnz <= 4.0) {
        return {"power-law", "PowerLawAware"};
    }
    if (f.bandwidth >= matrix.n / 4 && f.avgRowNnz >= 16.0 && f.avgRowNnz <= 96.0 &&
        f.coeffVarRowNnz >= 0.15 && f.coeffVarRowNnz <= 0.80 && f.adjacentFrac >= 0.20) {
        return {"block", "RowBucketed"};
    }
    if (f.bandwidth >= matrix.n / 4 && f.avgRowNnz >= 16.0 && f.avgRowNnz <= 48.0 && f.coeffVarRowNnz < 0.15) {
        return {"wide-level", "RandomAware"};
    }
    return {"random", "RandomAware"};
}

void printMatrixFeatures(const CsrMatrix& matrix)
{
    const MatrixFeatures f = analyzeMatrixFeatures(matrix);
    const BlockAnalysis block = chooseBlockSize(matrix);
    std::cout << "avgRowNnz      : " << std::fixed << std::setprecision(4) << f.avgRowNnz << "\n";
    std::cout << "coeffVarRowNnz : " << std::fixed << std::setprecision(4) << f.coeffVarRowNnz << "\n";
    std::cout << "adjacentFrac   : " << std::fixed << std::setprecision(4) << f.adjacentFrac << "\n";
    std::cout << "maxRowNnz      : " << f.maxRowNnz << "\n";
    std::cout << "bandwidth      : " << f.bandwidth << "\n";
    std::cout << "blockSize      : " << block.blockSize << "\n";
    std::cout << "blockOccupancy : " << std::fixed << std::setprecision(4) << block.occupancy << "\n";
}

LearnedFeatureVector makeLearnedFeatures(const CsrMatrix& matrix)
{
    const MatrixFeatures f = analyzeMatrixFeatures(matrix);
    const double avg = std::max(1e-12, f.avgRowNnz);
    return {
        std::log1p(static_cast<double>(matrix.n)),
        std::log1p(static_cast<double>(matrix.nnz)),
        f.avgRowNnz,
        f.coeffVarRowNnz,
        f.adjacentFrac,
        static_cast<double>(f.maxRowNnz) / avg,
        static_cast<double>(f.bandwidth) / std::max(1.0, static_cast<double>(matrix.n))
    };
}

std::string chooseLearnedKernel(const CsrMatrix& matrix)
{
    const MatrixFeatures f = analyzeMatrixFeatures(matrix);
    if (f.adjacentFrac >= 0.95 && f.bandwidth <= 16 && f.maxRowNnz <= 8) {
        return "BandedAware";
    }
    if (!kLearnedSelectorReady) {
        return chooseFamilyKernel(matrix).kernel;
    }

    const LearnedFeatureVector raw = makeLearnedFeatures(matrix);
    std::array<double, kLearnedFeatureCount> x{};
    for (int i = 0; i < kLearnedFeatureCount; ++i) {
        x[static_cast<size_t>(i)] = (raw[static_cast<size_t>(i)] - kLearnedFeatureMean[static_cast<size_t>(i)]) /
                                    kLearnedFeatureStd[static_cast<size_t>(i)];
    }

    int bestClass = 0;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (int cls = 0; cls < kLearnedClassCount; ++cls) {
        double score = kLearnedBiases[static_cast<size_t>(cls)];
        for (int i = 0; i < kLearnedFeatureCount; ++i) {
            score += kLearnedWeights[static_cast<size_t>(cls)][static_cast<size_t>(i)] * x[static_cast<size_t>(i)];
        }
        if (score > bestScore) {
            bestScore = score;
            bestClass = cls;
        }
    }
    return kLearnedClassNames[static_cast<size_t>(bestClass)];
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
        throw std::runtime_error("Unsupported Matrix Market header in " + path);
    }

    do {
        if (!std::getline(in, line)) {
            throw std::runtime_error("Missing size line in " + path);
        }
    } while (!line.empty() && line[0] == '%');

    std::istringstream dims(line);
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    dims >> rows >> cols >> nnz;
    if (rows <= 0 || cols <= 0 || nnz < 0 || rows != cols) {
        throw std::runtime_error("Expected square Matrix Market matrix in " + path);
    }

    std::vector<std::vector<std::pair<int, double>>> rowEntries(static_cast<size_t>(rows));
    for (int k = 0; k < nnz; ++k) {
        int i = 0;
        int j = 0;
        double value = 1.0;

        if (!(in >> i >> j)) {
            throw std::runtime_error("Failed to read coordinates in " + path);
        }
        if (field == "real" || field == "integer") {
            if (!(in >> value)) {
                throw std::runtime_error("Failed to read value in " + path);
            }
        } else if (field == "pattern") {
            value = 1.0;
        } else {
            throw std::runtime_error("Unsupported field type in " + path);
        }

        --i;
        --j;
        if (i < 0 || i >= rows || j < 0 || j >= cols) {
            throw std::runtime_error("Out-of-range coordinate in " + path);
        }

        if (i >= j) {
            rowEntries[static_cast<size_t>(i)].push_back({j, value});
        }
        if (symmetry == "symmetric" && i != j && j >= i) {
            rowEntries[static_cast<size_t>(j)].push_back({i, value});
        }
    }

    CsrMatrix matrix;
    matrix.n = rows;
    matrix.rowPtr.resize(static_cast<size_t>(rows + 1), 0);

    for (int row = 0; row < rows; ++row) {
        auto& entries = rowEntries[static_cast<size_t>(row)];
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<std::pair<int, double>> merged;
        merged.reserve(entries.size());
        for (const auto& [col, value] : entries) {
            if (!merged.empty() && merged.back().first == col) {
                merged.back().second += value;
            } else {
                merged.push_back({col, value});
            }
        }

        double offDiagAbsSum = 0.0;
        double diagonal = 0.0;
        bool hasDiagonal = false;
        for (const auto& [col, value] : merged) {
            if (col < row) {
                offDiagAbsSum += std::abs(value);
                matrix.colIdx.push_back(col);
                matrix.values.push_back(value);
            } else if (col == row) {
                diagonal += value;
                hasDiagonal = true;
            }
        }

        double safeDiagonal = hasDiagonal ? diagonal : 0.0;
        if (std::abs(safeDiagonal) <= offDiagAbsSum) {
            safeDiagonal = (safeDiagonal >= 0.0 ? 1.0 : -1.0) * (offDiagAbsSum + 1.0);
        } else if (std::abs(safeDiagonal) < 1e-12) {
            safeDiagonal = offDiagAbsSum + 1.0;
        }

        matrix.colIdx.push_back(row);
        matrix.values.push_back(safeDiagonal);
        matrix.rowPtr[static_cast<size_t>(row + 1)] = static_cast<int>(matrix.colIdx.size());
    }

    matrix.nnz = static_cast<int>(matrix.colIdx.size());
    return matrix;
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
        {"very_large", "wiki-Talk", "data/matrices/very_large/wiki-Talk/wiki-Talk.mtx"},
    };
}

void runRealMatrixSweep()
{
    const std::vector<RealMatrixCase> cases = makeRealMatrixCases();

    std::cout << "GPU New Real Matrix Sweep\n";
    std::cout << "Preprocessing: Matrix Market -> lower triangular CSR with strengthened diagonal\n";
    printDeviceInfo();
    std::cout << "\n";

    std::cout << std::left
              << std::setw(18) << "Family"
              << std::setw(20) << "Matrix"
              << std::setw(10) << "N"
              << std::setw(12) << "NNZ"
              << std::setw(14) << "AutoFam"
              << std::setw(14) << "AutoKern"
              << std::setw(16) << "FamilyAware"
              << std::setw(16) << "LearnedAware"
              << std::setw(16) << "cuSPARSE"
              << std::setw(16) << "Ratio"
              << "\n";
    std::cout << std::string(152, '-') << "\n";

    for (const auto& c : cases) {
        const CsrMatrix matrix = loadMatrixMarketAsLowerTriangular(c.path);
        const std::vector<double> rhs = generateRandomRhs(matrix.n, 99);
        const FamilyChoice choice = chooseFamilyKernel(matrix);

        std::vector<double> familyAwareSolution;
        std::vector<double> learnedAwareSolution;
        std::vector<double> cusparseSolution;
        bool familyAwareOk = false;
        bool learnedAwareOk = false;
        bool cusparseOk = false;

        const double familyAwareMs = timeMs([&]() { familyAwareOk = sptrsvGpuFamilyAware(matrix, rhs, familyAwareSolution); });
        const double learnedAwareMs = timeMs([&]() { learnedAwareOk = sptrsvGpuLearnedAware(matrix, rhs, learnedAwareSolution); });
        const double cusparseMs = timeMs([&]() { cusparseOk = sptrsvCuSparse(matrix, rhs, cusparseSolution); });

        std::cout << std::setw(18) << c.family
                  << std::setw(20) << c.name
                  << std::setw(10) << matrix.n
                  << std::setw(12) << matrix.nnz
                  << std::setw(14) << choice.family
                  << std::setw(14) << choice.kernel;
        if (familyAwareOk) {
            std::cout << std::setw(16) << std::fixed << std::setprecision(4) << familyAwareMs;
        } else {
            std::cout << std::setw(16) << "failed";
        }
        if (learnedAwareOk) {
            std::cout << std::setw(16) << std::fixed << std::setprecision(4) << learnedAwareMs;
        } else {
            std::cout << std::setw(16) << "failed";
        }
        if (cusparseOk) {
            std::cout << std::setw(16) << std::fixed << std::setprecision(4) << cusparseMs;
        } else {
            std::cout << std::setw(16) << "failed";
        }
        if (learnedAwareOk && cusparseOk && cusparseMs > 0.0) {
            std::cout << std::setw(16) << std::fixed << std::setprecision(4) << (learnedAwareMs / cusparseMs);
        } else {
            std::cout << std::setw(16) << "-";
        }
        std::cout << "\n";
    }
}

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

bool sptrsvGpuNaive(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(solution)");

        checkCuda(cudaMemcpy(dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(rowPtr)");
        checkCuda(cudaMemcpy(dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(colIdx)");
        checkCuda(cudaMemcpy(dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(values)");
        checkCuda(cudaMemcpy(dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(rhs)");
        checkCuda(cudaMemset(dSolution, 0, static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMemset(solution)");

        sptrsvNaiveKernel<<<1, 1>>>(matrix.n, dRowPtr, dColIdx, dValues, dRhs, dSolution);
        checkCuda(cudaGetLastError(), "sptrsvNaiveKernel");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(naive)");

        solution.resize(static_cast<size_t>(matrix.n));
        checkCuda(cudaMemcpy(solution.data(), dSolution, static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(solution)");
    } catch (...) {
        if (dSolution) cudaFree(dSolution);
        if (dRhs) cudaFree(dRhs);
        if (dValues) cudaFree(dValues);
        if (dColIdx) cudaFree(dColIdx);
        if (dRowPtr) cudaFree(dRowPtr);
        return false;
    }

    cudaFree(dSolution);
    cudaFree(dRhs);
    cudaFree(dValues);
    cudaFree(dColIdx);
    cudaFree(dRowPtr);
    return true;
}

bool sptrsvGpuCoalesced(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    constexpr int kBlockSize = 256;

    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(coalesced rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(coalesced colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(coalesced values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(coalesced rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(coalesced solution)");

        checkCuda(cudaMemcpy(dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(coalesced rowPtr)");
        checkCuda(cudaMemcpy(dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(coalesced colIdx)");
        checkCuda(cudaMemcpy(dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(coalesced values)");
        checkCuda(cudaMemcpy(dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(coalesced rhs)");
        checkCuda(cudaMemset(dSolution, 0, static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMemset(coalesced solution)");

        const size_t sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);
        sptrsvCoalescedKernel<<<1, kBlockSize, sharedBytes>>>(matrix.n, dRowPtr, dColIdx, dValues, dRhs, dSolution);
        checkCuda(cudaGetLastError(), "sptrsvCoalescedKernel");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(coalesced)");

        solution.resize(static_cast<size_t>(matrix.n));
        checkCuda(cudaMemcpy(solution.data(), dSolution, static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(coalesced solution)");
    } catch (...) {
        if (dSolution) cudaFree(dSolution);
        if (dRhs) cudaFree(dRhs);
        if (dValues) cudaFree(dValues);
        if (dColIdx) cudaFree(dColIdx);
        if (dRowPtr) cudaFree(dRowPtr);
        return false;
    }

    cudaFree(dSolution);
    cudaFree(dRhs);
    cudaFree(dValues);
    cudaFree(dColIdx);
    cudaFree(dRowPtr);
    return true;
}

bool sptrsvGpuThreadPerRow(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(thread rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(thread colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(thread values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(thread rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(thread solution)");

        checkCuda(cudaMemcpy(dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(thread rowPtr)");
        checkCuda(cudaMemcpy(dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(thread colIdx)");
        checkCuda(cudaMemcpy(dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(thread values)");
        checkCuda(cudaMemcpy(dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(thread rhs)");
        checkCuda(cudaMemset(dSolution, 0, static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMemset(thread solution)");

        for (int row = 0; row < matrix.n; ++row) {
            sptrsvThreadPerRowKernel<<<1, 1>>>(row, dRowPtr, dColIdx, dValues, dRhs, dSolution);
        }
        checkCuda(cudaGetLastError(), "sptrsvThreadPerRowKernel");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(thread)");

        solution.resize(static_cast<size_t>(matrix.n));
        checkCuda(cudaMemcpy(solution.data(), dSolution, static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(thread solution)");
    } catch (...) {
        if (dSolution) cudaFree(dSolution);
        if (dRhs) cudaFree(dRhs);
        if (dValues) cudaFree(dValues);
        if (dColIdx) cudaFree(dColIdx);
        if (dRowPtr) cudaFree(dRowPtr);
        return false;
    }

    cudaFree(dSolution);
    cudaFree(dRhs);
    cudaFree(dValues);
    cudaFree(dColIdx);
    cudaFree(dRowPtr);
    return true;
}

bool sptrsvGpuWarpPerRow(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(warp rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(warp colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(warp values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(warp rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(warp solution)");

        checkCuda(cudaMemcpy(dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(warp rowPtr)");
        checkCuda(cudaMemcpy(dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(warp colIdx)");
        checkCuda(cudaMemcpy(dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(warp values)");
        checkCuda(cudaMemcpy(dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(warp rhs)");
        checkCuda(cudaMemset(dSolution, 0, static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMemset(warp solution)");

        for (int row = 0; row < matrix.n; ++row) {
            sptrsvWarpPerRowKernel<<<1, 32>>>(row, dRowPtr, dColIdx, dValues, dRhs, dSolution);
        }
        checkCuda(cudaGetLastError(), "sptrsvWarpPerRowKernel");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warp)");

        solution.resize(static_cast<size_t>(matrix.n));
        checkCuda(cudaMemcpy(solution.data(), dSolution, static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(warp solution)");
    } catch (...) {
        if (dSolution) cudaFree(dSolution);
        if (dRhs) cudaFree(dRhs);
        if (dValues) cudaFree(dValues);
        if (dColIdx) cudaFree(dColIdx);
        if (dRowPtr) cudaFree(dRowPtr);
        return false;
    }

    cudaFree(dSolution);
    cudaFree(dRhs);
    cudaFree(dValues);
    cudaFree(dColIdx);
    cudaFree(dRowPtr);
    return true;
}

bool sptrsvGpuAdaptiveRow(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    constexpr int kThreadThreshold = 4;
    constexpr int kWarpThreshold = 32;
    constexpr int kBlockSize = 256;

    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(adaptive rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(adaptive colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(adaptive values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(adaptive rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(adaptive solution)");

        checkCuda(cudaMemcpy(dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(adaptive rowPtr)");
        checkCuda(cudaMemcpy(dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(adaptive colIdx)");
        checkCuda(cudaMemcpy(dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(adaptive values)");
        checkCuda(cudaMemcpy(dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(adaptive rhs)");
        checkCuda(cudaMemset(dSolution, 0, static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMemset(adaptive solution)");

        const size_t sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);
        for (int row = 0; row < matrix.n; ++row) {
            const int rowNnz = matrix.rowPtr[static_cast<size_t>(row + 1)] - matrix.rowPtr[static_cast<size_t>(row)];
            if (rowNnz <= kThreadThreshold) {
                sptrsvThreadPerRowKernel<<<1, 1>>>(row, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            } else if (rowNnz <= kWarpThreshold) {
                sptrsvWarpPerRowKernel<<<1, 32>>>(row, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            } else {
                sptrsvBlockPerRowKernel<<<1, kBlockSize, sharedBytes>>>(row, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            }
        }
        checkCuda(cudaGetLastError(), "sptrsvGpuAdaptiveRow dispatch");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(adaptive)");

        solution.resize(static_cast<size_t>(matrix.n));
        checkCuda(cudaMemcpy(solution.data(), dSolution, static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(adaptive solution)");
    } catch (...) {
        if (dSolution) cudaFree(dSolution);
        if (dRhs) cudaFree(dRhs);
        if (dValues) cudaFree(dValues);
        if (dColIdx) cudaFree(dColIdx);
        if (dRowPtr) cudaFree(dRowPtr);
        return false;
    }

    cudaFree(dSolution);
    cudaFree(dRhs);
    cudaFree(dValues);
    cudaFree(dColIdx);
    cudaFree(dRowPtr);
    return true;
}

bool sptrsvGpuBlockPerRow(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    constexpr int kBlockSize = 256;

    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(block rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(block colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(block values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(block rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(block solution)");

        checkCuda(cudaMemcpy(dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(block rowPtr)");
        checkCuda(cudaMemcpy(dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(block colIdx)");
        checkCuda(cudaMemcpy(dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(block values)");
        checkCuda(cudaMemcpy(dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(block rhs)");
        checkCuda(cudaMemset(dSolution, 0, static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMemset(block solution)");

        const size_t sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);
        for (int row = 0; row < matrix.n; ++row) {
            sptrsvBlockPerRowKernel<<<1, kBlockSize, sharedBytes>>>(row, dRowPtr, dColIdx, dValues, dRhs, dSolution);
        }
        checkCuda(cudaGetLastError(), "sptrsvBlockPerRowKernel");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(block)");

        solution.resize(static_cast<size_t>(matrix.n));
        checkCuda(cudaMemcpy(solution.data(), dSolution, static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(block solution)");
    } catch (...) {
        if (dSolution) cudaFree(dSolution);
        if (dRhs) cudaFree(dRhs);
        if (dValues) cudaFree(dValues);
        if (dColIdx) cudaFree(dColIdx);
        if (dRowPtr) cudaFree(dRowPtr);
        return false;
    }

    cudaFree(dSolution);
    cudaFree(dRhs);
    cudaFree(dValues);
    cudaFree(dColIdx);
    cudaFree(dRowPtr);
    return true;
}

bool sptrsvGpuRowBucketed(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    constexpr int kThreadThreshold = 4;
    constexpr int kWarpThreshold = 32;
    constexpr int kBlockSize = 256;

    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(bucketed rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(bucketed colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(bucketed values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(bucketed rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(bucketed solution)");

        checkCuda(cudaMemcpy(dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(bucketed rowPtr)");
        checkCuda(cudaMemcpy(dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(bucketed colIdx)");
        checkCuda(cudaMemcpy(dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(bucketed values)");
        checkCuda(cudaMemcpy(dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(bucketed rhs)");
        checkCuda(cudaMemset(dSolution, 0, static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMemset(bucketed solution)");

        std::vector<std::tuple<int, int, int>> segments;
        segments.reserve(static_cast<size_t>(matrix.n));

        auto bucketForRow = [&](int row) {
            const int rowNnz = matrix.rowPtr[static_cast<size_t>(row + 1)] - matrix.rowPtr[static_cast<size_t>(row)];
            if (rowNnz <= kThreadThreshold) {
                return 0;
            }
            if (rowNnz <= kWarpThreshold) {
                return 1;
            }
            return 2;
        };

        int segmentBegin = 0;
        int currentBucket = bucketForRow(0);
        for (int row = 1; row < matrix.n; ++row) {
            const int rowBucket = bucketForRow(row);
            if (rowBucket != currentBucket) {
                segments.emplace_back(currentBucket, segmentBegin, row);
                segmentBegin = row;
                currentBucket = rowBucket;
            }
        }
        segments.emplace_back(currentBucket, segmentBegin, matrix.n);

        const size_t sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);
        for (const auto& [bucket, rowBegin, rowEnd] : segments) {
            if (bucket == 0) {
                sptrsvThreadBucketedKernel<<<1, 1>>>(rowBegin, rowEnd, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            } else if (bucket == 1) {
                sptrsvWarpBucketedKernel<<<1, 32>>>(rowBegin, rowEnd, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            } else {
                sptrsvBlockBucketedKernel<<<1, kBlockSize, sharedBytes>>>(rowBegin, rowEnd, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            }
        }
        checkCuda(cudaGetLastError(), "sptrsvGpuRowBucketed dispatch");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(bucketed)");

        solution.resize(static_cast<size_t>(matrix.n));
        checkCuda(cudaMemcpy(solution.data(), dSolution, static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(bucketed solution)");
    } catch (...) {
        if (dSolution) cudaFree(dSolution);
        if (dRhs) cudaFree(dRhs);
        if (dValues) cudaFree(dValues);
        if (dColIdx) cudaFree(dColIdx);
        if (dRowPtr) cudaFree(dRowPtr);
        return false;
    }

    cudaFree(dSolution);
    cudaFree(dRhs);
    cudaFree(dValues);
    cudaFree(dColIdx);
    cudaFree(dRowPtr);
    return true;
}

bool sptrsvGpuLevelAware(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    constexpr int kThreadThreshold = 4;
    constexpr int kWarpThreshold = 32;
    constexpr int kThreadBlockSize = 128;
    constexpr int kWarpBlockSize = 128;
    constexpr int kBlockSize = 256;

    const std::vector<int> levels = computeRowLevels(matrix);
    int maxLevel = 0;
    for (int level : levels) {
        maxLevel = std::max(maxLevel, level);
    }

    std::vector<std::vector<int>> threadRows(static_cast<size_t>(maxLevel + 1));
    std::vector<std::vector<int>> warpRows(static_cast<size_t>(maxLevel + 1));
    std::vector<std::vector<int>> blockRows(static_cast<size_t>(maxLevel + 1));
    for (int row = 0; row < matrix.n; ++row) {
        const int rowNnz = matrix.rowPtr[static_cast<size_t>(row + 1)] - matrix.rowPtr[static_cast<size_t>(row)];
        const int level = levels[static_cast<size_t>(row)];
        if (rowNnz <= kThreadThreshold) {
            threadRows[static_cast<size_t>(level)].push_back(row);
        } else if (rowNnz <= kWarpThreshold) {
            warpRows[static_cast<size_t>(level)].push_back(row);
        } else {
            blockRows[static_cast<size_t>(level)].push_back(row);
        }
    }

    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;
    int* dThreadRows = nullptr;
    int* dWarpRows = nullptr;
    int* dBlockRows = nullptr;

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(level rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(level colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(level values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(level rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(level solution)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dThreadRows), static_cast<size_t>(matrix.n) * sizeof(int)), "cudaMalloc(level thread rows)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dWarpRows), static_cast<size_t>(matrix.n) * sizeof(int)), "cudaMalloc(level warp rows)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dBlockRows), static_cast<size_t>(matrix.n) * sizeof(int)), "cudaMalloc(level block rows)");

        checkCuda(cudaMemcpy(dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(level rowPtr)");
        checkCuda(cudaMemcpy(dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(level colIdx)");
        checkCuda(cudaMemcpy(dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(level values)");
        checkCuda(cudaMemcpy(dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(level rhs)");
        checkCuda(cudaMemset(dSolution, 0, static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMemset(level solution)");

        const size_t sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);
        for (int level = 0; level <= maxLevel; ++level) {
            const auto& threadList = threadRows[static_cast<size_t>(level)];
            if (!threadList.empty()) {
                const size_t bytes = threadList.size() * sizeof(int);
                checkCuda(cudaMemcpy(dThreadRows, threadList.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(level thread rows)");
                const int blocks = static_cast<int>((threadList.size() + kThreadBlockSize - 1) / kThreadBlockSize);
                sptrsvThreadRowListKernel<<<blocks, kThreadBlockSize>>>(
                    static_cast<int>(threadList.size()), dThreadRows, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            }

            const auto& warpList = warpRows[static_cast<size_t>(level)];
            if (!warpList.empty()) {
                const size_t bytes = warpList.size() * sizeof(int);
                checkCuda(cudaMemcpy(dWarpRows, warpList.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(level warp rows)");
                const int warpsPerBlock = kWarpBlockSize / 32;
                const int blocks = static_cast<int>((warpList.size() + warpsPerBlock - 1) / warpsPerBlock);
                sptrsvWarpRowListKernel<<<blocks, kWarpBlockSize>>>(
                    static_cast<int>(warpList.size()), dWarpRows, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            }

            const auto& blockList = blockRows[static_cast<size_t>(level)];
            if (!blockList.empty()) {
                const size_t bytes = blockList.size() * sizeof(int);
                checkCuda(cudaMemcpy(dBlockRows, blockList.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(level block rows)");
                sptrsvBlockRowListKernel<<<static_cast<int>(blockList.size()), kBlockSize, sharedBytes>>>(
                    static_cast<int>(blockList.size()), dBlockRows, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            }

            checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(level)");
        }

        checkCuda(cudaGetLastError(), "sptrsvGpuLevelAware dispatch");
        solution.resize(static_cast<size_t>(matrix.n));
        checkCuda(cudaMemcpy(solution.data(), dSolution, static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(level solution)");
    } catch (...) {
        if (dBlockRows) cudaFree(dBlockRows);
        if (dWarpRows) cudaFree(dWarpRows);
        if (dThreadRows) cudaFree(dThreadRows);
        if (dSolution) cudaFree(dSolution);
        if (dRhs) cudaFree(dRhs);
        if (dValues) cudaFree(dValues);
        if (dColIdx) cudaFree(dColIdx);
        if (dRowPtr) cudaFree(dRowPtr);
        return false;
    }

    cudaFree(dBlockRows);
    cudaFree(dWarpRows);
    cudaFree(dThreadRows);
    cudaFree(dSolution);
    cudaFree(dRhs);
    cudaFree(dValues);
    cudaFree(dColIdx);
    cudaFree(dRowPtr);
    return true;
}

bool sptrsvGpuRandomLevelAware(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    constexpr int kThreadThreshold = 2;
    constexpr int kWarpThreshold = 64;
    constexpr int kThreadBlockSize = 128;
    constexpr int kWarpBlockSize = 128;
    constexpr int kBlockSize = 256;

    const std::vector<int> levels = computeRowLevels(matrix);
    int maxLevel = 0;
    for (int level : levels) {
        maxLevel = std::max(maxLevel, level);
    }

    std::vector<std::vector<int>> threadRows(static_cast<size_t>(maxLevel + 1));
    std::vector<std::vector<int>> warpRows(static_cast<size_t>(maxLevel + 1));
    std::vector<std::vector<int>> blockRows(static_cast<size_t>(maxLevel + 1));
    for (int row = 0; row < matrix.n; ++row) {
        const int rowNnz = matrix.rowPtr[static_cast<size_t>(row + 1)] - matrix.rowPtr[static_cast<size_t>(row)];
        const int level = levels[static_cast<size_t>(row)];
        if (rowNnz <= kThreadThreshold) {
            auto& rows = threadRows[static_cast<size_t>(level)];
            rows.push_back(row);
        } else if (rowNnz <= kWarpThreshold) {
            auto& rows = warpRows[static_cast<size_t>(level)];
            rows.push_back(row);
        } else {
            auto& rows = blockRows[static_cast<size_t>(level)];
            rows.push_back(row);
        }
    }

    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;
    int* dThreadRows = nullptr;
    int* dWarpRows = nullptr;
    int* dBlockRows = nullptr;
    cudaStream_t threadStream = nullptr;
    cudaStream_t warpStream = nullptr;
    cudaStream_t blockStream = nullptr;
    cudaEvent_t threadDone = nullptr;
    cudaEvent_t warpDone = nullptr;
    cudaEvent_t blockDone = nullptr;

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(random level rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(random level colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(random level values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(random level rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(random level solution)");

        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dThreadRows), static_cast<size_t>(matrix.n) * sizeof(int)), "cudaMalloc(random thread rows)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dWarpRows), static_cast<size_t>(matrix.n) * sizeof(int)), "cudaMalloc(random warp rows)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dBlockRows), static_cast<size_t>(matrix.n) * sizeof(int)), "cudaMalloc(random block rows)");

        checkCuda(cudaStreamCreate(&threadStream), "cudaStreamCreate(random thread)");
        checkCuda(cudaStreamCreate(&warpStream), "cudaStreamCreate(random warp)");
        checkCuda(cudaStreamCreate(&blockStream), "cudaStreamCreate(random block)");
        checkCuda(cudaEventCreateWithFlags(&threadDone, cudaEventDisableTiming), "cudaEventCreate(random thread)");
        checkCuda(cudaEventCreateWithFlags(&warpDone, cudaEventDisableTiming), "cudaEventCreate(random warp)");
        checkCuda(cudaEventCreateWithFlags(&blockDone, cudaEventDisableTiming), "cudaEventCreate(random block)");

        checkCuda(cudaMemcpy(dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(random level rowPtr)");
        checkCuda(cudaMemcpy(dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(random level colIdx)");
        checkCuda(cudaMemcpy(dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(random level values)");
        checkCuda(cudaMemcpy(dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(random level rhs)");
        checkCuda(cudaMemset(dSolution, 0, static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMemset(random level solution)");

        const size_t sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);
        bool havePreviousLevelWork = false;
        bool prevThreadActive = false;
        bool prevWarpActive = false;
        bool prevBlockActive = false;
        for (int level = 0; level <= maxLevel; ++level) {
            const auto& threadList = threadRows[static_cast<size_t>(level)];
            const auto& warpList = warpRows[static_cast<size_t>(level)];
            const auto& blockList = blockRows[static_cast<size_t>(level)];

            const bool threadActive = !threadList.empty();
            const bool warpActive = !warpList.empty();
            const bool blockActive = !blockList.empty();

            if (havePreviousLevelWork) {
                if (threadActive) {
                    if (prevThreadActive) checkCuda(cudaStreamWaitEvent(threadStream, threadDone, 0), "cudaStreamWaitEvent(random thread/thread)");
                    if (prevWarpActive) checkCuda(cudaStreamWaitEvent(threadStream, warpDone, 0), "cudaStreamWaitEvent(random thread/warp)");
                    if (prevBlockActive) checkCuda(cudaStreamWaitEvent(threadStream, blockDone, 0), "cudaStreamWaitEvent(random thread/block)");
                }
                if (warpActive) {
                    if (prevThreadActive) checkCuda(cudaStreamWaitEvent(warpStream, threadDone, 0), "cudaStreamWaitEvent(random warp/thread)");
                    if (prevWarpActive) checkCuda(cudaStreamWaitEvent(warpStream, warpDone, 0), "cudaStreamWaitEvent(random warp/warp)");
                    if (prevBlockActive) checkCuda(cudaStreamWaitEvent(warpStream, blockDone, 0), "cudaStreamWaitEvent(random warp/block)");
                }
                if (blockActive) {
                    if (prevThreadActive) checkCuda(cudaStreamWaitEvent(blockStream, threadDone, 0), "cudaStreamWaitEvent(random block/thread)");
                    if (prevWarpActive) checkCuda(cudaStreamWaitEvent(blockStream, warpDone, 0), "cudaStreamWaitEvent(random block/warp)");
                    if (prevBlockActive) checkCuda(cudaStreamWaitEvent(blockStream, blockDone, 0), "cudaStreamWaitEvent(random block/block)");
                }
            }

            if (threadActive) {
                const size_t bytes = threadList.size() * sizeof(int);
                checkCuda(cudaMemcpyAsync(dThreadRows, threadList.data(), bytes, cudaMemcpyHostToDevice, threadStream), "cudaMemcpyAsync(random thread rows)");
                const int blocks = static_cast<int>((threadList.size() + kThreadBlockSize - 1) / kThreadBlockSize);
                sptrsvThreadRowListKernel<<<blocks, kThreadBlockSize, 0, threadStream>>>(
                    static_cast<int>(threadList.size()), dThreadRows, dRowPtr, dColIdx, dValues, dRhs, dSolution);
                checkCuda(cudaEventRecord(threadDone, threadStream), "cudaEventRecord(random thread)");
            }

            if (warpActive) {
                const size_t bytes = warpList.size() * sizeof(int);
                checkCuda(cudaMemcpyAsync(dWarpRows, warpList.data(), bytes, cudaMemcpyHostToDevice, warpStream), "cudaMemcpyAsync(random warp rows)");
                const int warpsPerBlock = kWarpBlockSize / 32;
                const int blocks = static_cast<int>((warpList.size() + warpsPerBlock - 1) / warpsPerBlock);
                sptrsvWarpRowListKernel<<<blocks, kWarpBlockSize, 0, warpStream>>>(
                    static_cast<int>(warpList.size()), dWarpRows, dRowPtr, dColIdx, dValues, dRhs, dSolution);
                checkCuda(cudaEventRecord(warpDone, warpStream), "cudaEventRecord(random warp)");
            }

            if (blockActive) {
                const size_t bytes = blockList.size() * sizeof(int);
                checkCuda(cudaMemcpyAsync(dBlockRows, blockList.data(), bytes, cudaMemcpyHostToDevice, blockStream), "cudaMemcpyAsync(random block rows)");
                sptrsvBlockRowListKernel<<<static_cast<int>(blockList.size()), kBlockSize, sharedBytes, blockStream>>>(
                    static_cast<int>(blockList.size()), dBlockRows, dRowPtr, dColIdx, dValues, dRhs, dSolution);
                checkCuda(cudaEventRecord(blockDone, blockStream), "cudaEventRecord(random block)");
            }

            havePreviousLevelWork = threadActive || warpActive || blockActive;
            prevThreadActive = threadActive;
            prevWarpActive = warpActive;
            prevBlockActive = blockActive;
        }

        if (prevThreadActive) checkCuda(cudaStreamSynchronize(threadStream), "cudaStreamSynchronize(random thread)");
        if (prevWarpActive) checkCuda(cudaStreamSynchronize(warpStream), "cudaStreamSynchronize(random warp)");
        if (prevBlockActive) checkCuda(cudaStreamSynchronize(blockStream), "cudaStreamSynchronize(random block)");

        checkCuda(cudaGetLastError(), "sptrsvGpuRandomLevelAware dispatch");
        solution.resize(static_cast<size_t>(matrix.n));
        checkCuda(cudaMemcpy(solution.data(), dSolution, static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(random level solution)");
    } catch (const std::exception& ex) {
        std::cerr << "sptrsvGpuRandomLevelAware failed: " << ex.what() << "\n";
        if (threadDone) cudaEventDestroy(threadDone);
        if (warpDone) cudaEventDestroy(warpDone);
        if (blockDone) cudaEventDestroy(blockDone);
        if (threadStream) cudaStreamDestroy(threadStream);
        if (warpStream) cudaStreamDestroy(warpStream);
        if (blockStream) cudaStreamDestroy(blockStream);
        if (dBlockRows) cudaFree(dBlockRows);
        if (dWarpRows) cudaFree(dWarpRows);
        if (dThreadRows) cudaFree(dThreadRows);
        if (dSolution) cudaFree(dSolution);
        if (dRhs) cudaFree(dRhs);
        if (dValues) cudaFree(dValues);
        if (dColIdx) cudaFree(dColIdx);
        if (dRowPtr) cudaFree(dRowPtr);
        return false;
    }

    cudaEventDestroy(threadDone);
    cudaEventDestroy(warpDone);
    cudaEventDestroy(blockDone);
    cudaStreamDestroy(threadStream);
    cudaStreamDestroy(warpStream);
    cudaStreamDestroy(blockStream);
    cudaFree(dBlockRows);
    cudaFree(dWarpRows);
    cudaFree(dThreadRows);
    cudaFree(dSolution);
    cudaFree(dRhs);
    cudaFree(dValues);
    cudaFree(dColIdx);
    cudaFree(dRowPtr);
    return true;
}

bool sptrsvGpuPowerLawAware(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    constexpr int kWarpThreshold = 24;
    constexpr int kWarpBlockSize = 128;
    constexpr int kBlockSize = 256;

    const std::vector<int> levels = computeRowLevels(matrix);
    int maxLevel = 0;
    for (int level : levels) {
        maxLevel = std::max(maxLevel, level);
    }

    std::vector<std::vector<int>> warpRows(static_cast<size_t>(maxLevel + 1));
    std::vector<std::vector<int>> blockRows(static_cast<size_t>(maxLevel + 1));

    for (int row = 0; row < matrix.n; ++row) {
        const int rowNnz = matrix.rowPtr[static_cast<size_t>(row + 1)] - matrix.rowPtr[static_cast<size_t>(row)];
        const int level = levels[static_cast<size_t>(row)];
        if (rowNnz <= kWarpThreshold) {
            warpRows[static_cast<size_t>(level)].push_back(row);
        } else {
            blockRows[static_cast<size_t>(level)].push_back(row);
        }
    }

    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;
    int* dWarpRows = nullptr;
    int* dBlockRows = nullptr;
    cudaStream_t workStream = nullptr;

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(power rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(power colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(power values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(power rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(power solution)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dWarpRows), static_cast<size_t>(matrix.n) * sizeof(int)), "cudaMalloc(power warp rows)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dBlockRows), static_cast<size_t>(matrix.n) * sizeof(int)), "cudaMalloc(power block rows)");

        checkCuda(cudaStreamCreate(&workStream), "cudaStreamCreate(power work)");

        checkCuda(cudaMemcpy(dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(power rowPtr)");
        checkCuda(cudaMemcpy(dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(power colIdx)");
        checkCuda(cudaMemcpy(dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(power values)");
        checkCuda(cudaMemcpy(dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(power rhs)");
        checkCuda(cudaMemset(dSolution, 0, static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMemset(power solution)");

        const size_t sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);
        for (int level = 0; level <= maxLevel; ++level) {
            const auto& warpList = warpRows[static_cast<size_t>(level)];
            const auto& blockList = blockRows[static_cast<size_t>(level)];

            const bool warpActive = !warpList.empty();
            const bool blockActive = !blockList.empty();

            if (warpActive) {
                const size_t bytes = warpList.size() * sizeof(int);
                checkCuda(cudaMemcpyAsync(dWarpRows, warpList.data(), bytes, cudaMemcpyHostToDevice, workStream), "cudaMemcpyAsync(power warp rows)");
                const int warpsPerBlock = kWarpBlockSize / 32;
                const int blocks = static_cast<int>((warpList.size() + warpsPerBlock - 1) / warpsPerBlock);
                sptrsvWarpRowListKernel<<<blocks, kWarpBlockSize, 0, workStream>>>(
                    static_cast<int>(warpList.size()), dWarpRows, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            }

            if (blockActive) {
                const size_t bytes = blockList.size() * sizeof(int);
                checkCuda(cudaMemcpyAsync(dBlockRows, blockList.data(), bytes, cudaMemcpyHostToDevice, workStream), "cudaMemcpyAsync(power block rows)");
                sptrsvBlockRowListKernel<<<static_cast<int>(blockList.size()), kBlockSize, sharedBytes, workStream>>>(
                    static_cast<int>(blockList.size()), dBlockRows, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            }

            if (warpActive || blockActive) {
                checkCuda(cudaStreamSynchronize(workStream), "cudaStreamSynchronize(power work)");
            }
        }

        checkCuda(cudaGetLastError(), "sptrsvGpuPowerLawAware dispatch");
        solution.resize(static_cast<size_t>(matrix.n));
        checkCuda(cudaMemcpy(solution.data(), dSolution, static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(power solution)");
    } catch (const std::exception& ex) {
        std::cerr << "sptrsvGpuPowerLawAware failed: " << ex.what() << "\n";
        if (workStream) cudaStreamDestroy(workStream);
        if (dBlockRows) cudaFree(dBlockRows);
        if (dWarpRows) cudaFree(dWarpRows);
        if (dSolution) cudaFree(dSolution);
        if (dRhs) cudaFree(dRhs);
        if (dValues) cudaFree(dValues);
        if (dColIdx) cudaFree(dColIdx);
        if (dRowPtr) cudaFree(dRowPtr);
        return false;
    }

    cudaStreamDestroy(workStream);
    cudaFree(dBlockRows);
    cudaFree(dWarpRows);
    cudaFree(dSolution);
    cudaFree(dRhs);
    cudaFree(dValues);
    cudaFree(dColIdx);
    cudaFree(dRowPtr);
    return true;
}

bool sptrsvGpuBlockAware(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    const BlockAnalysis analysis = chooseBlockSize(matrix);
    const int estimatedBlock = analysis.blockSize;
    const int kThreadThreshold = std::max(2, estimatedBlock / 2);
    const int kWarpThreshold = std::max(16, estimatedBlock * estimatedBlock);
    constexpr int kBlockSize = 256;

    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(blockaware rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(blockaware colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(blockaware values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(blockaware rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(blockaware solution)");

        checkCuda(cudaMemcpy(dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(blockaware rowPtr)");
        checkCuda(cudaMemcpy(dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(blockaware colIdx)");
        checkCuda(cudaMemcpy(dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(blockaware values)");
        checkCuda(cudaMemcpy(dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(blockaware rhs)");
        checkCuda(cudaMemset(dSolution, 0, static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMemset(blockaware solution)");

        std::vector<std::tuple<int, int, int>> segments;
        segments.reserve(static_cast<size_t>(matrix.n));

        auto bucketForRow = [&](int row) {
            const int rowNnz = matrix.rowPtr[static_cast<size_t>(row + 1)] - matrix.rowPtr[static_cast<size_t>(row)];
            if (rowNnz <= kThreadThreshold) {
                return 0;
            }
            if (rowNnz <= kWarpThreshold) {
                return 1;
            }
            return 2;
        };

        int segmentBegin = 0;
        int currentBucket = bucketForRow(0);
        for (int row = 1; row < matrix.n; ++row) {
            const int rowBucket = bucketForRow(row);
            if (rowBucket != currentBucket) {
                segments.emplace_back(currentBucket, segmentBegin, row);
                segmentBegin = row;
                currentBucket = rowBucket;
            }
        }
        segments.emplace_back(currentBucket, segmentBegin, matrix.n);

        std::vector<std::tuple<int, int, int>> merged;
        merged.reserve(segments.size());
        for (const auto& seg : segments) {
            if (!merged.empty()) {
                auto& prev = merged.back();
                const int prevBucket = std::get<0>(prev);
                const int prevBegin = std::get<1>(prev);
                const int prevEnd = std::get<2>(prev);
                const int curBucket = std::get<0>(seg);
                const int curBegin = std::get<1>(seg);
                const int curEnd = std::get<2>(seg);
                const int prevLen = prevEnd - prevBegin;
                const int curLen = curEnd - curBegin;
                if ((prevBucket == 1 || prevBucket == 2) &&
                    (curBucket == 1 || curBucket == 2) &&
                    prevEnd == curBegin &&
                    prevLen + curLen <= 64) {
                    std::get<0>(prev) = 2;
                    std::get<2>(prev) = curEnd;
                    continue;
                }
            }
            merged.push_back(seg);
        }

        const size_t sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);
        for (const auto& [bucket, rowBegin, rowEnd] : merged) {
            if (bucket == 0) {
                sptrsvThreadBucketedKernel<<<1, 1>>>(rowBegin, rowEnd, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            } else if (bucket == 1) {
                sptrsvWarpBucketedKernel<<<1, 32>>>(rowBegin, rowEnd, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            } else {
                sptrsvBlockBucketedKernel<<<1, kBlockSize, sharedBytes>>>(rowBegin, rowEnd, dRowPtr, dColIdx, dValues, dRhs, dSolution);
            }
        }
        checkCuda(cudaGetLastError(), "sptrsvGpuBlockAware dispatch");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(blockaware)");

        solution.resize(static_cast<size_t>(matrix.n));
        checkCuda(cudaMemcpy(solution.data(), dSolution, static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(blockaware solution)");
    } catch (...) {
        if (dSolution) cudaFree(dSolution);
        if (dRhs) cudaFree(dRhs);
        if (dValues) cudaFree(dValues);
        if (dColIdx) cudaFree(dColIdx);
        if (dRowPtr) cudaFree(dRowPtr);
        return false;
    }

    cudaFree(dSolution);
    cudaFree(dRhs);
    cudaFree(dValues);
    cudaFree(dColIdx);
    cudaFree(dRowPtr);
    return true;
}

bool sptrsvGpuBandedAware(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    if (useCoalescedForBandedAware(matrix)) {
        return sptrsvGpuCoalesced(matrix, rhs, solution);
    }
    return sptrsvGpuRowBucketed(matrix, rhs, solution);
}

bool sptrsvGpuRegularAware(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    constexpr int kBlockSize = 256;

    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;

    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(regular rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(regular colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(regular values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(regular rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(regular solution)");

        checkCuda(cudaMemcpy(dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(regular rowPtr)");
        checkCuda(cudaMemcpy(dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(regular colIdx)");
        checkCuda(cudaMemcpy(dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(regular values)");
        checkCuda(cudaMemcpy(dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(regular rhs)");
        checkCuda(cudaMemset(dSolution, 0, static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMemset(regular solution)");

        const MatrixFeatures f = analyzeMatrixFeatures(matrix);
        if (f.maxRowNnz <= 32) {
            sptrsvWarpBucketedKernel<<<1, 32>>>(0, matrix.n, dRowPtr, dColIdx, dValues, dRhs, dSolution);
        } else {
            const size_t sharedBytes = static_cast<size_t>(2 * kBlockSize) * sizeof(double);
            sptrsvBlockBucketedKernel<<<1, kBlockSize, sharedBytes>>>(0, matrix.n, dRowPtr, dColIdx, dValues, dRhs, dSolution);
        }
        checkCuda(cudaGetLastError(), "sptrsvGpuRegularAware dispatch");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(regular)");

        solution.resize(static_cast<size_t>(matrix.n));
        checkCuda(cudaMemcpy(solution.data(), dSolution, static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(regular solution)");
    } catch (...) {
        if (dSolution) cudaFree(dSolution);
        if (dRhs) cudaFree(dRhs);
        if (dValues) cudaFree(dValues);
        if (dColIdx) cudaFree(dColIdx);
        if (dRowPtr) cudaFree(dRowPtr);
        return false;
    }

    cudaFree(dSolution);
    cudaFree(dRhs);
    cudaFree(dValues);
    cudaFree(dColIdx);
    cudaFree(dRowPtr);
    return true;
}

bool sptrsvGpuFamilyAware(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    const FamilyChoice choice = chooseFamilyKernel(matrix);
    if (choice.kernel == "LevelAware") {
        return sptrsvGpuLevelAware(matrix, rhs, solution);
    }
    if (choice.kernel == "RandomAware") {
        return sptrsvGpuRandomLevelAware(matrix, rhs, solution);
    }
    if (choice.kernel == "PowerLawAware") {
        return sptrsvGpuPowerLawAware(matrix, rhs, solution);
    }
    if (choice.kernel == "BandedAware") {
        return sptrsvGpuBandedAware(matrix, rhs, solution);
    }
    if (choice.kernel == "RegularAware") {
        return sptrsvGpuRegularAware(matrix, rhs, solution);
    }
    return sptrsvGpuRowBucketed(matrix, rhs, solution);
}

bool sptrsvGpuLearnedAware(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    const std::string kernel = chooseLearnedKernel(matrix);
    if (kernel == "LevelAware") {
        return sptrsvGpuLevelAware(matrix, rhs, solution);
    }
    if (kernel == "RandomAware") {
        return sptrsvGpuRandomLevelAware(matrix, rhs, solution);
    }
    if (kernel == "PowerLawAware") {
        return sptrsvGpuPowerLawAware(matrix, rhs, solution);
    }
    if (kernel == "BandedAware") {
        return sptrsvGpuBandedAware(matrix, rhs, solution);
    }
    if (kernel == "RegularAware") {
        return sptrsvGpuRegularAware(matrix, rhs, solution);
    }
    if (kernel == "Coalesced") {
        return sptrsvGpuCoalesced(matrix, rhs, solution);
    }
    return sptrsvGpuRowBucketed(matrix, rhs, solution);
}

bool sptrsvCuSparse(const CsrMatrix& matrix, const std::vector<double>& rhs, std::vector<double>& solution)
{
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr;
    cusparseDnVecDescr_t vecY = nullptr;
    cusparseSpSVDescr_t spsvDescr = nullptr;
    int* dRowPtr = nullptr;
    int* dColIdx = nullptr;
    double* dValues = nullptr;
    double* dRhs = nullptr;
    double* dSolution = nullptr;
    void* dBuffer = nullptr;
    size_t bufferSize = 0;

    try {
        checkCusparse(cusparseCreate(&handle), "cusparseCreate");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRowPtr), matrix.rowPtr.size() * sizeof(int)), "cudaMalloc(cusparse rowPtr)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dColIdx), matrix.colIdx.size() * sizeof(int)), "cudaMalloc(cusparse colIdx)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dValues), matrix.values.size() * sizeof(double)), "cudaMalloc(cusparse values)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dRhs), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(cusparse rhs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dSolution), static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMalloc(cusparse solution)");

        checkCuda(cudaMemcpy(dRowPtr, matrix.rowPtr.data(), matrix.rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(cusparse rowPtr)");
        checkCuda(cudaMemcpy(dColIdx, matrix.colIdx.data(), matrix.colIdx.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(cusparse colIdx)");
        checkCuda(cudaMemcpy(dValues, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(cusparse values)");
        checkCuda(cudaMemcpy(dRhs, rhs.data(), static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(cusparse rhs)");
        checkCuda(cudaMemset(dSolution, 0, static_cast<size_t>(matrix.n) * sizeof(double)), "cudaMemset(cusparse solution)");

        checkCusparse(
            cusparseCreateCsr(
                &matA,
                matrix.n,
                matrix.n,
                matrix.nnz,
                dRowPtr,
                dColIdx,
                dValues,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_64F),
            "cusparseCreateCsr");
        setMatrixAttributes(matA);

        checkCusparse(cusparseCreateDnVec(&vecX, matrix.n, dRhs, CUDA_R_64F), "cusparseCreateDnVec(x)");
        checkCusparse(cusparseCreateDnVec(&vecY, matrix.n, dSolution, CUDA_R_64F), "cusparseCreateDnVec(y)");
        checkCusparse(cusparseSpSV_createDescr(&spsvDescr), "cusparseSpSV_createDescr");

        const double alpha = 1.0;
        checkCusparse(
            cusparseSpSV_bufferSize(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                matA,
                vecX,
                vecY,
                CUDA_R_64F,
                CUSPARSE_SPSV_ALG_DEFAULT,
                spsvDescr,
                &bufferSize),
            "cusparseSpSV_bufferSize");
        checkCuda(cudaMalloc(&dBuffer, bufferSize), "cudaMalloc(cusparse buffer)");

        checkCusparse(
            cusparseSpSV_analysis(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                matA,
                vecX,
                vecY,
                CUDA_R_64F,
                CUSPARSE_SPSV_ALG_DEFAULT,
                spsvDescr,
                dBuffer),
            "cusparseSpSV_analysis");
        checkCusparse(
            cusparseSpSV_solve(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                matA,
                vecX,
                vecY,
                CUDA_R_64F,
                CUSPARSE_SPSV_ALG_DEFAULT,
                spsvDescr),
            "cusparseSpSV_solve");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(cusparse)");

        solution.resize(static_cast<size_t>(matrix.n));
        checkCuda(cudaMemcpy(solution.data(), dSolution, static_cast<size_t>(matrix.n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(cusparse solution)");
    } catch (...) {
        if (dBuffer) cudaFree(dBuffer);
        if (dSolution) cudaFree(dSolution);
        if (dRhs) cudaFree(dRhs);
        if (dValues) cudaFree(dValues);
        if (dColIdx) cudaFree(dColIdx);
        if (dRowPtr) cudaFree(dRowPtr);
        if (spsvDescr) cusparseSpSV_destroyDescr(spsvDescr);
        if (vecY) cusparseDestroyDnVec(vecY);
        if (vecX) cusparseDestroyDnVec(vecX);
        if (matA) cusparseDestroySpMat(matA);
        if (handle) cusparseDestroy(handle);
        return false;
    }

    cudaFree(dBuffer);
    cudaFree(dSolution);
    cudaFree(dRhs);
    cudaFree(dValues);
    cudaFree(dColIdx);
    cudaFree(dRowPtr);
    cusparseSpSV_destroyDescr(spsvDescr);
    cusparseDestroyDnVec(vecY);
    cusparseDestroyDnVec(vecX);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);
    return true;
}

void printDeviceInfo()
{
    int device = 0;
    cudaDeviceProp prop{};
    checkCuda(cudaGetDevice(&device), "cudaGetDevice");
    checkCuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    std::cout << "GPU device : " << prop.name
              << " (compute capability " << prop.major << "." << prop.minor << ")\n";
}

template <typename Func>
double timeMs(Func&& func)
{
    const auto start = std::chrono::steady_clock::now();
    func();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void runFamilySweep(const std::string& familyFilter = "", int onlyN = 0)
{
    const std::vector<int> sizes = {1024, 4096, 16384, 32768};
    const std::vector<BenchmarkCase> cases = makeBenchmarkCases();

    std::cout << "GPU New Family Sweep\n";
    printDeviceInfo();
    std::cout << "\n";

    std::cout << std::left
              << std::setw(14) << "Family"
              << std::setw(10) << "N"
              << std::setw(12) << "NNZ"
              << std::setw(14) << "AutoFam"
              << std::setw(14) << "AutoKern"
              << std::setw(16) << "Coalesced"
              << std::setw(16) << "RowBucketed"
              << std::setw(16) << "LevelAware"
              << std::setw(16) << "RandomAware"
              << std::setw(16) << "PowerLawAware"
              << std::setw(16) << "BlockAware"
              << std::setw(16) << "BandedAware"
              << std::setw(16) << "FamilyAware"
              << std::setw(16) << "LearnedAware"
              << std::setw(16) << "cuSPARSE"
              << "\n";
    std::cout << std::string(224, '-') << "\n";

    for (const auto& benchmarkCase : cases) {
        if (!familyFilter.empty() && benchmarkCase.name != familyFilter) {
            continue;
        }
        for (int n : sizes) {
            if (onlyN > 0 && n != onlyN) {
                continue;
            }
            const CsrMatrix matrix = benchmarkCase.factory(n);
            const std::vector<double> rhs = generateRandomRhs(n, 99);

            std::vector<double> coalescedSolution;
            std::vector<double> rowBucketedSolution;
            std::vector<double> levelAwareSolution;
            std::vector<double> randomAwareSolution;
            std::vector<double> powerLawAwareSolution;
            std::vector<double> blockAwareSolution;
            std::vector<double> bandedAwareSolution;
            std::vector<double> familyAwareSolution;
            std::vector<double> learnedAwareSolution;
            std::vector<double> cusparseSolution;

            const FamilyChoice familyChoice = chooseFamilyKernel(matrix);
            bool coalescedOk = false;
            bool rowBucketedOk = false;
            bool levelAwareOk = false;
            bool randomAwareOk = false;
            bool powerLawAwareOk = false;
            bool blockAwareOk = false;
            bool bandedAwareOk = false;
            bool familyAwareOk = false;
            bool learnedAwareOk = false;
            bool cusparseOk = false;
            double coalescedMs = timeMs([&]() { coalescedOk = sptrsvGpuCoalesced(matrix, rhs, coalescedSolution); });
            double rowBucketedMs = timeMs([&]() { rowBucketedOk = sptrsvGpuRowBucketed(matrix, rhs, rowBucketedSolution); });
            double levelAwareMs = timeMs([&]() { levelAwareOk = sptrsvGpuLevelAware(matrix, rhs, levelAwareSolution); });
            double randomAwareMs = timeMs([&]() { randomAwareOk = sptrsvGpuRandomLevelAware(matrix, rhs, randomAwareSolution); });
            double powerLawAwareMs = timeMs([&]() { powerLawAwareOk = sptrsvGpuPowerLawAware(matrix, rhs, powerLawAwareSolution); });
            double blockAwareMs = timeMs([&]() { blockAwareOk = sptrsvGpuBlockAware(matrix, rhs, blockAwareSolution); });
            double bandedAwareMs = timeMs([&]() { bandedAwareOk = sptrsvGpuBandedAware(matrix, rhs, bandedAwareSolution); });
            double familyAwareMs = timeMs([&]() { familyAwareOk = sptrsvGpuFamilyAware(matrix, rhs, familyAwareSolution); });
            double learnedAwareMs = timeMs([&]() { learnedAwareOk = sptrsvGpuLearnedAware(matrix, rhs, learnedAwareSolution); });
            double cusparseMs = timeMs([&]() { cusparseOk = sptrsvCuSparse(matrix, rhs, cusparseSolution); });

            std::cout << std::setw(14) << benchmarkCase.name
                      << std::setw(10) << n
                      << std::setw(12) << matrix.nnz
                      << std::setw(14) << familyChoice.family
                      << std::setw(14) << familyChoice.kernel;
            if (coalescedOk) {
                std::cout << std::setw(16) << std::fixed << std::setprecision(4) << coalescedMs;
            } else {
                std::cout << std::setw(16) << "failed";
            }
            if (rowBucketedOk) {
                std::cout << std::setw(16) << std::fixed << std::setprecision(4) << rowBucketedMs;
            } else {
                std::cout << std::setw(16) << "failed";
            }
            if (levelAwareOk) {
                std::cout << std::setw(16) << std::fixed << std::setprecision(4) << levelAwareMs;
            } else {
                std::cout << std::setw(16) << "failed";
            }
            if (randomAwareOk) {
                std::cout << std::setw(16) << std::fixed << std::setprecision(4) << randomAwareMs;
            } else {
                std::cout << std::setw(16) << "failed";
            }
            if (powerLawAwareOk) {
                std::cout << std::setw(16) << std::fixed << std::setprecision(4) << powerLawAwareMs;
            } else {
                std::cout << std::setw(16) << "failed";
            }
            if (blockAwareOk) {
                std::cout << std::setw(16) << std::fixed << std::setprecision(4) << blockAwareMs;
            } else {
                std::cout << std::setw(16) << "failed";
            }
            if (bandedAwareOk) {
                std::cout << std::setw(16) << std::fixed << std::setprecision(4) << bandedAwareMs;
            } else {
                std::cout << std::setw(16) << "failed";
            }
            if (familyAwareOk) {
                std::cout << std::setw(16) << std::fixed << std::setprecision(4) << familyAwareMs;
            } else {
                std::cout << std::setw(16) << "failed";
            }
            if (learnedAwareOk) {
                std::cout << std::setw(16) << std::fixed << std::setprecision(4) << learnedAwareMs;
            } else {
                std::cout << std::setw(16) << "failed";
            }
            if (cusparseOk) {
                std::cout << std::setw(16) << std::fixed << std::setprecision(4) << cusparseMs;
            } else {
                std::cout << std::setw(16) << "failed";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "real") {
        runRealMatrixSweep();
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "real-one") {
        const std::string matrixName = (argc > 2) ? argv[2] : "ASIC_100k";
        const std::string mode = (argc > 3) ? argv[3] : "learnedaware";

        const std::vector<RealMatrixCase> cases = makeRealMatrixCases();
        auto it = std::find_if(
            cases.begin(),
            cases.end(),
            [&](const RealMatrixCase& c) { return c.name == matrixName; });
        if (it == cases.end()) {
            std::cerr << "Unknown real matrix: " << matrixName << "\n";
            return 1;
        }

        const CsrMatrix matrix = loadMatrixMarketAsLowerTriangular(it->path);
        const std::vector<double> rhs = generateRandomRhs(matrix.n, 99);
        std::vector<double> solution;
        bool ok = false;
        double ms = 0.0;

        if (mode == "features") {
            const FamilyChoice choice = chooseFamilyKernel(matrix);
            std::cout << "Real matrix features\n";
            std::cout << "Family      : " << it->family << "\n";
            std::cout << "Matrix      : " << it->name << "\n";
            std::cout << "N           : " << matrix.n << "\n";
            std::cout << "NNZ         : " << matrix.nnz << "\n";
            printMatrixFeatures(matrix);
            std::cout << "AutoFam     : " << choice.family << "\n";
            std::cout << "AutoKern    : " << choice.kernel << "\n";
            std::cout << "LearnedKern : " << chooseLearnedKernel(matrix) << "\n";
            return 0;
        }

        if (mode == "coalesced") {
            ms = timeMs([&]() { ok = sptrsvGpuCoalesced(matrix, rhs, solution); });
        } else if (mode == "rowbucketed") {
            ms = timeMs([&]() { ok = sptrsvGpuRowBucketed(matrix, rhs, solution); });
        } else if (mode == "levelaware") {
            ms = timeMs([&]() { ok = sptrsvGpuLevelAware(matrix, rhs, solution); });
        } else if (mode == "randomaware") {
            ms = timeMs([&]() { ok = sptrsvGpuRandomLevelAware(matrix, rhs, solution); });
        } else if (mode == "powerlawaware") {
            ms = timeMs([&]() { ok = sptrsvGpuPowerLawAware(matrix, rhs, solution); });
        } else if (mode == "blockaware") {
            ms = timeMs([&]() { ok = sptrsvGpuBlockAware(matrix, rhs, solution); });
        } else if (mode == "bandedaware") {
            ms = timeMs([&]() { ok = sptrsvGpuBandedAware(matrix, rhs, solution); });
        } else if (mode == "regularaware") {
            ms = timeMs([&]() { ok = sptrsvGpuRegularAware(matrix, rhs, solution); });
        } else if (mode == "familyaware") {
            ms = timeMs([&]() { ok = sptrsvGpuFamilyAware(matrix, rhs, solution); });
        } else if (mode == "learnedaware") {
            ms = timeMs([&]() { ok = sptrsvGpuLearnedAware(matrix, rhs, solution); });
        } else if (mode == "cusparse") {
            ms = timeMs([&]() { ok = sptrsvCuSparse(matrix, rhs, solution); });
        } else {
            std::cerr << "Unknown mode: " << mode << "\n";
            return 1;
        }

        std::cout << "Real matrix one-shot\n";
        std::cout << "Family      : " << it->family << "\n";
        std::cout << "Matrix      : " << it->name << "\n";
        std::cout << "Mode        : " << mode << "\n";
        std::cout << "N           : " << matrix.n << "\n";
        std::cout << "NNZ         : " << matrix.nnz << "\n";
        printDeviceInfo();
        if (!ok) {
            std::cout << "Status      : failed\n";
            return 1;
        }
        const double residual = computeResidual(matrix, solution, rhs);
        std::cout << "Time (ms)   : " << std::fixed << std::setprecision(4) << ms << "\n";
        std::cout << "Residual    : " << std::scientific << residual << "\n";
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "families") {
        const std::string familyFilter = (argc > 2) ? argv[2] : "";
        const int onlyN = (argc > 3) ? std::atoi(argv[3]) : 0;
        runFamilySweep(familyFilter, onlyN);
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "family-one") {
        const std::string family = (argc > 2) ? argv[2] : "random";
        const int n = (argc > 3) ? std::atoi(argv[3]) : 4096;
        const std::string mode = (argc > 4) ? argv[4] : "all";

        const std::vector<BenchmarkCase> cases = makeBenchmarkCases();
        auto it = std::find_if(
            cases.begin(),
            cases.end(),
            [&](const BenchmarkCase& c) { return c.name == family; });
        if (it == cases.end()) {
            std::cerr << "Unknown family: " << family << "\n";
            return 1;
        }

        const CsrMatrix matrix = it->factory(n);
        const std::vector<double> rhs = generateRandomRhs(n, 99);
        std::vector<double> solution;
        bool ok = false;
        double ms = 0.0;
        double residual = 0.0;

        if (mode == "features") {
            const FamilyChoice choice = chooseFamilyKernel(matrix);
            std::cout << "Family features\n";
            std::cout << "Family      : " << family << "\n";
            std::cout << "N           : " << n << "\n";
            std::cout << "NNZ         : " << matrix.nnz << "\n";
            printMatrixFeatures(matrix);
            std::cout << "AutoFam     : " << choice.family << "\n";
            std::cout << "AutoKern    : " << choice.kernel << "\n";
            std::cout << "LearnedKern : " << chooseLearnedKernel(matrix) << "\n";
            return 0;
        }

        if (mode == "coalesced") {
            ms = timeMs([&]() { ok = sptrsvGpuCoalesced(matrix, rhs, solution); });
        } else if (mode == "rowbucketed") {
            ms = timeMs([&]() { ok = sptrsvGpuRowBucketed(matrix, rhs, solution); });
        } else if (mode == "levelaware") {
            ms = timeMs([&]() { ok = sptrsvGpuLevelAware(matrix, rhs, solution); });
        } else if (mode == "randomaware") {
            ms = timeMs([&]() { ok = sptrsvGpuRandomLevelAware(matrix, rhs, solution); });
        } else if (mode == "powerlawaware") {
            ms = timeMs([&]() { ok = sptrsvGpuPowerLawAware(matrix, rhs, solution); });
        } else if (mode == "blockaware") {
            ms = timeMs([&]() { ok = sptrsvGpuBlockAware(matrix, rhs, solution); });
        } else if (mode == "bandedaware") {
            ms = timeMs([&]() { ok = sptrsvGpuBandedAware(matrix, rhs, solution); });
        } else if (mode == "regularaware") {
            ms = timeMs([&]() { ok = sptrsvGpuRegularAware(matrix, rhs, solution); });
        } else if (mode == "familyaware") {
            ms = timeMs([&]() { ok = sptrsvGpuFamilyAware(matrix, rhs, solution); });
        } else if (mode == "learnedaware") {
            ms = timeMs([&]() { ok = sptrsvGpuLearnedAware(matrix, rhs, solution); });
        } else if (mode == "cusparse") {
            ms = timeMs([&]() { ok = sptrsvCuSparse(matrix, rhs, solution); });
        } else {
            std::cerr << "Unknown mode: " << mode << "\n";
            return 1;
        }

        std::cout << "Family one-shot\n";
        std::cout << "Family      : " << family << "\n";
        std::cout << "Mode        : " << mode << "\n";
        std::cout << "N           : " << n << "\n";
        std::cout << "NNZ         : " << matrix.nnz << "\n";
        printDeviceInfo();
        if (!ok) {
            std::cout << "Status      : failed\n";
            return 1;
        }
        residual = computeResidual(matrix, solution, rhs);
        std::cout << "Time (ms)   : " << std::fixed << std::setprecision(4) << ms << "\n";
        std::cout << "Residual    : " << std::scientific << residual << "\n";
        return 0;
    }

    const int n = (argc > 1) ? std::atoi(argv[1]) : 8192;
    const double density = (argc > 2) ? std::atof(argv[2]) : 0.01;
    const std::string mode = (argc > 3) ? argv[3] : "all";

    const bool runNaive = (mode == "all" || mode == "naive");
    const bool runCoalesced = (mode == "all" || mode == "coalesced");
    const bool runThreadRow = (mode == "all" || mode == "thread");
    const bool runWarpRow = (mode == "all" || mode == "warp");
    const bool runAdaptiveRow = (mode == "all" || mode == "adaptive");
    const bool runBlockRow = (mode == "all" || mode == "block");
    const bool runRowBucketed = (mode == "all" || mode == "rowbucketed");
    const bool runLevelAware = (mode == "all" || mode == "levelaware");
    const bool runRandomAware = (mode == "all" || mode == "randomaware");
    const bool runPowerLawAware = (mode == "all" || mode == "powerlawaware");
    const bool runBlockAware = (mode == "all" || mode == "blockaware");
    const bool runBandedAware = (mode == "all" || mode == "bandedaware");
    const bool runFamilyAware = (mode == "all" || mode == "familyaware");
    const bool runLearnedAware = (mode == "all" || mode == "learnedaware");
    const bool runCusparse = (mode == "all" || mode == "cusparse");

    const CsrMatrix matrix = generateLowerTriangular(n, density, 42);
    const std::vector<double> rhs = generateRandomRhs(n, 99);

    std::vector<double> cpuSolution;
    std::vector<double> naiveSolution;
    std::vector<double> coalescedSolution;
    std::vector<double> threadRowSolution;
    std::vector<double> warpRowSolution;
    std::vector<double> adaptiveRowSolution;
    std::vector<double> blockRowSolution;
    std::vector<double> rowBucketedSolution;
    std::vector<double> levelAwareSolution;
    std::vector<double> randomAwareSolution;
    std::vector<double> powerLawAwareSolution;
    std::vector<double> blockAwareSolution;
    std::vector<double> bandedAwareSolution;
    std::vector<double> familyAwareSolution;
    std::vector<double> learnedAwareSolution;
    std::vector<double> cusparseSolution;

    const double cpuMs = timeMs([&]() { sptrsvCpuReference(matrix, rhs, cpuSolution); });
    const double cpuResidual = computeResidual(matrix, cpuSolution, rhs);

    bool naiveOk = false;
    double naiveMs = 0.0;
    double naiveResidual = 0.0;
    if (runNaive) {
        naiveMs = timeMs([&]() { naiveOk = sptrsvGpuNaive(matrix, rhs, naiveSolution); });
    }
    if (runNaive && naiveOk) {
        naiveResidual = computeResidual(matrix, naiveSolution, rhs);
    }

    bool coalescedOk = false;
    double coalescedMs = 0.0;
    double coalescedResidual = 0.0;
    if (runCoalesced) {
        coalescedMs = timeMs([&]() { coalescedOk = sptrsvGpuCoalesced(matrix, rhs, coalescedSolution); });
    }
    if (runCoalesced && coalescedOk) {
        coalescedResidual = computeResidual(matrix, coalescedSolution, rhs);
    }

    bool threadRowOk = false;
    double threadRowMs = 0.0;
    double threadRowResidual = 0.0;
    if (runThreadRow) {
        threadRowMs = timeMs([&]() { threadRowOk = sptrsvGpuThreadPerRow(matrix, rhs, threadRowSolution); });
    }
    if (runThreadRow && threadRowOk) {
        threadRowResidual = computeResidual(matrix, threadRowSolution, rhs);
    }

    bool warpRowOk = false;
    double warpRowMs = 0.0;
    double warpRowResidual = 0.0;
    if (runWarpRow) {
        warpRowMs = timeMs([&]() { warpRowOk = sptrsvGpuWarpPerRow(matrix, rhs, warpRowSolution); });
    }
    if (runWarpRow && warpRowOk) {
        warpRowResidual = computeResidual(matrix, warpRowSolution, rhs);
    }

    bool adaptiveRowOk = false;
    double adaptiveRowMs = 0.0;
    double adaptiveRowResidual = 0.0;
    if (runAdaptiveRow) {
        adaptiveRowMs = timeMs([&]() { adaptiveRowOk = sptrsvGpuAdaptiveRow(matrix, rhs, adaptiveRowSolution); });
    }
    if (runAdaptiveRow && adaptiveRowOk) {
        adaptiveRowResidual = computeResidual(matrix, adaptiveRowSolution, rhs);
    }

    bool blockRowOk = false;
    double blockRowMs = 0.0;
    double blockRowResidual = 0.0;
    if (runBlockRow) {
        blockRowMs = timeMs([&]() { blockRowOk = sptrsvGpuBlockPerRow(matrix, rhs, blockRowSolution); });
    }
    if (runBlockRow && blockRowOk) {
        blockRowResidual = computeResidual(matrix, blockRowSolution, rhs);
    }

    bool rowBucketedOk = false;
    double rowBucketedMs = 0.0;
    double rowBucketedResidual = 0.0;
    if (runRowBucketed) {
        rowBucketedMs = timeMs([&]() { rowBucketedOk = sptrsvGpuRowBucketed(matrix, rhs, rowBucketedSolution); });
    }
    if (runRowBucketed && rowBucketedOk) {
        rowBucketedResidual = computeResidual(matrix, rowBucketedSolution, rhs);
    }

    bool levelAwareOk = false;
    double levelAwareMs = 0.0;
    double levelAwareResidual = 0.0;
    if (runLevelAware) {
        levelAwareMs = timeMs([&]() { levelAwareOk = sptrsvGpuLevelAware(matrix, rhs, levelAwareSolution); });
    }
    if (runLevelAware && levelAwareOk) {
        levelAwareResidual = computeResidual(matrix, levelAwareSolution, rhs);
    }

    bool randomAwareOk = false;
    double randomAwareMs = 0.0;
    double randomAwareResidual = 0.0;
    if (runRandomAware) {
        randomAwareMs = timeMs([&]() { randomAwareOk = sptrsvGpuRandomLevelAware(matrix, rhs, randomAwareSolution); });
    }
    if (runRandomAware && randomAwareOk) {
        randomAwareResidual = computeResidual(matrix, randomAwareSolution, rhs);
    }

    bool powerLawAwareOk = false;
    double powerLawAwareMs = 0.0;
    double powerLawAwareResidual = 0.0;
    if (runPowerLawAware) {
        powerLawAwareMs = timeMs([&]() { powerLawAwareOk = sptrsvGpuPowerLawAware(matrix, rhs, powerLawAwareSolution); });
    }
    if (runPowerLawAware && powerLawAwareOk) {
        powerLawAwareResidual = computeResidual(matrix, powerLawAwareSolution, rhs);
    }

    bool blockAwareOk = false;
    double blockAwareMs = 0.0;
    double blockAwareResidual = 0.0;
    if (runBlockAware) {
        blockAwareMs = timeMs([&]() { blockAwareOk = sptrsvGpuBlockAware(matrix, rhs, blockAwareSolution); });
    }
    if (runBlockAware && blockAwareOk) {
        blockAwareResidual = computeResidual(matrix, blockAwareSolution, rhs);
    }

    bool bandedAwareOk = false;
    double bandedAwareMs = 0.0;
    double bandedAwareResidual = 0.0;
    if (runBandedAware) {
        bandedAwareMs = timeMs([&]() { bandedAwareOk = sptrsvGpuBandedAware(matrix, rhs, bandedAwareSolution); });
    }
    if (runBandedAware && bandedAwareOk) {
        bandedAwareResidual = computeResidual(matrix, bandedAwareSolution, rhs);
    }

    bool familyAwareOk = false;
    double familyAwareMs = 0.0;
    double familyAwareResidual = 0.0;
    if (runFamilyAware) {
        familyAwareMs = timeMs([&]() { familyAwareOk = sptrsvGpuFamilyAware(matrix, rhs, familyAwareSolution); });
    }
    if (runFamilyAware && familyAwareOk) {
        familyAwareResidual = computeResidual(matrix, familyAwareSolution, rhs);
    }

    bool learnedAwareOk = false;
    double learnedAwareMs = 0.0;
    double learnedAwareResidual = 0.0;
    if (runLearnedAware) {
        learnedAwareMs = timeMs([&]() { learnedAwareOk = sptrsvGpuLearnedAware(matrix, rhs, learnedAwareSolution); });
    }
    if (runLearnedAware && learnedAwareOk) {
        learnedAwareResidual = computeResidual(matrix, learnedAwareSolution, rhs);
    }

    bool cusparseOk = false;
    double cusparseMs = 0.0;
    double cusparseResidual = 0.0;
    if (runCusparse) {
        cusparseMs = timeMs([&]() { cusparseOk = sptrsvCuSparse(matrix, rhs, cusparseSolution); });
    }
    if (runCusparse && cusparseOk) {
        cusparseResidual = computeResidual(matrix, cusparseSolution, rhs);
    }

    std::cout << "GPU New SpTRSV Baseline Benchmark\n";
    std::cout << "Matrix size : " << n << "\n";
    std::cout << "Density     : " << density << "\n";
    std::cout << "NNZ         : " << matrix.nnz << "\n";
    printDeviceInfo();
    std::cout << "\n";

    std::cout << std::left
              << std::setw(18) << "Solver"
              << std::setw(16) << "Time (ms)"
              << std::setw(16) << "Residual"
              << "\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << std::setw(18) << "CPU Ref"
              << std::setw(16) << std::fixed << std::setprecision(4) << cpuMs
              << std::setw(16) << std::scientific << cpuResidual
              << "\n";
    if (runNaive && naiveOk) {
        std::cout << std::setw(18) << "GPU Naive"
                  << std::setw(16) << std::fixed << std::setprecision(4) << naiveMs
                  << std::setw(16) << std::scientific << naiveResidual
                  << "\n";
    } else if (runNaive) {
        std::cout << std::setw(18) << "GPU Naive"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runCoalesced && coalescedOk) {
        std::cout << std::setw(18) << "GPU Coalesced"
                  << std::setw(16) << std::fixed << std::setprecision(4) << coalescedMs
                  << std::setw(16) << std::scientific << coalescedResidual
                  << "\n";
    } else if (runCoalesced) {
        std::cout << std::setw(18) << "GPU Coalesced"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runThreadRow && threadRowOk) {
        std::cout << std::setw(18) << "GPU ThreadRow"
                  << std::setw(16) << std::fixed << std::setprecision(4) << threadRowMs
                  << std::setw(16) << std::scientific << threadRowResidual
                  << "\n";
    } else if (runThreadRow) {
        std::cout << std::setw(18) << "GPU ThreadRow"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runWarpRow && warpRowOk) {
        std::cout << std::setw(18) << "GPU WarpRow"
                  << std::setw(16) << std::fixed << std::setprecision(4) << warpRowMs
                  << std::setw(16) << std::scientific << warpRowResidual
                  << "\n";
    } else if (runWarpRow) {
        std::cout << std::setw(18) << "GPU WarpRow"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runAdaptiveRow && adaptiveRowOk) {
        std::cout << std::setw(18) << "GPU AdaptiveRow"
                  << std::setw(16) << std::fixed << std::setprecision(4) << adaptiveRowMs
                  << std::setw(16) << std::scientific << adaptiveRowResidual
                  << "\n";
    } else if (runAdaptiveRow) {
        std::cout << std::setw(18) << "GPU AdaptiveRow"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runBlockRow && blockRowOk) {
        std::cout << std::setw(18) << "GPU BlockRow"
                  << std::setw(16) << std::fixed << std::setprecision(4) << blockRowMs
                  << std::setw(16) << std::scientific << blockRowResidual
                  << "\n";
    } else if (runBlockRow) {
        std::cout << std::setw(18) << "GPU BlockRow"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runRowBucketed && rowBucketedOk) {
        std::cout << std::setw(18) << "GPU RowBucketed"
                  << std::setw(16) << std::fixed << std::setprecision(4) << rowBucketedMs
                  << std::setw(16) << std::scientific << rowBucketedResidual
                  << "\n";
    } else if (runRowBucketed) {
        std::cout << std::setw(18) << "GPU RowBucketed"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runLevelAware && levelAwareOk) {
        std::cout << std::setw(18) << "GPU LevelAware"
                  << std::setw(16) << std::fixed << std::setprecision(4) << levelAwareMs
                  << std::setw(16) << std::scientific << levelAwareResidual
                  << "\n";
    } else if (runLevelAware) {
        std::cout << std::setw(18) << "GPU LevelAware"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runRandomAware && randomAwareOk) {
        std::cout << std::setw(18) << "GPU RandomAware"
                  << std::setw(16) << std::fixed << std::setprecision(4) << randomAwareMs
                  << std::setw(16) << std::scientific << randomAwareResidual
                  << "\n";
    } else if (runRandomAware) {
        std::cout << std::setw(18) << "GPU RandomAware"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runPowerLawAware && powerLawAwareOk) {
        std::cout << std::setw(18) << "GPU PowerLawAware"
                  << std::setw(16) << std::fixed << std::setprecision(4) << powerLawAwareMs
                  << std::setw(16) << std::scientific << powerLawAwareResidual
                  << "\n";
    } else if (runPowerLawAware) {
        std::cout << std::setw(18) << "GPU PowerLawAware"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runBlockAware && blockAwareOk) {
        std::cout << std::setw(18) << "GPU BlockAware"
                  << std::setw(16) << std::fixed << std::setprecision(4) << blockAwareMs
                  << std::setw(16) << std::scientific << blockAwareResidual
                  << "\n";
    } else if (runBlockAware) {
        std::cout << std::setw(18) << "GPU BlockAware"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runBandedAware && bandedAwareOk) {
        std::cout << std::setw(18) << "GPU BandedAware"
                  << std::setw(16) << std::fixed << std::setprecision(4) << bandedAwareMs
                  << std::setw(16) << std::scientific << bandedAwareResidual
                  << "\n";
    } else if (runBandedAware) {
        std::cout << std::setw(18) << "GPU BandedAware"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runFamilyAware && familyAwareOk) {
        std::cout << std::setw(18) << "GPU FamilyAware"
                  << std::setw(16) << std::fixed << std::setprecision(4) << familyAwareMs
                  << std::setw(16) << std::scientific << familyAwareResidual
                  << "\n";
    } else if (runFamilyAware) {
        std::cout << std::setw(18) << "GPU FamilyAware"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runLearnedAware && learnedAwareOk) {
        std::cout << std::setw(18) << "GPU LearnedAware"
                  << std::setw(16) << std::fixed << std::setprecision(4) << learnedAwareMs
                  << std::setw(16) << std::scientific << learnedAwareResidual
                  << "\n";
    } else if (runLearnedAware) {
        std::cout << std::setw(18) << "GPU LearnedAware"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }
    if (runCusparse && cusparseOk) {
        std::cout << std::setw(18) << "cuSPARSE"
                  << std::setw(16) << std::fixed << std::setprecision(4) << cusparseMs
                  << std::setw(16) << std::scientific << cusparseResidual
                  << "\n";
    } else if (runCusparse) {
        std::cout << std::setw(18) << "cuSPARSE"
                  << std::setw(16) << "failed"
                  << std::setw(16) << "-"
                  << "\n";
    }

    return 0;
}
