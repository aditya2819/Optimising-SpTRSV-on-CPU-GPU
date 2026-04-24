#pragma once

#include <vector>

struct CsrMatrix {
    int n = 0;
    int nnz = 0;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<double> values;
};

struct LevelSchedule {
    std::vector<int> rowLevel;
    std::vector<std::vector<int>> levels;
};

LevelSchedule buildLevelSchedule(const CsrMatrix& matrix);

void sptrsvCpuBasic(const CsrMatrix& matrix,
                    const std::vector<double>& rhs,
                    std::vector<double>& solution);

void sptrsvCpuBranchReduced(const CsrMatrix& matrix,
                            const std::vector<double>& rhs,
                            std::vector<double>& solution);

void sptrsvCpuUnrolled(const CsrMatrix& matrix,
                       const std::vector<double>& rhs,
                       std::vector<double>& solution);

void sptrsvCpuThresholdOmp(const CsrMatrix& matrix,
                           const LevelSchedule& schedule,
                           const std::vector<double>& rhs,
                           std::vector<double>& solution);

void sptrsvCpuChunkedOmp(const CsrMatrix& matrix,
                         const LevelSchedule& schedule,
                         const std::vector<double>& rhs,
                         std::vector<double>& solution);

void sptrsvCpuCappedChunkOmp(const CsrMatrix& matrix,
                             const LevelSchedule& schedule,
                             const std::vector<double>& rhs,
                             std::vector<double>& solution);
