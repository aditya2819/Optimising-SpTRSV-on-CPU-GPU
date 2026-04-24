#include "sptrsv_kernel.h"

#include <cstddef>
#include <omp.h>

namespace {

void solveBasicRow(const CsrMatrix& matrix,
                   const std::vector<double>& rhs,
                   const std::vector<double>& solution,
                   int row,
                   double& out)
{
    double rowRhs = rhs[static_cast<std::size_t>(row)];
    double diagonal = 1.0;

    for (int entry = matrix.rowPtr[static_cast<std::size_t>(row)];
         entry < matrix.rowPtr[static_cast<std::size_t>(row + 1)];
         ++entry) {
        const int col = matrix.colIdx[static_cast<std::size_t>(entry)];
        const double value = matrix.values[static_cast<std::size_t>(entry)];
        if (col < row) {
            rowRhs -= value * solution[static_cast<std::size_t>(col)];
        } else if (col == row) {
            diagonal = value;
        }
    }

    out = rowRhs / diagonal;
}

void solveBranchReducedRow(const CsrMatrix& matrix,
                           const std::vector<double>& rhs,
                           const std::vector<double>& solution,
                           int row,
                           double& out)
{
    double rowRhs = rhs[static_cast<std::size_t>(row)];
    const int rowStart = matrix.rowPtr[static_cast<std::size_t>(row)];
    const int rowEnd = matrix.rowPtr[static_cast<std::size_t>(row + 1)];
    int diagonalEntry = rowEnd - 1;

    while (diagonalEntry >= rowStart &&
           matrix.colIdx[static_cast<std::size_t>(diagonalEntry)] != row) {
        --diagonalEntry;
    }

    for (int entry = rowStart; entry < rowEnd; ++entry) {
        if (entry == diagonalEntry) {
            continue;
        }
        const int col = matrix.colIdx[static_cast<std::size_t>(entry)];
        if (col < row) {
            rowRhs -= matrix.values[static_cast<std::size_t>(entry)] *
                      solution[static_cast<std::size_t>(col)];
        }
    }

    const double diagonal =
        (diagonalEntry >= rowStart)
            ? matrix.values[static_cast<std::size_t>(diagonalEntry)]
            : 1.0;
    out = rowRhs / diagonal;
}

void solveUnrolledRow(const CsrMatrix& matrix,
                      const std::vector<double>& rhs,
                      const std::vector<double>& solution,
                      int row,
                      double& out)
{
    double rowRhs = rhs[static_cast<std::size_t>(row)];
    const int rowStart = matrix.rowPtr[static_cast<std::size_t>(row)];
    const int rowEnd = matrix.rowPtr[static_cast<std::size_t>(row + 1)];
    int diagonalEntry = rowEnd - 1;

    while (diagonalEntry >= rowStart &&
           matrix.colIdx[static_cast<std::size_t>(diagonalEntry)] != row) {
        --diagonalEntry;
    }

    int entry = rowStart;
    while (entry + 3 < rowEnd) {
        double accum = 0.0;

        if (entry != diagonalEntry) {
            const int col0 = matrix.colIdx[static_cast<std::size_t>(entry)];
            if (col0 < row) {
                accum += matrix.values[static_cast<std::size_t>(entry)] *
                         solution[static_cast<std::size_t>(col0)];
            }
        }
        if (entry + 1 != diagonalEntry) {
            const int col1 = matrix.colIdx[static_cast<std::size_t>(entry + 1)];
            if (col1 < row) {
                accum += matrix.values[static_cast<std::size_t>(entry + 1)] *
                         solution[static_cast<std::size_t>(col1)];
            }
        }
        if (entry + 2 != diagonalEntry) {
            const int col2 = matrix.colIdx[static_cast<std::size_t>(entry + 2)];
            if (col2 < row) {
                accum += matrix.values[static_cast<std::size_t>(entry + 2)] *
                         solution[static_cast<std::size_t>(col2)];
            }
        }
        if (entry + 3 != diagonalEntry) {
            const int col3 = matrix.colIdx[static_cast<std::size_t>(entry + 3)];
            if (col3 < row) {
                accum += matrix.values[static_cast<std::size_t>(entry + 3)] *
                         solution[static_cast<std::size_t>(col3)];
            }
        }

        rowRhs -= accum;
        entry += 4;
    }

    for (; entry < rowEnd; ++entry) {
        if (entry == diagonalEntry) {
            continue;
        }
        const int col = matrix.colIdx[static_cast<std::size_t>(entry)];
        if (col < row) {
            rowRhs -= matrix.values[static_cast<std::size_t>(entry)] *
                      solution[static_cast<std::size_t>(col)];
        }
    }

    const double diagonal =
        (diagonalEntry >= rowStart)
            ? matrix.values[static_cast<std::size_t>(diagonalEntry)]
            : 1.0;
    out = rowRhs / diagonal;
}

std::size_t levelWorkEstimate(const CsrMatrix& matrix, const std::vector<int>& levelRows)
{
    std::size_t work = 0;
    for (int row : levelRows) {
        work += static_cast<std::size_t>(
            matrix.rowPtr[static_cast<std::size_t>(row + 1)] -
            matrix.rowPtr[static_cast<std::size_t>(row)]);
    }
    return work;
}

}  // namespace

LevelSchedule buildLevelSchedule(const CsrMatrix& matrix)
{
    LevelSchedule schedule;
    schedule.rowLevel.assign(static_cast<std::size_t>(matrix.n), 0);

    int maxLevel = 0;
    for (int row = 0; row < matrix.n; ++row) {
        int level = 0;
        for (int entry = matrix.rowPtr[static_cast<std::size_t>(row)];
             entry < matrix.rowPtr[static_cast<std::size_t>(row + 1)];
             ++entry) {
            const int col = matrix.colIdx[static_cast<std::size_t>(entry)];
            if (col < row) {
                level = std::max(level, schedule.rowLevel[static_cast<std::size_t>(col)] + 1);
            }
        }
        schedule.rowLevel[static_cast<std::size_t>(row)] = level;
        maxLevel = std::max(maxLevel, level);
    }

    schedule.levels.resize(static_cast<std::size_t>(maxLevel + 1));
    for (int row = 0; row < matrix.n; ++row) {
        schedule.levels[static_cast<std::size_t>(
            schedule.rowLevel[static_cast<std::size_t>(row)])].push_back(row);
    }

    return schedule;
}

void sptrsvCpuBasic(const CsrMatrix& matrix,
                    const std::vector<double>& rhs,
                    std::vector<double>& solution)
{
    solution.assign(static_cast<std::size_t>(matrix.n), 0.0);

    for (int row = 0; row < matrix.n; ++row) {
        double rowRhs = rhs[static_cast<std::size_t>(row)];
        double diagonal = 1.0;

        for (int entry = matrix.rowPtr[static_cast<std::size_t>(row)];
             entry < matrix.rowPtr[static_cast<std::size_t>(row + 1)];
             ++entry) {
            const int col = matrix.colIdx[static_cast<std::size_t>(entry)];
            const double value = matrix.values[static_cast<std::size_t>(entry)];
            if (col < row) {
                rowRhs -= value * solution[static_cast<std::size_t>(col)];
            } else if (col == row) {
                diagonal = value;
            }
        }

        solution[static_cast<std::size_t>(row)] = rowRhs / diagonal;
    }
}

void sptrsvCpuBranchReduced(const CsrMatrix& matrix,
                            const std::vector<double>& rhs,
                            std::vector<double>& solution)
{
    solution.assign(static_cast<std::size_t>(matrix.n), 0.0);

    for (int row = 0; row < matrix.n; ++row) {
        double rowRhs = rhs[static_cast<std::size_t>(row)];
        const int rowStart = matrix.rowPtr[static_cast<std::size_t>(row)];
        const int rowEnd = matrix.rowPtr[static_cast<std::size_t>(row + 1)];
        int diagonalEntry = rowEnd - 1;

        while (diagonalEntry >= rowStart &&
               matrix.colIdx[static_cast<std::size_t>(diagonalEntry)] != row) {
            --diagonalEntry;
        }

        for (int entry = rowStart; entry < rowEnd; ++entry) {
            if (entry == diagonalEntry) {
                continue;
            }
            const int col = matrix.colIdx[static_cast<std::size_t>(entry)];
            if (col < row) {
                rowRhs -= matrix.values[static_cast<std::size_t>(entry)] *
                          solution[static_cast<std::size_t>(col)];
            }
        }

        const double diagonal =
            (diagonalEntry >= rowStart)
                ? matrix.values[static_cast<std::size_t>(diagonalEntry)]
                : 1.0;
        solution[static_cast<std::size_t>(row)] = rowRhs / diagonal;
    }
}

void sptrsvCpuUnrolled(const CsrMatrix& matrix,
                       const std::vector<double>& rhs,
                       std::vector<double>& solution)
{
    solution.assign(static_cast<std::size_t>(matrix.n), 0.0);

    for (int row = 0; row < matrix.n; ++row) {
        double rowRhs = rhs[static_cast<std::size_t>(row)];
        const int rowStart = matrix.rowPtr[static_cast<std::size_t>(row)];
        const int rowEnd = matrix.rowPtr[static_cast<std::size_t>(row + 1)];
        int diagonalEntry = rowEnd - 1;

        while (diagonalEntry >= rowStart &&
               matrix.colIdx[static_cast<std::size_t>(diagonalEntry)] != row) {
            --diagonalEntry;
        }

        int entry = rowStart;
        while (entry + 3 < rowEnd) {
            double accum = 0.0;

            if (entry != diagonalEntry) {
                const int col0 = matrix.colIdx[static_cast<std::size_t>(entry)];
                if (col0 < row) {
                    accum += matrix.values[static_cast<std::size_t>(entry)] *
                             solution[static_cast<std::size_t>(col0)];
                }
            }
            if (entry + 1 != diagonalEntry) {
                const int col1 = matrix.colIdx[static_cast<std::size_t>(entry + 1)];
                if (col1 < row) {
                    accum += matrix.values[static_cast<std::size_t>(entry + 1)] *
                             solution[static_cast<std::size_t>(col1)];
                }
            }
            if (entry + 2 != diagonalEntry) {
                const int col2 = matrix.colIdx[static_cast<std::size_t>(entry + 2)];
                if (col2 < row) {
                    accum += matrix.values[static_cast<std::size_t>(entry + 2)] *
                             solution[static_cast<std::size_t>(col2)];
                }
            }
            if (entry + 3 != diagonalEntry) {
                const int col3 = matrix.colIdx[static_cast<std::size_t>(entry + 3)];
                if (col3 < row) {
                    accum += matrix.values[static_cast<std::size_t>(entry + 3)] *
                             solution[static_cast<std::size_t>(col3)];
                }
            }

            rowRhs -= accum;
            entry += 4;
        }

        for (; entry < rowEnd; ++entry) {
            if (entry == diagonalEntry) {
                continue;
            }
            const int col = matrix.colIdx[static_cast<std::size_t>(entry)];
            if (col < row) {
                rowRhs -= matrix.values[static_cast<std::size_t>(entry)] *
                          solution[static_cast<std::size_t>(col)];
            }
        }

        const double diagonal =
            (diagonalEntry >= rowStart)
                ? matrix.values[static_cast<std::size_t>(diagonalEntry)]
                : 1.0;
        solution[static_cast<std::size_t>(row)] = rowRhs / diagonal;
    }
}

void sptrsvCpuThresholdOmp(const CsrMatrix& matrix,
                           const LevelSchedule& schedule,
                           const std::vector<double>& rhs,
                           std::vector<double>& solution)
{
    solution.assign(static_cast<std::size_t>(matrix.n), 0.0);

    constexpr int kMinParallelRows = 2048;
    constexpr std::size_t kMinParallelWork = 65536;

    for (const auto& levelRows : schedule.levels) {
        const bool runParallel =
            static_cast<int>(levelRows.size()) >= kMinParallelRows &&
            levelWorkEstimate(matrix, levelRows) >= kMinParallelWork;

        if (!runParallel) {
            for (int row : levelRows) {
                double rowValue = 0.0;
                solveUnrolledRow(matrix, rhs, solution, row, rowValue);
                solution[static_cast<std::size_t>(row)] = rowValue;
            }
        } else {
#pragma omp parallel for schedule(static) num_threads(2)
            for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(levelRows.size()); ++idx) {
                const int row = levelRows[static_cast<std::size_t>(idx)];
                double rowValue = 0.0;
                solveUnrolledRow(matrix, rhs, solution, row, rowValue);
                solution[static_cast<std::size_t>(row)] = rowValue;
            }
        }
    }
}

void sptrsvCpuChunkedOmp(const CsrMatrix& matrix,
                         const LevelSchedule& schedule,
                         const std::vector<double>& rhs,
                         std::vector<double>& solution)
{
    solution.assign(static_cast<std::size_t>(matrix.n), 0.0);

    constexpr int kMinParallelRows = 1024;
    constexpr std::size_t kMinParallelWork = 32768;
    constexpr int kChunkRows = 256;

    for (const auto& levelRows : schedule.levels) {
        const bool runParallel =
            static_cast<int>(levelRows.size()) >= kMinParallelRows &&
            levelWorkEstimate(matrix, levelRows) >= kMinParallelWork;

        if (!runParallel) {
            for (int row : levelRows) {
                double rowValue = 0.0;
                solveUnrolledRow(matrix, rhs, solution, row, rowValue);
                solution[static_cast<std::size_t>(row)] = rowValue;
            }
        } else {
#pragma omp parallel for schedule(static, kChunkRows) num_threads(2)
            for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(levelRows.size()); ++idx) {
                const int row = levelRows[static_cast<std::size_t>(idx)];
                double rowValue = 0.0;
                solveUnrolledRow(matrix, rhs, solution, row, rowValue);
                solution[static_cast<std::size_t>(row)] = rowValue;
            }
        }
    }
}

void sptrsvCpuCappedChunkOmp(const CsrMatrix& matrix,
                             const LevelSchedule& schedule,
                             const std::vector<double>& rhs,
                             std::vector<double>& solution)
{
    solution.assign(static_cast<std::size_t>(matrix.n), 0.0);

    constexpr int kMinParallelRows = 1024;
    constexpr std::size_t kMinParallelWork = 32768;
    constexpr int kChunkRows = 256;

    for (const auto& levelRows : schedule.levels) {
        const std::size_t levelWork = levelWorkEstimate(matrix, levelRows);
        const bool runParallel =
            static_cast<int>(levelRows.size()) >= kMinParallelRows &&
            levelWork >= kMinParallelWork;

        int threadCap = 1;
        if (levelWork >= 65536 || static_cast<int>(levelRows.size()) >= 2048) {
            threadCap = 2;
        }

        if (!runParallel || threadCap == 1) {
            for (int row : levelRows) {
                double rowValue = 0.0;
                solveUnrolledRow(matrix, rhs, solution, row, rowValue);
                solution[static_cast<std::size_t>(row)] = rowValue;
            }
        } else {
#pragma omp parallel for schedule(static, kChunkRows) num_threads(2)
            for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(levelRows.size()); ++idx) {
                const int row = levelRows[static_cast<std::size_t>(idx)];
                double rowValue = 0.0;
                solveUnrolledRow(matrix, rhs, solution, row, rowValue);
                solution[static_cast<std::size_t>(row)] = rowValue;
            }
        }
    }
}
