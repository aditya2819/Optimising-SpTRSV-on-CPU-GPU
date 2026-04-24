#include "sptrsv_kernel.cuh"

__global__ void sptrsvNaiveKernel(
    int n,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (int row = 0; row < n; ++row) {
        double rowRhs = rhs[row];
        double diagonal = 1.0;
        for (int entry = rowPtr[row]; entry < rowPtr[row + 1]; ++entry) {
            const int col = colIdx[entry];
            const double value = values[entry];
            if (col < row) {
                rowRhs -= value * solution[col];
            } else if (col == row) {
                diagonal = value;
            }
        }
        solution[row] = rowRhs / diagonal;
    }
}

__global__ void sptrsvCoalescedKernel(
    int n,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution)
{
    if (blockIdx.x != 0) {
        return;
    }

    extern __shared__ double shared[];
    double* partialSums = shared;
    double* partialDiags = shared + blockDim.x;

    for (int row = 0; row < n; ++row) {
        const int rowStart = rowPtr[row];
        const int rowEnd = rowPtr[row + 1];
        double partial = 0.0;
        double diagonal = 0.0;

        for (int entry = rowStart + threadIdx.x; entry < rowEnd; entry += blockDim.x) {
            const int col = colIdx[entry];
            const double value = values[entry];
            if (col < row) {
                partial += value * solution[col];
            } else if (col == row) {
                diagonal = value;
            }
        }

        partialSums[threadIdx.x] = partial;
        partialDiags[threadIdx.x] = diagonal;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
                if (partialDiags[threadIdx.x] == 0.0) {
                    partialDiags[threadIdx.x] = partialDiags[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            solution[row] = (rhs[row] - partialSums[0]) / partialDiags[0];
        }
        __syncthreads();
    }
}

__global__ void sptrsvThreadPerRowKernel(
    int row,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    double rowRhs = rhs[row];
    double diagonal = 1.0;
    for (int entry = rowPtr[row]; entry < rowPtr[row + 1]; ++entry) {
        const int col = colIdx[entry];
        const double value = values[entry];
        if (col < row) {
            rowRhs -= value * solution[col];
        } else if (col == row) {
            diagonal = value;
        }
    }
    solution[row] = rowRhs / diagonal;
}

__global__ void sptrsvWarpPerRowKernel(
    int row,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution)
{
    if (blockIdx.x != 0 || threadIdx.x >= warpSize) {
        return;
    }

    double partial = 0.0;
    double diagonal = 0.0;
    const int rowStart = rowPtr[row];
    const int rowEnd = rowPtr[row + 1];

    for (int entry = rowStart + threadIdx.x; entry < rowEnd; entry += warpSize) {
        const int col = colIdx[entry];
        const double value = values[entry];
        if (col < row) {
            partial += value * solution[col];
        } else if (col == row) {
            diagonal = value;
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
        const double otherDiagonal = __shfl_down_sync(0xffffffffu, diagonal, offset);
        if (diagonal == 0.0) {
            diagonal = otherDiagonal;
        }
    }

    if (threadIdx.x == 0) {
        solution[row] = (rhs[row] - partial) / diagonal;
    }
}

__global__ void sptrsvBlockPerRowKernel(
    int row,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution)
{
    extern __shared__ double shared[];
    double* partialSums = shared;
    double* partialDiags = shared + blockDim.x;

    const int rowStart = rowPtr[row];
    const int rowEnd = rowPtr[row + 1];
    double partial = 0.0;
    double diagonal = 0.0;

    for (int entry = rowStart + threadIdx.x; entry < rowEnd; entry += blockDim.x) {
        const int col = colIdx[entry];
        const double value = values[entry];
        if (col < row) {
            partial += value * solution[col];
        } else if (col == row) {
            diagonal = value;
        }
    }

    partialSums[threadIdx.x] = partial;
    partialDiags[threadIdx.x] = diagonal;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
            if (partialDiags[threadIdx.x] == 0.0) {
                partialDiags[threadIdx.x] = partialDiags[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        solution[row] = (rhs[row] - partialSums[0]) / partialDiags[0];
    }
}

__global__ void sptrsvThreadBucketedKernel(
    int rowBegin,
    int rowEnd,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (int row = rowBegin; row < rowEnd; ++row) {
        double rowRhs = rhs[row];
        double diagonal = 1.0;
        for (int entry = rowPtr[row]; entry < rowPtr[row + 1]; ++entry) {
            const int col = colIdx[entry];
            const double value = values[entry];
            if (col < row) {
                rowRhs -= value * solution[col];
            } else if (col == row) {
                diagonal = value;
            }
        }
        solution[row] = rowRhs / diagonal;
    }
}

__global__ void sptrsvWarpBucketedKernel(
    int rowBegin,
    int rowEnd,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution)
{
    if (blockIdx.x != 0 || threadIdx.x >= warpSize) {
        return;
    }

    for (int row = rowBegin; row < rowEnd; ++row) {
        double partial = 0.0;
        double diagonal = 0.0;
        const int rowStart = rowPtr[row];
        const int rowEndPtr = rowPtr[row + 1];

        for (int entry = rowStart + threadIdx.x; entry < rowEndPtr; entry += warpSize) {
            const int col = colIdx[entry];
            const double value = values[entry];
            if (col < row) {
                partial += value * solution[col];
            } else if (col == row) {
                diagonal = value;
            }
        }

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffffu, partial, offset);
            const double otherDiagonal = __shfl_down_sync(0xffffffffu, diagonal, offset);
            if (diagonal == 0.0) {
                diagonal = otherDiagonal;
            }
        }

        if (threadIdx.x == 0) {
            solution[row] = (rhs[row] - partial) / diagonal;
        }
    }
}

__global__ void sptrsvBlockBucketedKernel(
    int rowBegin,
    int rowEnd,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution)
{
    extern __shared__ double shared[];
    double* partialSums = shared;
    double* partialDiags = shared + blockDim.x;

    for (int row = rowBegin; row < rowEnd; ++row) {
        const int rowStart = rowPtr[row];
        const int rowEndPtr = rowPtr[row + 1];
        double partial = 0.0;
        double diagonal = 0.0;

        for (int entry = rowStart + threadIdx.x; entry < rowEndPtr; entry += blockDim.x) {
            const int col = colIdx[entry];
            const double value = values[entry];
            if (col < row) {
                partial += value * solution[col];
            } else if (col == row) {
                diagonal = value;
            }
        }

        partialSums[threadIdx.x] = partial;
        partialDiags[threadIdx.x] = diagonal;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
                if (partialDiags[threadIdx.x] == 0.0) {
                    partialDiags[threadIdx.x] = partialDiags[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            solution[row] = (rhs[row] - partialSums[0]) / partialDiags[0];
        }
        __syncthreads();
    }
}

__global__ void sptrsvThreadRowListKernel(
    int count,
    const int* rows,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    const int row = rows[idx];
    double rowRhs = rhs[row];
    double diagonal = 1.0;
    for (int entry = rowPtr[row]; entry < rowPtr[row + 1]; ++entry) {
        const int col = colIdx[entry];
        const double value = values[entry];
        if (col < row) {
            rowRhs -= value * solution[col];
        } else if (col == row) {
            diagonal = value;
        }
    }
    solution[row] = rowRhs / diagonal;
}

__global__ void sptrsvWarpRowListKernel(
    int count,
    const int* rows,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution)
{
    const int globalThread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warpId = globalThread / warpSize;
    const int lane = threadIdx.x & (warpSize - 1);
    if (warpId >= count) {
        return;
    }

    const int row = rows[warpId];
    double partial = 0.0;
    double diagonal = 0.0;
    const int rowStart = rowPtr[row];
    const int rowEnd = rowPtr[row + 1];

    for (int entry = rowStart + lane; entry < rowEnd; entry += warpSize) {
        const int col = colIdx[entry];
        const double value = values[entry];
        if (col < row) {
            partial += value * solution[col];
        } else if (col == row) {
            diagonal = value;
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
        const double otherDiagonal = __shfl_down_sync(0xffffffffu, diagonal, offset);
        if (diagonal == 0.0) {
            diagonal = otherDiagonal;
        }
    }

    if (lane == 0) {
        solution[row] = (rhs[row] - partial) / diagonal;
    }
}

__global__ void sptrsvBlockRowListKernel(
    int count,
    const int* rows,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution)
{
    const int rowListIdx = blockIdx.x;
    if (rowListIdx >= count) {
        return;
    }

    extern __shared__ double shared[];
    double* partialSums = shared;
    double* partialDiags = shared + blockDim.x;

    const int row = rows[rowListIdx];
    const int rowStart = rowPtr[row];
    const int rowEnd = rowPtr[row + 1];
    double partial = 0.0;
    double diagonal = 0.0;

    for (int entry = rowStart + threadIdx.x; entry < rowEnd; entry += blockDim.x) {
        const int col = colIdx[entry];
        const double value = values[entry];
        if (col < row) {
            partial += value * solution[col];
        } else if (col == row) {
            diagonal = value;
        }
    }

    partialSums[threadIdx.x] = partial;
    partialDiags[threadIdx.x] = diagonal;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
            if (partialDiags[threadIdx.x] == 0.0) {
                partialDiags[threadIdx.x] = partialDiags[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        solution[row] = (rhs[row] - partialSums[0]) / partialDiags[0];
    }
}

__global__ void sptrsvBcsrKernel(
    int n,
    int numBlockRows,
    int blockSize,
    const int* blockRowPtr,
    const int* blockColIdx,
    const double* blockValues,
    const double* rhs,
    double* solution)
{
    if (blockIdx.x != 0) {
        return;
    }

    extern __shared__ double shared[];
    double* accum = shared;
    double* diag = accum + blockSize;

    for (int blockRow = 0; blockRow < numBlockRows; ++blockRow) {
        if (threadIdx.x < blockSize) {
            const int row = blockRow * blockSize + threadIdx.x;
            accum[threadIdx.x] = (row < n) ? rhs[row] : 0.0;
        }
        for (int idx = threadIdx.x; idx < blockSize * blockSize; idx += blockDim.x) {
            diag[idx] = 0.0;
        }
        __syncthreads();

        const int rowStart = blockRowPtr[blockRow];
        const int rowEnd = blockRowPtr[blockRow + 1];
        for (int blockEntry = rowStart; blockEntry < rowEnd; ++blockEntry) {
            const int blockCol = blockColIdx[blockEntry];
            const double* values = blockValues + static_cast<size_t>(blockEntry) * blockSize * blockSize;

            if (blockCol < blockRow) {
                for (int idx = threadIdx.x; idx < blockSize * blockSize; idx += blockDim.x) {
                    const int r = idx / blockSize;
                    const int c = idx % blockSize;
                    const int col = blockCol * blockSize + c;
                    if (col < n) {
                        atomicAdd(&accum[r], -values[idx] * solution[col]);
                    }
                }
            } else if (blockCol == blockRow) {
                for (int idx = threadIdx.x; idx < blockSize * blockSize; idx += blockDim.x) {
                    diag[idx] = values[idx];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            double x[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int r = 0; r < blockSize; ++r) {
                double sum = accum[r];
                for (int c = 0; c < r; ++c) {
                    sum -= diag[r * blockSize + c] * x[c];
                }
                x[r] = sum / diag[r * blockSize + r];
            }
            for (int r = 0; r < blockSize; ++r) {
                const int row = blockRow * blockSize + r;
                if (row < n) {
                    solution[row] = x[r];
                }
            }
        }
        __syncthreads();
    }
}

__global__ void sptrsvBcsrRowListKernel(
    int n,
    int count,
    int blockSize,
    const int* rows,
    const int* blockRowPtr,
    const int* blockColIdx,
    const double* blockValues,
    const double* rhs,
    double* solution)
{
    const int rowListIdx = blockIdx.x;
    if (rowListIdx >= count) {
        return;
    }

    const int blockRow = rows[rowListIdx];
    extern __shared__ double shared[];
    double* accum = shared;
    double* diag = accum + blockSize;

    if (threadIdx.x < blockSize) {
        const int row = blockRow * blockSize + threadIdx.x;
        accum[threadIdx.x] = (row < n) ? rhs[row] : 0.0;
    }
    for (int idx = threadIdx.x; idx < blockSize * blockSize; idx += blockDim.x) {
        diag[idx] = 0.0;
    }
    __syncthreads();

    const int rowStart = blockRowPtr[blockRow];
    const int rowEnd = blockRowPtr[blockRow + 1];
    for (int blockEntry = rowStart; blockEntry < rowEnd; ++blockEntry) {
        const int blockCol = blockColIdx[blockEntry];
        const double* values = blockValues + static_cast<size_t>(blockEntry) * blockSize * blockSize;

        if (blockCol < blockRow) {
            for (int idx = threadIdx.x; idx < blockSize * blockSize; idx += blockDim.x) {
                const int r = idx / blockSize;
                const int c = idx % blockSize;
                const int col = blockCol * blockSize + c;
                if (col < n) {
                    atomicAdd(&accum[r], -values[idx] * solution[col]);
                }
            }
        } else if (blockCol == blockRow) {
            for (int idx = threadIdx.x; idx < blockSize * blockSize; idx += blockDim.x) {
                diag[idx] = values[idx];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double x[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int r = 0; r < blockSize; ++r) {
            double sum = accum[r];
            for (int c = 0; c < r; ++c) {
                sum -= diag[r * blockSize + c] * x[c];
            }
            x[r] = sum / diag[r * blockSize + r];
        }
        for (int r = 0; r < blockSize; ++r) {
            const int row = blockRow * blockSize + r;
            if (row < n) {
                solution[row] = x[r];
            }
        }
    }
}

__global__ void sptrsvBcsrChunkKernel(
    int n,
    int blockSize,
    int chunkBegin,
    int chunkEnd,
    const int* blockRowPtr,
    const int* blockColIdx,
    const double* blockValues,
    const double* rhs,
    double* solution)
{
    if (blockIdx.x != 0) {
        return;
    }

    extern __shared__ double shared[];
    double* accum = shared;
    double* diag = accum + blockSize;

    for (int blockRow = chunkBegin; blockRow < chunkEnd; ++blockRow) {
        if (threadIdx.x < blockSize) {
            const int row = blockRow * blockSize + threadIdx.x;
            accum[threadIdx.x] = (row < n) ? rhs[row] : 0.0;
        }
        for (int idx = threadIdx.x; idx < blockSize * blockSize; idx += blockDim.x) {
            diag[idx] = 0.0;
        }
        __syncthreads();

        const int rowStart = blockRowPtr[blockRow];
        const int rowEnd = blockRowPtr[blockRow + 1];
        for (int blockEntry = rowStart; blockEntry < rowEnd; ++blockEntry) {
            const int blockCol = blockColIdx[blockEntry];
            const double* values = blockValues + static_cast<size_t>(blockEntry) * blockSize * blockSize;

            if (blockCol < blockRow) {
                if (threadIdx.x < blockSize) {
                    const int r = static_cast<int>(threadIdx.x);
                    double local = 0.0;
                    const int colBase = blockCol * blockSize;
                    for (int c = 0; c < blockSize; ++c) {
                        const int col = colBase + c;
                        if (col < n) {
                            local += values[r * blockSize + c] * solution[col];
                        }
                    }
                    accum[r] -= local;
                }
            } else if (blockCol == blockRow) {
                for (int idx = threadIdx.x; idx < blockSize * blockSize; idx += blockDim.x) {
                    diag[idx] = values[idx];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            double x[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int r = 0; r < blockSize; ++r) {
                double sum = accum[r];
                for (int c = 0; c < r; ++c) {
                    sum -= diag[r * blockSize + c] * x[c];
                }
                x[r] = sum / diag[r * blockSize + r];
            }
            for (int r = 0; r < blockSize; ++r) {
                const int row = blockRow * blockSize + r;
                if (row < n) {
                    solution[row] = x[r];
                }
            }
        }
        __syncthreads();
    }
}

__global__ void sptrsvBandedAwareKernel(
    int n,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution)
{
    if (blockIdx.x != 0) {
        return;
    }

    extern __shared__ double shared[];
    double* partialSums = shared;
    double* partialDiags = shared + blockDim.x;

    for (int row = 0; row < n; ++row) {
        const int rowStart = rowPtr[row];
        const int rowEnd = rowPtr[row + 1];
        double partial = 0.0;
        double diagonal = 0.0;

        for (int entry = rowStart + threadIdx.x; entry < rowEnd; entry += blockDim.x) {
            const int col = colIdx[entry];
            const double value = values[entry];
            if (col < row) {
                partial += value * solution[col];
            } else if (col == row) {
                diagonal = value;
            }
        }

        partialSums[threadIdx.x] = partial;
        partialDiags[threadIdx.x] = diagonal;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
                if (partialDiags[threadIdx.x] == 0.0) {
                    partialDiags[threadIdx.x] = partialDiags[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            solution[row] = (rhs[row] - partialSums[0]) / partialDiags[0];
        }
        __syncthreads();
    }
}
