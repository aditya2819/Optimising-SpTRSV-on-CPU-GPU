#pragma once

__global__ void sptrsvNaiveKernel(
    int n,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution);

__global__ void sptrsvCoalescedKernel(
    int n,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution);

__global__ void sptrsvThreadPerRowKernel(
    int row,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution);

__global__ void sptrsvWarpPerRowKernel(
    int row,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution);

__global__ void sptrsvBlockPerRowKernel(
    int row,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution);

__global__ void sptrsvThreadBucketedKernel(
    int rowBegin,
    int rowEnd,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution);

__global__ void sptrsvWarpBucketedKernel(
    int rowBegin,
    int rowEnd,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution);

__global__ void sptrsvBlockBucketedKernel(
    int rowBegin,
    int rowEnd,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution);

__global__ void sptrsvThreadRowListKernel(
    int count,
    const int* rows,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution);

__global__ void sptrsvWarpRowListKernel(
    int count,
    const int* rows,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution);

__global__ void sptrsvBlockRowListKernel(
    int count,
    const int* rows,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution);

__global__ void sptrsvBcsrKernel(
    int n,
    int numBlockRows,
    int blockSize,
    const int* blockRowPtr,
    const int* blockColIdx,
    const double* blockValues,
    const double* rhs,
    double* solution);

__global__ void sptrsvBcsrRowListKernel(
    int n,
    int count,
    int blockSize,
    const int* rows,
    const int* blockRowPtr,
    const int* blockColIdx,
    const double* blockValues,
    const double* rhs,
    double* solution);

__global__ void sptrsvBcsrChunkKernel(
    int n,
    int blockSize,
    int chunkBegin,
    int chunkEnd,
    const int* blockRowPtr,
    const int* blockColIdx,
    const double* blockValues,
    const double* rhs,
    double* solution);

__global__ void sptrsvBandedAwareKernel(
    int n,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* rhs,
    double* solution);
