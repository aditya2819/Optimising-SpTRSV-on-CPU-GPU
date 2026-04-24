#pragma once

#include <string>
#include <vector>

struct CsrMatrix {
    int n = 0;
    int nnz = 0;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<double> values;
};

CsrMatrix generateLowerTriangular(int n, double density, unsigned seed = 42);
std::vector<double> generateRandomRhs(int n, unsigned seed = 99);

void sptrsvCpuReference(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

double computeResidual(
    const CsrMatrix& matrix,
    const std::vector<double>& solution,
    const std::vector<double>& rhs);

bool sptrsvGpuNaive(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuCoalesced(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuThreadPerRow(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuWarpPerRow(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuAdaptiveRow(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuBlockPerRow(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuRowBucketed(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuLevelAware(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuRandomLevelAware(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuPowerLawAware(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuBlockAware(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuBandedAware(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuRegularAware(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuFamilyAware(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvGpuLearnedAware(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

bool sptrsvCuSparse(
    const CsrMatrix& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution);

void printDeviceInfo();
