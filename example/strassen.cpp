#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <algorithm>

using namespace std;


void addMatrices(float *A, float *B, float *C, int size)
{
    for (int i = 0; i < size; ++i)
        C[i] = A[i] + B[i];
}

void subtractMatrices(float *A, float *B, float *C, int size)
{
    for (int i = 0; i < size; ++i)
        C[i] = A[i] - B[i];
}

void simpleMultiply(float *A, float *B, float *C, int m, int n, int p)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            C[i * p + j] = 0;
            for (int k = 0; k < n; ++k)
            {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

void StrassenMatrixMultiply(float *A, float *B, float *C, int m, int n, int p)
{
    if (m <= 1 || n <= 1 || p <= 1)
    {
        simpleMultiply(A, B, C, m, n, p);
        return;
    }

    int newSize = max({m, n, p});
    newSize = (newSize % 2 == 0) ? newSize : newSize + 1;

    int halfSize = newSize / 2;

    vector<float> A11(halfSize * halfSize, 0), A12(halfSize * halfSize, 0), A21(halfSize * halfSize, 0), A22(halfSize * halfSize, 0);
    vector<float> B11(halfSize * halfSize, 0), B12(halfSize * halfSize, 0), B21(halfSize * halfSize, 0), B22(halfSize * halfSize, 0);
    vector<float> C11(halfSize * halfSize, 0), C12(halfSize * halfSize, 0), C21(halfSize * halfSize, 0), C22(halfSize * halfSize, 0);
    vector<float> M1(halfSize * halfSize, 0), M2(halfSize * halfSize, 0), M3(halfSize * halfSize, 0), M4(halfSize * halfSize, 0), M5(halfSize * halfSize, 0), M6(halfSize * halfSize, 0), M7(halfSize * halfSize, 0);
    vector<float> tempA(halfSize * halfSize, 0), tempB(halfSize * halfSize, 0);

    for (int i = 0; i < halfSize; ++i)
    {
        for (int j = 0; j < halfSize; ++j)
        {
            if (i < m && j < n) A11[i * halfSize + j] = A[i * n + j];
            if (i < m && j + halfSize < n) A12[i * halfSize + j] = A[i * n + j + halfSize];
            if (i + halfSize < m && j < n) A21[i * halfSize + j] = A[(i + halfSize) * n + j];
            if (i + halfSize < m && j + halfSize < n) A22[i * halfSize + j] = A[(i + halfSize) * n + j + halfSize];

            if (i < n && j < p) B11[i * halfSize + j] = B[i * p + j];
            if (i < n && j + halfSize < p) B12[i * halfSize + j] = B[i * p + j + halfSize];
            if (i + halfSize < n && j < p) B21[i * halfSize + j] = B[(i + halfSize) * p + j];
            if (i + halfSize < n && j + halfSize < p) B22[i * halfSize + j] = B[(i + halfSize) * p + j + halfSize];
        }
    }

    addMatrices(&A11[0], &A22[0], &tempA[0], halfSize * halfSize);
    addMatrices(&B11[0], &B22[0], &tempB[0], halfSize * halfSize);
    StrassenMatrixMultiply(&tempA[0], &tempB[0], &M1[0], halfSize, halfSize, halfSize);

    addMatrices(&A21[0], &A22[0], &tempA[0], halfSize * halfSize);
    StrassenMatrixMultiply(&tempA[0], &B11[0], &M2[0], halfSize, halfSize, halfSize);

    subtractMatrices(&B12[0], &B22[0], &tempB[0], halfSize * halfSize);
    StrassenMatrixMultiply(&A11[0], &tempB[0], &M3[0], halfSize, halfSize, halfSize);

    subtractMatrices(&B21[0], &B11[0], &tempB[0], halfSize * halfSize);
    StrassenMatrixMultiply(&A22[0], &tempB[0], &M4[0], halfSize, halfSize, halfSize);

    addMatrices(&A11[0], &A12[0], &tempA[0], halfSize * halfSize);
    StrassenMatrixMultiply(&tempA[0], &B22[0], &M5[0], halfSize, halfSize, halfSize);

    subtractMatrices(&A21[0], &A11[0], &tempA[0], halfSize * halfSize);
    addMatrices(&B11[0], &B12[0], &tempB[0], halfSize * halfSize);
    StrassenMatrixMultiply(&tempA[0], &tempB[0], &M6[0], halfSize, halfSize, halfSize);

    subtractMatrices(&A12[0], &A22[0], &tempA[0], halfSize * halfSize);
    addMatrices(&B21[0], &B22[0], &tempB[0], halfSize * halfSize);
    StrassenMatrixMultiply(&tempA[0], &tempB[0], &M7[0], halfSize, halfSize, halfSize);

    addMatrices(&M1[0], &M4[0], &tempA[0], halfSize * halfSize);
    subtractMatrices(&tempA[0], &M5[0], &tempB[0], halfSize * halfSize);
    addMatrices(&tempB[0], &M7[0], &C11[0], halfSize * halfSize);

    addMatrices(&M3[0], &M5[0], &C12[0], halfSize * halfSize);
    addMatrices(&M2[0], &M4[0], &C21[0], halfSize * halfSize);
    addMatrices(&M1[0], &M3[0], &tempA[0], halfSize * halfSize);
    subtractMatrices(&tempA[0], &M2[0], &tempB[0], halfSize * halfSize);
    addMatrices(&tempB[0], &M6[0], &C22[0], halfSize * halfSize);

    for (int i = 0; i < halfSize; ++i)
    {
        for (int j = 0; j < halfSize; ++j)
        {
            if (i < m && j < p) C[i * p + j] = C11[i * halfSize + j];
            if (i < m && j + halfSize < p) C[i * p + j + halfSize] = C12[i * halfSize + j];
            if (i + halfSize < m && j < p) C[(i + halfSize) * p + j] = C21[i * halfSize + j];
            if (i + halfSize < m && j + halfSize < p) C[(i + halfSize) * p + j + halfSize] = C22[i * halfSize + j];
        }
    }
}
