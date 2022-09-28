/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample evaluates fair call price for a
 * given set of European options under binomial model.
 * See supplied whitepaper for more explanations.
 */



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "binomialOptions_common.h"
#include "realtype.h"

#define DOUBLE 0
#define FLOAT 1
#define INT 2
#define LONG 3

void writeQualityFile(const char *fileName, void *ptr, int type, size_t numElements){
    FILE *fd = fopen(fileName, "wb");
    assert(fd && "Could Not Open File\n");
    fwrite(&numElements, sizeof(size_t), 1, fd);
    fwrite(&type, sizeof(int), 1, fd);
    if ( type == DOUBLE)
        fwrite(ptr, sizeof(double), numElements, fd);
    else if ( type == FLOAT)
        fwrite(ptr, sizeof(float), numElements, fd);
    else if ( type == INT)
        fwrite(ptr, sizeof(int), numElements, fd);
    else
        assert(0 && "Not supported data type to write\n");
    fclose(fd);
}
void readData(FILE *fd, double **data,  size_t* numElements){
    assert(fd && "File pointer is not valid\n");
    fread(numElements, sizeof(size_t),1,fd);
    size_t elements = *numElements;
    double *ptr = (double*) malloc (sizeof(double)*elements);
    assert(ptr && "Could Not allocate pointer\n");
    *data = ptr;
    size_t i;
    int type;
    fread(&type, sizeof(int), 1, fd); 
    if ( type == DOUBLE){
        fread(ptr, sizeof(double), elements, fd);
    }
    else if ( type == FLOAT){
        float *tmp = (float*) malloc (sizeof(float)*elements);
        fread(tmp, sizeof(float), elements,fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (double) tmp[i];
        }
        free (tmp);
    }
    else if( type == INT ){
        int *tmp = (int*) malloc (sizeof(int)*elements);
        fread(tmp, sizeof(int), elements, fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (double) tmp[i];
        }
        free(tmp);
    }
    return; 
}

void readData(FILE *fd, float **data,  size_t* numElements){
    assert(fd && "File pointer is not valid\n");
    fread(numElements, sizeof(size_t),1,fd);
    size_t elements = *numElements;

    float *ptr = (float*) malloc (sizeof(float)*elements);
    assert(ptr && "Could Not allocate pointer\n");
    *data = ptr;

    size_t i;
    int type;
    fread(&type, sizeof(int), 1, fd); 
    if ( type == FLOAT ){
        fread(ptr, sizeof(float), elements, fd);
    }
    else if ( type == DOUBLE){
        double *tmp = (double*) malloc (sizeof(double)*elements);
        fread(tmp, sizeof(double), elements,fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (float) tmp[i];
        }
        free (tmp);
    }
    else if ( type == INT ){
        int *tmp = (int*) malloc (sizeof(int) * elements);
        fread(tmp, sizeof(int), elements, fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (float) tmp[i];
        }
        free(tmp);
    }
    return; 
}

void readData(FILE *fd, int **data,   size_t* numElements){
    assert(fd && "File pointer is not valid\n");
    fread(numElements, sizeof(size_t),1,fd);
    size_t elements = *numElements;

    int *ptr = (int*) malloc (sizeof(int)*elements);
    assert(ptr && "Could Not allocate pointer\n");
    *data = ptr;

    size_t i;
    int type;
    fread(&type, sizeof(int), 1, fd); 
    if ( type == INT ){
        fread(ptr, sizeof(int), elements, fd);
    }
    else if ( type == DOUBLE){
        double *tmp = (double*) malloc (sizeof(double)*elements);
        fread(tmp, sizeof(double), elements,fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (int) tmp[i];
        }
        free (tmp);
    }
    else if( type == FLOAT ){
        float *tmp = (float*) malloc (sizeof(float)*elements);
        fread(tmp, sizeof(float), elements, fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (int) tmp[i];
        }
        free(tmp);
    }
    return; 
}


////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for binomial tree results validation
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCall(
    real &callResult,
    TOptionData optionData
);

////////////////////////////////////////////////////////////////////////////////
// Process single option on CPU
// Note that CPU code is for correctness testing only and not for benchmarking.
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsCPU(
    real &callResult,
    TOptionData optionData
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsGPU(
    real *callValue,
    TOptionData  *optionData,
    int optN
);

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
real randData(real low, real high)
{
    real t = (real)rand() / (real)RAND_MAX;
    return ((real)1.0 - t) * low + t * high;
}



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("[%s] - Starting...\n", argv[0]);

    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    if (((deviceProp.major << 4) + deviceProp.minor) < 0x20)
    {
        fprintf(stderr, "binomialOptions requires Compute Capability of SM 2.0 or higher to run.\n");
        cudaDeviceReset();
        exit(EXIT_WAIVED);
    }

  FILE *file;

  if(!(argc == 3 || argc == 4))
    {
      std::cout << "USAGE: " << argv[0] << " input_file num_steps [output_file]";
      return EXIT_FAILURE;
    }

  char *inputFile = argv[1];

    //Read input data from file
    file = fopen(inputFile, "rb");
    if(file == NULL) {
        printf("ERROR: Unable to open file `%s'.\n", inputFile);
        exit(1);
    }

  bool write_output = false;
  std::string ofname;
  if(argc == 4)
    {
      write_output = true;
      ofname = argv[3];
    }


  // sptprice
  real *S;
  // strike
  real *X;
  // time
  real *T;
  // rate
  real *R;
  // volatility
  real *V;
  int *otype;

  real
    sumDelta, sumRef, gpuTime, errorVal;

  printf("Reading input data...\n");
  size_t numOptions = 0;

#define PAD 256
#define LINESIZE 64
    readData(file,&otype, &numOptions);  
    readData(file,&S, &numOptions);  
    readData(file,&X, &numOptions);  
    readData(file,&R, &numOptions);  
    readData(file,&V, &numOptions);  
    readData(file,&T, &numOptions);  

  const int NUM_OPTIONS = numOptions;

    const int OPT_N = MAX_OPTIONS;

    TOptionData *optionData = new TOptionData[MAX_OPTIONS];
    real *callValueBS = new real[MAX_OPTIONS];
    real *callValueGPU = new real[MAX_OPTIONS];
    real *callValueCPU = new real[MAX_OPTIONS];

    StopWatchInterface *hTimer = NULL;
    int i;

    sdkCreateTimer(&hTimer);

    printf("Generating input data...\n");
    //Generate options set
    srand(123);

    for (i = 0; i < OPT_N; i++)
    {
        optionData[i].S = S[i];
        optionData[i].X = X[i];
        optionData[i].T = T[i];
        optionData[i].R = R[i];
        optionData[i].V = V[i];
        BlackScholesCall(callValueBS[i], optionData[i]);
    }

    printf("Running GPU binomial tree...\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    binomialOptionsGPU(callValueGPU, optionData, OPT_N);

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    printf("Options count            : %i     \n", OPT_N);
    printf("Time steps               : %i     \n", NUM_STEPS);
    printf("binomialOptionsGPU() time: %f msec\n", gpuTime);
    printf("Options per second       : %f     \n", OPT_N / (gpuTime * 0.001));

    printf("Running CPU binomial tree...\n");

    for (i = 0; i < OPT_N; i++)
    {
        binomialOptionsCPU(callValueCPU[i], optionData[i]);
    }

    if(write_output)
      {
        writeQualityFile(ofname.c_str(), callValueGPU, DOUBLE, NUM_OPTIONS);
      }


    printf("Comparing the results...\n");
    sumDelta = 0;
    sumRef   = 0;
    printf("GPU binomial vs. Black-Scholes\n");

    for (i = 0; i < OPT_N; i++)
    {
        sumDelta += fabs(callValueBS[i] - callValueGPU[i]);
        sumRef += fabs(callValueBS[i]);
    }

    if (sumRef >1E-5)
    {
        printf("L1 norm: %E\n", (double)(sumDelta / sumRef));
    }
    else
    {
        printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));
    }

    printf("CPU binomial vs. Black-Scholes\n");
    sumDelta = 0;
    sumRef   = 0;

    for (i = 0; i < OPT_N; i++)
    {
        sumDelta += fabs(callValueBS[i]- callValueCPU[i]);
        sumRef += fabs(callValueBS[i]);
    }

    if (sumRef >1E-5)
    {
        printf("L1 norm: %E\n", sumDelta / sumRef);
    }
    else
    {
        printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));
    }

    printf("CPU binomial vs. GPU binomial\n");
    sumDelta = 0;
    sumRef   = 0;

    for (i = 0; i < OPT_N; i++)
    {
        sumDelta += fabs(callValueGPU[i] - callValueCPU[i]);
        sumRef += callValueCPU[i];
    }

    if (sumRef > 1E-5)
    {
        printf("L1 norm: %E\n", errorVal = sumDelta / sumRef);
    }
    else
    {
        printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));
    }

    printf("Shutting down...\n");

    sdkDeleteTimer(&hTimer);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");

    if (errorVal > 5e-4)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
