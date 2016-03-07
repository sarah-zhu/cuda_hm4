//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <iostream>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
__global__ void radix_sort(unsigned int* const d_inputVals, unsigned int* d_inputPos, 
                            unsigned int* const d_outputVals, unsigned int* d_outputPos, 
                          unsigned int* const d_scanInput,
                          int bit, const size_t numElems)
{
    unsigned int globId = blockDim.x*blockIdx.x + threadIdx.x;
    if(globId>=numElems)
        return;
    unsigned int val = d_inputVals[globId];
    unsigned int digit = (val>>bit) & 1;
    unsigned int old_pos = d_inputPos[globId];


    unsigned  int T_before = d_scanInput[globId];
    unsigned  int T_total = d_scanInput[numElems-1];
    unsigned  int F_total = numElems - T_total;
    
    
    unsigned  int new_pos;
    if(digit)
        new_pos = F_total + T_before-1;
    else
        new_pos = globId - T_before;
    
    d_outputPos[new_pos] = old_pos;
    d_outputVals[new_pos] = val;
    
}


__global__ void scan( unsigned int* const d_inputVals,
                      unsigned int* const d_scanInput,
                      int bit,
                      int blockId,
                      int blockSize,
                      int numElems)
{
    unsigned int threadId = threadIdx.x;
    unsigned int globId = threadIdx.x + blockId*blockSize;
    if(globId>=numElems)
        return;
 
    int n = blockDim.x;
    
    d_scanInput[globId] = (d_inputVals[globId]>>bit) & 1;
    __syncthreads();

    for(int i=1; i<n; i*=2)
    {
        int temp;
        if(threadId>=i)
            temp = d_scanInput[globId - i];
        __syncthreads();
        if(threadId>=i)
            d_scanInput[globId] += temp;
        __syncthreads();
    }
    if(blockId>0)
        d_scanInput[globId] += d_scanInput[blockId*blockSize-1];

}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    dim3 blockSize(512,1,1);
    dim3 gridSize(ceil(float(numElems)/blockSize.x),1,1);

    unsigned int* d_scanInput;
    checkCudaErrors(cudaMalloc(&d_scanInput, sizeof(unsigned int)*numElems));

    
    for(int bit=0; bit<32; bit++)
    {
        checkCudaErrors(cudaMemset(d_scanInput, 0, sizeof(unsigned int)*numElems));

        for(int blockId=0; blockId<gridSize.x; blockId++)
        {
            scan<<<1, blockSize>>>(d_inputVals, d_scanInput, bit, blockId, blockSize.x, numElems);
            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());
        }
        
        radix_sort<<<gridSize, blockSize>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, d_scanInput, bit, numElems);
        cudaDeviceSynchronize(); 
        checkCudaErrors(cudaGetLastError());
        
        checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
        cudaDeviceSynchronize();
        
    }
    
    checkCudaErrors(cudaFree(d_scanInput));
    
}