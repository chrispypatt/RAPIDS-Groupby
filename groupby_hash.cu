#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <random>
#include <iostream>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <cassert>

#include "cpuGroupby.h"
#include "groupby_hash.cuh"

// is there dynamic size constant memory?
__constant__ reductionType ops_c[512];


#include "groupby_hash_templates.cu"

size_t size_alignment(size_t size, size_t alignment)
{
  return (size + alignment - 1) / alignment;
}

void groupby_hash_GPU(const int hash_size, const int* key_columns_h, int num_key_columns, int num_key_rows,
		      const int* value_columns_h, int num_value_columns, int num_value_rows,
		      reductionType* ops, int num_ops, int* output_keys, int* output_values, int &num_output_rows)
{
#ifdef DEBUG
  constexpr unsigned int BLOCKDIM = 512;
#else
  constexpr unsigned int BLOCKDIM = 1024;
#endif
  unsigned int HASH_TABLE_SIZE = hash_size;
#ifndef TESLA
  constexpr unsigned int GRIDDIM = 40; 
#else
  constexpr unsigned int GRIDDIM = 112; 
#endif
  
  using Tval = int; // replace int with actual variable type if needed;
  
  //set restarting flags;
  int hashsize_mutiplier = 1;
  int* overflow_flag = NULL;
  cudaMallocManaged(&overflow_flag,sizeof(int));
  overflow_flag[0] = 0; // No overflow happens

  // variableAllocating
  int* key_columns_d = NULL;
  int* value_columns_d = NULL;
  int* hash_key_idx_d = NULL;
  int* hash_count_d = NULL;
  int* hash_results_d = NULL;

  gpuErrchk(cudaMalloc(&key_columns_d, sizeof(int)*num_key_columns*num_key_rows));
  gpuErrchk(cudaMalloc(&value_columns_d, sizeof(int)*num_value_columns*num_value_rows));
  
  // copy to target
  gpuErrchk(cudaMemcpy(key_columns_d, key_columns_h, sizeof(int)*num_key_columns*num_key_rows, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(value_columns_d, value_columns_h, sizeof(int)*num_value_columns*num_value_rows, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(ops_c, ops, sizeof(reductionType) * num_ops));

  // sample hash table length
#ifdef CPU_SAMPLE
  unsigned int predictedLength = predictTableLength_CPU<int>(key_columns_h,
							     num_key_rows,
							     num_key_columns);
  std::cout << "Predicted Hash Table Length:" << predictedLength << std::endl;
  HASH_TABLE_SIZE = predictedLength;
#elif defined(GPU_SAMPLE)
  unsigned int* count = NULL;
  curandState* state = NULL;
  gpuErrchk(cudaMallocManaged(&count, sizeof(unsigned int)*3));
  gpuErrchk(cudaMalloc(&state, 1*BLOCKDIM*sizeof(curandState)));
  unsigned int iterations = num_key_rows / BLOCKDIM / 100 + 1;
  fillCURANDState<<<1, BLOCKDIM>>>(state, gen());
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  predictTableLength_GPU<int><<<1, BLOCKDIM>>>(key_columns_d,
					       num_key_rows,
					       num_key_columns,
					       iterations,
					       count,
					       state);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  unsigned int countTotal = count[0] + count[1] + count[2];
  unsigned int predictedLength;
  if (count[1]) {
    float delta = std::sqrt((float)countTotal*((float)countTotal*9 - (float)count[1]*12));
    predictedLength = 2.6 * ((3*countTotal + delta) / (2*count[1]));
  } else {
    std::cout << "Cannot predict number of keys (too big?)" << std::endl
	      << "Using about 1% of original rows as start" << std::endl;;
    predictedLength = num_key_rows / 81;
    if (predictedLength % 2 == 0) ++ predictedLength;
  }
  std::cout << "Predicted Hash Table Length:" << predictedLength << std::endl;
  HASH_TABLE_SIZE = predictedLength;
#endif
  

#ifndef PRIVATIZATION
    
  do {
    overflow_flag[0] = 0;

    
  
    gpuErrchk(cudaMalloc(&hash_key_idx_d, sizeof(int)*HASH_TABLE_SIZE*hashsize_mutiplier));
    gpuErrchk(cudaMalloc(&hash_count_d, sizeof(int)*HASH_TABLE_SIZE*hashsize_mutiplier));
    gpuErrchk(cudaMalloc(&hash_results_d, sizeof(int)*HASH_TABLE_SIZE*num_ops*hashsize_mutiplier)); 
  
    initializeVariable<int><<<GRIDDIM, BLOCKDIM>>>(hash_key_idx_d, hash_count_d, hash_results_d, HASH_TABLE_SIZE*hashsize_mutiplier, num_ops);
    gpuErrchk(cudaDeviceSynchronize());


    fillTable<int, int><<<GRIDDIM, BLOCKDIM>>>(key_columns_d, num_key_rows, num_key_columns,
               value_columns_d, num_value_rows, num_value_columns,
               hash_key_idx_d, hash_count_d, hash_results_d,
               HASH_TABLE_SIZE*hashsize_mutiplier, num_ops, overflow_flag);
    gpuErrchk(cudaDeviceSynchronize());
    printf("The overflow_flag is: %d\n", overflow_flag[0]);
    printf("Current hash size is: %d\n", hashsize_mutiplier*HASH_TABLE_SIZE);
    if (overflow_flag[0] == 1) {
      hashsize_mutiplier *= 3;
      cudaFree(hash_key_idx_d);
      cudaFree(hash_count_d);
      cudaFree(hash_results_d);
      hash_key_idx_d = NULL;
      hash_count_d = NULL;
      hash_results_d = NULL;
    
    }

  } while(overflow_flag[0] == 1);
  
    //printf("The overflow_flag is: %d\n", overflow_flag[0]);

#else
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  size_t sharedMemPerBlock = deviceProp.sharedMemPerBlock;
  printf("Total amount of sharedmemory per block %u\n", sharedMemPerBlock);
# ifdef TESLA
  sharedMemPerBlock = 32 * 1024;
# endif
  size_t max_capacity = sharedMemPerBlock - 48; // for some reason 48 is required for reserved variable
  size_t s_len_table = max_capacity / (2*sizeof(int) + sizeof(Tval)*num_ops);
  size_t sharedMemorySize = 0;
  while (true) { // calculate the suitable length of shared memory table
    sharedMemorySize = size_alignment(2*sizeof(int)*s_len_table, sizeof(Tval)) * sizeof(int);
    sharedMemorySize += sizeof(Tval)*num_ops*s_len_table;
    if (sharedMemorySize < max_capacity)
      if (s_len_table % 2 == 1) break; // always make length an odd number to avoid serious collision
    --s_len_table;
  }
  printf("Length of Shared Table: %u\n", s_len_table);
  printf("Total extern shared memory: %u\n", sharedMemorySize);
  fillTable_privatization
    <int, int><<<GRIDDIM, BLOCKDIM, sharedMemorySize>>>(key_columns_d, num_key_rows,
							num_key_columns, value_columns_d,
							num_value_rows, num_value_columns,
							hash_key_idx_d, hash_count_d,
							hash_results_d, HASH_TABLE_SIZE*hashsize_mutiplier,
							s_len_table, num_ops);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
#endif
  //shrink the hash table to output array
  //Create array of idices for hash table
  int *seq, *hashTable_idxs;
  int hash_table_size_fixed = HASH_TABLE_SIZE*hashsize_mutiplier;
  cudaMalloc((void**)&seq, HASH_TABLE_SIZE*hashsize_mutiplier*sizeof(int)); //for hash index sequence  
  cudaMalloc((void**)&hashTable_idxs, HASH_TABLE_SIZE*hashsize_mutiplier*sizeof(int));  //for key indexs without -1   
  thrust::device_ptr<int> hash_d_seq = thrust::device_pointer_cast(seq); //for hash index sequence 
  thrust::device_ptr<int> hashTable_idxs_d = thrust::device_pointer_cast(hashTable_idxs); //for key indexs without -1 
  thrust::sequence(thrust::device, hash_d_seq, hash_d_seq + hash_table_size_fixed); //fill hash index seq


  //copy hash idex of keys, removeing -1's which signify not used
//   copy_if(policy, index seq start, index seq end, hash keys for comparison, result containing idx to non -1's, comparator)
  auto newEnd = thrust::copy_if(thrust::device, hash_d_seq, hash_d_seq + hash_table_size_fixed, hash_key_idx_d, hashTable_idxs_d, is_pos());
  
  num_output_rows = newEnd - hashTable_idxs_d;
  printf("%d output rows!\n", num_output_rows);
  printf("%d hash length!\n", HASH_TABLE_SIZE*hashsize_mutiplier);

  int* output_key_columns_d = NULL;
  cudaMalloc(&output_key_columns_d, sizeof(int)*num_key_columns*num_output_rows);
  copyUnique<int><<<GRIDDIM,BLOCKDIM>>>(hashTable_idxs, hash_key_idx_d,key_columns_d, output_key_columns_d, num_output_rows, num_key_columns, num_key_rows);

  //gpuErrchk(cudaDeviceSynchronize());

  int* output_value_columns_d = NULL;
  gpuErrchk(cudaMalloc(&output_value_columns_d, sizeof(int)*num_value_columns*num_output_rows));
  copyValues<int><<<GRIDDIM,BLOCKDIM>>>(hashTable_idxs, hash_results_d,hash_count_d, value_columns_d, output_value_columns_d, num_output_rows, num_value_columns, num_value_rows, num_ops, hash_table_size_fixed);


  gpuErrchk(cudaDeviceSynchronize());

  // copy back

  gpuErrchk(cudaMemcpy(output_keys,output_key_columns_d,sizeof(int)*num_key_columns*num_output_rows,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(output_values,output_value_columns_d,sizeof(int)*num_value_columns*num_output_rows,cudaMemcpyDeviceToHost)); 


  // free elements

  cudaFree(key_columns_d);
  cudaFree(value_columns_d);
  cudaFree(hash_key_idx_d);
  cudaFree(hash_count_d);
  cudaFree(hash_results_d);
  cudaFree(output_key_columns_d);
  cudaFree(output_value_columns_d);
  cudaFree(seq);
  cudaFree(hashTable_idxs);
  
}
