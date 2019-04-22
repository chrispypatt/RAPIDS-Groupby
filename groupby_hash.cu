#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

#include "cpuGroupby.h"
#include "groupby_hash.cuh"

// is there dynamic size constant memory?
__constant__ reductionType ops_c[512];


#include "groupby_hash_templates.cu"


void groupby_hash_GPU(const int* key_columns_h, int num_key_columns, int num_key_rows,
		      const int* value_columns_h, int num_value_columns, int num_value_rows,
		      reductionType* ops, int num_ops, int* output_keys, int* output_values, int &num_output_rows)
{
  constexpr unsigned int BLOCKDIM = 1024;
  constexpr unsigned int HASH_TABLE_SIZE = 1003;
  constexpr unsigned int GRIDDIM = 40; // 40 as GTX 1080 only have 20 SMs and it can schedule 2048 threads
                                       // change to 56*2 = 112 if testing on Tesla P100
  
  // variableAllocating
  int* key_columns_d = NULL;
  int* value_columns_d = NULL;
  int* hash_key_idx_d = NULL;
  int* hash_count_d = NULL;
  int* hash_results_d = NULL;

  gpuErrchk(cudaMalloc(&key_columns_d, sizeof(int)*num_key_columns*num_key_rows));
  gpuErrchk(cudaMalloc(&value_columns_d, sizeof(int)*num_value_columns*num_value_rows));
  gpuErrchk(cudaMalloc(&hash_key_idx_d, sizeof(int)*HASH_TABLE_SIZE));
  gpuErrchk(cudaMalloc(&hash_count_d, sizeof(int)*HASH_TABLE_SIZE));
  gpuErrchk(cudaMalloc(&hash_results_d, sizeof(int)*HASH_TABLE_SIZE*num_ops));
  
  // initialize values
  gpuErrchk(cudaMemcpy(key_columns_d, key_columns_h, sizeof(int)*num_key_columns*num_key_rows, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(value_columns_d, value_columns_h, sizeof(int)*num_value_columns*num_value_rows, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(ops_c, ops, sizeof(reductionType) * num_ops));
  initializeVariable<int><<<GRIDDIM, BLOCKDIM>>>(hash_key_idx_d, hash_count_d, hash_results_d, HASH_TABLE_SIZE, num_ops);
  gpuErrchk(cudaDeviceSynchronize());

  // fill hash table
  fillTable<int, int><<<GRIDDIM, BLOCKDIM>>>(key_columns_d, num_key_rows, num_key_columns,
					     value_columns_d, num_value_rows, num_value_columns,
					     hash_key_idx_d, hash_count_d, hash_results_d,
					     HASH_TABLE_SIZE, num_ops);
  gpuErrchk(cudaDeviceSynchronize());

  //shrink the hash table to output array
  //Create array of idices for hash table
  int *seq, *hashTable_idxs;
  cudaMalloc((void**)&seq, HASH_TABLE_SIZE*sizeof(int)); //for hash index sequence  
  cudaMalloc((void**)&hashTable_idxs, HASH_TABLE_SIZE*sizeof(int));  //for key indexs without -1   
  thrust::device_ptr<int> hash_d_seq = thrust::device_pointer_cast(seq); //for hash index sequence 
  thrust::device_ptr<int> hashTable_idxs_d = thrust::device_pointer_cast(hashTable_idxs); //for key indexs without -1 
  thrust::sequence(thrust::device, hash_d_seq, hash_d_seq + HASH_TABLE_SIZE); //fill hash index seq


  //copy hash idex of keys, removeing -1's which signify not used
//   copy_if(policy, index seq start, index seq end, hash keys for comparison, result containing idx to non -1's, comparator)
  auto newEnd = thrust::copy_if(thrust::device, hash_d_seq, hash_d_seq + HASH_TABLE_SIZE, hash_key_idx_d, hashTable_idxs_d, is_not_neg_1());
  
  num_output_rows = newEnd - hashTable_idxs_d;

  int* output_key_columns_d = NULL;
  cudaMalloc(&output_key_columns_d, sizeof(int)*num_key_columns*num_output_rows);
  copyUnique<int><<<GRIDDIM,BLOCKDIM>>>(hashTable_idxs, hash_key_idx_d,key_columns_d, output_key_columns_d, num_output_rows, num_key_columns, num_key_rows);

  int* output_value_columns_d = NULL;
  cudaMalloc(&output_value_columns_d, sizeof(int)*num_value_columns*num_output_rows);
  copyValues<int><<<GRIDDIM,BLOCKDIM>>>(hashTable_idxs, hash_results_d,hash_count_d, value_columns_d, output_value_columns_d, num_output_rows, num_value_columns, num_value_rows, num_ops, HASH_TABLE_SIZE);

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
