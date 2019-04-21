#include <cuda.h>
#include <cuda_runtime.h>

#include "cpuGroupby.h"
#include "groupby_hash_templates.cu"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
} 

// is there dynamic size constant memory?
__constant__ reductionType ops_d[512];

void groupby_hash_GPU(const int* key_columns_h, int num_key_columns, int num_key_rows,
		      const int* value_columns_h, int num_value_columns, int num_value_rows,
		      reductionType* ops, int num_ops, int* output_keys, int* output_values, int &num_output_rows)
{
  constexpr unsigned int BLOCKDIM = 1024;
  constexpr unsigned int HASH_TABLE_SIZE = 1003;
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
  gpuErrchk(cudaMemcpyToSymbol(ops_d, ops, sizeof(reductionType) * num_ops));
  initializeVariable<int><<<50, BLOCKDIM>>>(hash_key_idx_d, hash_count_d, hash_results_d, HASH_TABLE_SIZE, ops_d, num_ops);
  gpuErrchk(cudaDeviceSynchronize());

  // fill hash table
  fillTable<int, int><<<50, BLOCKDIM>>>(key_columns_d, num_key_rows, num_key_columns,
					value_columns_d, num_value_rows, num_value_columns,
					hash_key_idx_d, hash_count_d, hash_results_d,
					HASH_TABLE_SIZE, ops_d, num_ops);
  gpuErrchk(cudaDeviceSynchronize());

  // shrink the hash table to output array

  // copy back

  // free elements

  cudaFree(key_columns_d);
  cudaFree(value_columns_d);
  cudaFree(hash_key_idx_d);
  cudaFree(hash_count_d);
  cudaFree(hash_results_d);
  
}
