#ifndef GROUPBY_HASH_CUH
#define GROUPBY_HASH_CUH

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
} 

void groupby_hash_GPU(const int* key_columns_h, int num_key_columns, int num_key_rows,
		      const int* value_columns_h, int num_value_columns, int num_value_rows,
		      reductionType* ops, int num_ops, int* output_keys, int* output_values, int &num_output_rows);


#endif