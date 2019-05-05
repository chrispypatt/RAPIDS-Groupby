#include <cstddef>

#include "limits.cuh"

// assume column major here
template <typename T> __host__ __device__
bool keyEqualCM(const T* key_columns, size_t idx1, size_t idx2, size_t num_key_rows, size_t num_key_columns)
{
  for (size_t i=0; i < num_key_columns; ++i) {
    if (key_columns[i*num_key_rows+idx1] != key_columns[i*num_key_rows+idx2])
      return false;
  }
  return true;
}

// assume row major here
template <typename T> __host__ __device__
bool keyEqualRM(const T* key_columns, size_t idx1, size_t idx2, size_t num_key_rows, size_t num_key_columns)
{
  for (size_t i=0; i < num_key_columns; ++i) {
    if (key_columns[i+num_key_rows*idx1] != key_columns[i+num_key_rows*idx2])
      return false;
  }
  return true;
}

// hashKey generating
template <typename T> __host__ __device__
size_t HashKey(size_t idx, T* key_columns, size_t num_key_rows, size_t num_key_columns) {
  size_t hash_key = 0;
  for (size_t i=0; i < num_key_columns; ++i) {
    hash_key = (31 * hash_key) + key_columns[i*num_key_rows+idx];
  }
  return hash_key;
}

template <typename Tval> __device__
void updateEntry(Tval* value_columns,
		 size_t num_val_rows,
		 size_t num_ops,
		 size_t idx,
		 size_t hashPos,
		 Tval* hash_results,
		 int* countPtr,
		 size_t len_hash_table,
		 int count=1)
{
  // update count
  atomicAdd(countPtr, count);
  // update each item
  for (size_t i = 0; i < num_ops; ++i) {
    Tval value = value_columns[i * num_val_rows + idx];
    size_t val_idx = i * len_hash_table + hashPos;
    switch(ops_c[i]) {
    case rmin:
      atomicMin(&(hash_results[val_idx]), value);
      break;
    case rmax:
      atomicMax(&(hash_results[val_idx]), value);
      break;
    case rcount:
      atomicAdd(&(hash_results[val_idx]), count);
      break;
    case rmean: // fall-thru
    case rsum:
      atomicAdd(&(hash_results[val_idx]), value);
      break;
    }
  }
}

template <typename Tkey, typename Tval> __global__
void fillTable(Tkey* key_columns,
	       size_t num_key_rows,
	       size_t num_key_cols,
	       Tval* value_columns,
	       size_t num_val_rows,
	       size_t num_val_cols,
	       int* hash_key_idx,
	       int* hash_count,
	       Tval* hash_results,
	       size_t len_hash_table,
	       size_t num_ops,
         int* overflow_flag
         )
{

  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t offset = gridDim.x * blockDim.x;
  for (size_t i = idx; i < num_key_rows; i += offset) {
    // try inserting, assume there is enough space
    size_t curPos = HashKey(i, key_columns, num_key_rows, num_key_cols) % len_hash_table;

    unsigned int collisionCount = 0;
    bool isInserted = false;
    while (!isInserted) {
      // first read the value out 
      int old = hash_key_idx[curPos];
      // if it is -1, try update, else don't
      if (old == -1) 
	old = atomicCAS(&(hash_key_idx[curPos]), -1, i);
      // now old contains either -1 or a new address, if it is a new address meaning other thread claimed it
      
      if (old != -1) {
	// note: old should not contain -1 now, safe to cast to size_t
	if (!keyEqualCM<Tkey>(key_columns, (size_t)old, i, num_key_rows, num_key_cols)) {
	  // collision
	  curPos = (curPos + 1) % len_hash_table; // linear probing
	  if (++collisionCount >= len_hash_table * 0.75)
	    break; // break the loop if it looped over the hash table and still failed
	  continue;
	}
      }
      // now it is safe to update the entry
      isInserted = true;
      updateEntry<Tval>(value_columns, num_val_rows, num_ops, i, curPos, hash_results, &(hash_count[curPos]), len_hash_table);
    }
    if (!isInserted) {
      // Do sth in the case of overflowing hash table
      overflow_flag[0] = 1;
      //printf("Overflow happened at %d \n", len_hash_table);
    }
  }
}

template <typename Tkey, typename Tval> __global__
void fillTable_privatization(Tkey* key_columns,
			     size_t num_key_rows,
			     size_t num_key_cols,
			     Tval* value_columns,
			     size_t num_val_rows,
			     size_t num_val_cols,
			     int* hash_key_idx,
			     int* hash_count,
			     Tval* hash_results,
			     size_t len_hash_table,
			     size_t len_shared_hash_table,
			     size_t num_ops)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t offset = gridDim.x * blockDim.x;
  __shared__ unsigned int filled_hash_table_shared;
  extern __shared__ char hash_table_shared[];
  int* s_hash_key_idx = (int*)hash_table_shared;
  int* s_hash_count = (int*)&(hash_table_shared[len_shared_hash_table*sizeof(int)]);
  size_t s_offset = (2*len_shared_hash_table*sizeof(int) + sizeof(Tval) - 1) / sizeof(Tval);
  
  Tval* s_hash_results = (Tval*)&(hash_table_shared[s_offset*sizeof(Tval)]);
  
  // initialization
  for (size_t i = threadIdx.x; i < len_shared_hash_table; i += blockDim.x) {
    s_hash_key_idx[i] = -1;
    s_hash_count[i] = 0;
    for (size_t j = 0; j < num_ops; ++j) {
      // replace following with specialized limit template in the future
      if (ops_c[j] == rmin) {
	s_hash_results[j * len_shared_hash_table + i] = cuda_custom::limits<Tval>::max();
      } else if (ops_c[j] == rmax) {
        s_hash_results[j * len_shared_hash_table + i] = cuda_custom::limits<Tval>::lowest();
      } else {
	s_hash_results[j * len_shared_hash_table + i] = 0;
      }
    }
  }
  if (threadIdx.x == 0) filled_hash_table_shared = 0;
  __syncthreads();
  
  for (size_t i = idx; i < num_key_rows; i += offset) {
    // try inserting, assume there is enough space
    size_t curPos = HashKey(i, key_columns, num_key_rows, num_key_cols) % len_shared_hash_table;
    unsigned int collisionCount = 0;
    bool isInserted = false;
    while (!isInserted) {
      // quit if shared hash table is 80% full
      if (filled_hash_table_shared >= ( 8 * len_shared_hash_table / 10)) break;
      int old = s_hash_key_idx[curPos];
      // if it is -1, try update, else don't
      if (old == -1) 
	old = atomicCAS(&(s_hash_key_idx[curPos]), -1, i);
      // now old contains either -1 or a new address, if it is a new address meaning other thread claimed it
      
      if (old != -1) {
	// note: old should not contain -1 now, safe to cast to size_t
	if (!keyEqualCM<Tkey>(key_columns, (size_t)old, i, num_key_rows, num_key_cols)) {
	  // collision
	  curPos = (curPos + 1) % len_shared_hash_table; // linear probing
	  if (++collisionCount == len_shared_hash_table)
	    break; // break the loop if it looped over the hash table and still failed
	  continue;
	}
      } else {
	atomicAdd(&filled_hash_table_shared, 1);
      }
      // now it is safe to update the entry
      isInserted = true;
      updateEntry<Tval>(value_columns, num_val_rows, num_ops, i, curPos, s_hash_results, &(s_hash_count[curPos]), len_shared_hash_table);
    }
    // if current column not inserted, insert to global one
    curPos = HashKey(i, key_columns, num_key_rows, num_key_cols) % len_hash_table;
    collisionCount = 0;
    while (!isInserted) {
      // first read the value out

      int old = hash_key_idx[curPos];
      // if it is -1, try update, else don't
      if (old == -1) 
	old = atomicCAS(&(hash_key_idx[curPos]), -1, i);
      // now old contains either -1 or a new address, if it is a new address meaning other thread claimed it
      
      if (old != -1) {
	// note: old should not contain -1 now, safe to cast to size_t
	if (!keyEqualCM<Tkey>(key_columns, (size_t)old, i, num_key_rows, num_key_cols)) {
	  // collision
	  curPos = (curPos + 1) % len_hash_table; // linear probing
	  if (++collisionCount == len_hash_table)
	    break; // break the loop if it looped over the hash table and still failed
	  continue;
	}
      }
      // now it is safe to update the entry
      isInserted = true;
      updateEntry<Tval>(value_columns, num_val_rows, num_ops, i, curPos, hash_results, &(hash_count[curPos]), len_hash_table);

    }
    if (!isInserted) {
      // Do sth in the case of overflowing hash table
    }
  }
  __syncthreads();
  for (size_t i = threadIdx.x; i < len_shared_hash_table; i += blockDim.x) {
    int real_idx = s_hash_key_idx[i];
    if (real_idx != -1) {
      size_t curPos = HashKey(real_idx, key_columns, num_key_rows, num_key_cols) % len_hash_table;
      int collisionCount = 0;
      bool isInserted = false;
      while (!isInserted) {
	// first read the value out
	int old = hash_key_idx[curPos];
	// if it is -1, try update, else don't
	if (old == -1) 
	  old = atomicCAS(&(hash_key_idx[curPos]), -1, real_idx);
	// now old contains either -1 or a new address, if it is a new address meaning other thread claimed it
	
	if (old != -1) {
	  // note: old should not contain -1 now, safe to cast to size_t
	  if (!keyEqualCM<Tkey>(key_columns, (size_t)old, real_idx, num_key_rows, num_key_cols)) {
	    // collision
	    curPos = (curPos + 1) % len_hash_table; // linear probing
	    if (++collisionCount == len_hash_table)
	      break; // break the loop if it looped over the hash table and still failed
	    continue;
	  }
	}
	// now it is safe to update the entry
	isInserted = true;
	updateEntry<Tval>(s_hash_results, len_shared_hash_table, num_ops,
			  i, curPos, hash_results, &(hash_count[curPos]), len_hash_table, s_hash_count[i]);
	
      }
      if (!isInserted) {
	// Do sth in the case of overflowing hash table
      }
    }
  } 
}

template <typename Tval> __global__
void initializeVariable(int* hash_key_idx,
			int* hash_count,
			Tval* hash_results,
			size_t len_hash_table,
			size_t num_ops)
{
  // each thread responsible for one entry (with thread coarsening)
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t offset = gridDim.x * blockDim.x;
  for (size_t i = idx; i < len_hash_table; i += offset) {
    hash_key_idx[i] = -1;
    hash_count[i] = 0;
    for (size_t j = 0; j < num_ops; ++j) {
      // replace following with specialized limit template in the future
      if (ops_c[j] == rmin) {
	hash_results[j * len_hash_table + i] = cuda_custom::limits<Tval>::max();
      } else if (ops_c[j] == rmax) {
        hash_results[j * len_hash_table + i] = cuda_custom::limits<Tval>::lowest();
      } else {
	hash_results[j * len_hash_table + i] = 0;
      }
    }
  }
}


template <typename Tval> __global__
void copyUnique(
      int *hashTable_idxs_d, //where key resides in hash vector
      int *hash_key_idx_d, //where key resides in original key matrix 
      Tval* key_columns_d, 
      Tval* output_key_columns_d, 
      int num_output_rows, 
      int num_key_columns,
      int num_key_rows)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < num_output_rows){//num_output_rows){
    // printf("%d |    : %d : %d \n ",hash_key_idx_d[hashTable_idxs_d[idx]], key_columns_d[hash_key_idx_d[hashTable_idxs_d[idx]]+num_key_rows*0], key_columns_d[hash_key_idx_d[hashTable_idxs_d[idx]]+num_key_rows*1]);

    for (int i = 0; i < num_key_columns; i++){//each column of key matrix
      // printf(" : %d",key_columns_d[hash_key_idx_d[hashTable_idxs_d[idx]]+num_key_rows*i]);
      output_key_columns_d[idx+num_output_rows*i] = key_columns_d[hash_key_idx_d[hashTable_idxs_d[idx]]+num_key_rows*i];//copy original key entry to output
    }
    // printf("\n");
    idx += gridDim.x*blockDim.x;//increment idx by thread space
  }
}

template <typename Tval> __global__
void copyValues(
      int *hashTable_idxs_d, 
      Tval* hash_results_d,
      int *hash_count_d,
      Tval* value_columns_d, 
      Tval* output_value_columns_d, 
      int num_output_rows, 
      int num_value_columns,
      int num_value_rows,
      size_t num_ops,
      size_t len_hash_table
    )
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //printf("%d\n",idx);
  while (idx < num_output_rows){
    for (size_t i = 0; i < num_ops; ++i) {
      size_t val_idx = i * len_hash_table + hashTable_idxs_d[idx];
      switch(ops_c[i]) {
      case rmin:
        output_value_columns_d[idx+num_output_rows*i] = hash_results_d[val_idx];//copy result to output
        break;
      case rmax:
        output_value_columns_d[idx+num_output_rows*i] = hash_results_d[val_idx];//copy result to output
      break;
      case rcount:
        output_value_columns_d[idx+num_output_rows*i] = hash_results_d[val_idx];//copy result to output
      break;
      case rmean: 
        output_value_columns_d[idx+num_output_rows*i] = hash_results_d[val_idx]/hash_count_d[hashTable_idxs_d[idx]];//copy result to output
        break;
      case rsum:
        output_value_columns_d[idx+num_output_rows*i] = hash_results_d[val_idx];//copy result to output
        break;
      }
    }

    idx += gridDim.x*blockDim.x;//increment idx by thread space
  }
}

struct is_pos
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x >= 0;
  }
};

extern std::mt19937 gen;

template <typename T> __host__
unsigned int predictTableLength_CPU(const T* key_columns,
				    size_t num_key_rows,
				    size_t num_key_columns)
{
  // Predict Hash Table length based on 2 state transfer matrix
  unsigned int numEqual = 0;
  unsigned int numTotal = 0;

  std::uniform_int_distribution<unsigned int> keyRange(0, num_key_rows-1);

  // max try 1% of key_rows
  for (size_t i=0; i < num_key_rows/100; ++i) {
    size_t idx1 = keyRange(gen);
    size_t idx2 = keyRange(gen);
    bool result = keyEqualCM(key_columns, idx1, idx2, num_key_rows, num_key_columns);
    if (result) 
      ++numEqual;
    ++numTotal;
    if (numEqual == 10)
      break;
  }
  if (numEqual < 2) // very few sample, return 1/4 of original
    return num_key_rows / 4;
  return (unsigned int) 2.6f * ((float)(numTotal) / numEqual);
}

__global__
void fillCURANDState(curandState* state, unsigned long seed)
{
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}

template <typename T> __global__
void predictTableLength_GPU(const T* key_columns,
			    size_t num_key_rows,
			    size_t num_key_columns,
			    size_t iterations,
			    unsigned int* count,
			    curandState* state)
{  
#ifdef DEBUG
  constexpr unsigned int BLOCKSIZE = 512;
#else
  constexpr unsigned int BLOCKSIZE = 1024;
#endif

  __shared__ unsigned int count_shared[3*BLOCKSIZE];
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  // initial shared memory
  for (size_t i = 0; i < 3; ++i) {
    count_shared[i*BLOCKSIZE + threadIdx.x] = 0;
  }
  for (size_t i = 0; i < iterations; ++i) {
    unsigned int test_idx[3];
    bool result[3];
    for (size_t j = 0; j < 3; ++j) 
      test_idx[j] = floorf(curand_uniform(&state[idx]) * num_key_rows);
    // compare keys
    for (size_t j = 0; j < 3; ++j) 
      result[j] = keyEqualCM(key_columns, test_idx[j],
			     test_idx[(j+1)%3], num_key_rows,
			     num_key_columns);
    if (result[0] && result[1]) // any two is true then 3 are equal
      count_shared[threadIdx.x] += 1;
    else if (result[0] || result[1] || result[2]) // any one is true then 2 are equal
      count_shared[BLOCKSIZE + threadIdx.x] += 1;
    else // three are different
      count_shared[BLOCKSIZE*2 + threadIdx.x] += 1;
  }
  __syncthreads();
  // reduction
  for (size_t stride = (blockDim.x >> 1);
       stride >= 1;
       stride >>= 1) {
    if (threadIdx.x < stride) {
      for (size_t i = 0; i < 3; ++i) {
	count_shared[threadIdx.x + BLOCKSIZE*i]
	  += count_shared[threadIdx.x + BLOCKSIZE*i + stride];  
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    for (size_t i = 0; i < 3; ++i) {
      count[i] = count_shared[BLOCKSIZE*i];
    }
}
