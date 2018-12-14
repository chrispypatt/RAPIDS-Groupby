//
//  HashFunc.cpp
//  
//
//  Created by menglu liang on 12/6/18.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "HashFunc.h"


template <typename T>
__global__ void MurmurHash3_x64_32_hash(const T* key_columns,
                           const int num_key_columns,
                           const int num_key_rows,
                           const uint32_t MurmurHash3_x64_128_tab[],
    					   uint32_t MurmurHash3_x64_128_result[], uint32_t seed)
{
    uint32_t hash = seed;
    // num of bytes in the message
	size_t Message_size = sizeof(T) * num_key_columns;
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_key_rows) {
        hash *= 0xc2b2ae35;
        for (size_t i = 0; i < Message_size; ++i) {
            uint8_t message = retrieve_byte<T>(key_columns, num_key_columns, idx, i);
            hash = MurmurHash3_x64_128_tab[(hash ^ message)] ^ (hash >> 16);
        }
        MurmurHash3_x64_128_result[idx] ^= hash;
    }
}

template <typename T>
__device__ uint8_t retrieve_byte(const T* key_columns[],
				 const int num_key_columns,
				 const size_t idx,
				 const size_t current_byte)
{
  size_t column_idx = current_byte / sizeof(T);
  size_t shift = 8 * (current_byte % sizeof(T));
  return (key_columns[column_idx][idx] >> shift) & 0xff;
}