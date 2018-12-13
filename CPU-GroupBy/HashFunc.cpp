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
__global__ void MurmurHash3_x64_128_hash(const T* key_columns[],
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
        // uint32_t hash *= 0xc2b2ae35;
        for (size_t i = 0; i < Message_size; ++i) {
            hash = MurmurHash3_x64_128_tab[(hash ^ message)] ^ (hash >> 16);
        }
        MurmurHash3_x64_128_result[idx] ^= hash;
    }
}