//
//  HashFunc.cpp
//  
//
//  Created by menglu liang on 12/6/18.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cstdint>
#include <murmur3.h>
#include "HashFunc.h"


//there are three functions
//typedef unsigned char uint8_t;
//typedef unsigned int uint32_t;
//typedef unsigned __int64 uint64_t;
//MurmurHash3_x86_32 has the lowest throughput, but also the lowest latency.
//If you're making a hash table that usually has small keys, this is probably
//the one you want to use on 32-bit machines. It has a 32-bit output.

//MurmurHash3_x86_128 is also designed for 32-bit systems, but produces a 128-bit output,
//and has about 30% higher throughput than the previous hash. Be warned, though,
//that its latency for a single 16-byte key is about 86% longer!

//MurmurHash3_x64_128 is the best of the lot, if you're using a 64-bit machine.
//Its throughput is 250% higher than MurmurHash3_x86_32, but it has roughly the same latency. It has a 128-bit output.
//MurmurHash3_x86_32(const void * key, int len, uint32_t seed, void * out);
//MurmurHash3_x86_128(const void * key, int len, uint32_t seed, void * out);
//MurmurHash3_x64_128(const void * key, int len, uint32_t seed, void * out);

//int main(int argc, char **argv) {
//    uint32_t hash[4];                /* Output for the hash */
 //   uint32_t seed = 42;              /* Seed value for hash */
    
//    if (argc != 2) {
//        printf("usage: %s \"string to hash\"\n", argv[0]);
//        exit(1);
//    }
 //   MurmurHash3_x86_32(argv[1], strlen(argv[1]), seed, hash);
//    MurmurHash3_x86_128(argv[1], strlen(argv[1]), seed, hash);
//    MurmurHash3_x64_128(argv[1], strlen(argv[1]), seed, hash);
//    return 0;
// }

//simple CPU version
//
//uint32_t MurmurHash3_x86_32(const uint8_t* key, size_t len, uint32_t seed) {
//    uint32_t h = seed;
 //   if (len > 3) {
 //       const uint32_t* key_x4 = (const uint32_t*) key;
 //       size_t i = len >> 2;
//        do {
//            uint32_t k = *key_x4++;
//            k *= 0xcc9e2d51;
//            k = (k << 15) | (k >> 17);
//            k *= 0x1b873593;
//            h ^= k;
//            h = (h << 13) | (h >> 19);
 //           h = (h * 5) + 0xe6546b64;
 //       } while (--i);
//        key = (const uint8_t*) key_x4;
//    }
//    if (len & 3) {
//        size_t i = len & 3;
//        uint32_t k = 0;
//        key = &key[i - 1];
//        do {
//            k <<= 8;
//            k |= *key--;
 //       } while (--i);
 //       k *= 0xcc9e2d51;
//        k = (k << 15) | (k >> 17);
 //       k *= 0x1b873593;
//        h ^= k;
 //   }
  //  h ^= len;
//    h ^= h >> 16;
//    h *= 0x85ebca6b;
//    h ^= h >> 13;
//    h *= 0xc2b2ae35;
//    h ^= h >> 16;
//    return h;
//}



// host code
template <typename T>
__host__ void MurmurHash3_x64_128_host(const T* key_columns[],
                              const int num_key_columns,
                              const int num_key_rows,
                              const uint32_t MurmurHash3_x64_128_tab[],
                              uint32_t MurmurHash3_x64_128_result[],uint32_t seed)
{
    uint32_t* MurmurHash3_x64_128_result_d;
    cudaMemcpytoSymbol(MurmurHash3_x64_128_tab_c, MurmurHash3_x64_128_tab, 256*sizeof(uint32_t));
    cudaMalloc((void**)&MurmurHash3_x64_128_result_d, num_key_rows * sizeof(uint32_t));
    size_t dimGrid = (num_key_rows + 1023) / 1024;
    crc32_hash<<<dimGrid, 1024>>>(key_columns, num_key_columns, num_key_rows,
                                  MurmurHash3_x64_128_tab_c, MurmurHash3_x64_128_result_d);
    cudaMemcpy(MurmurHash3_x64_128_result, MurmurHash3_x64_128_result_d, num_key_rows * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
}
