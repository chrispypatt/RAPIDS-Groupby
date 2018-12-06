//
//  murmur3.hpp
//  
//
//  Created by menglu liang on 12/5/18.
//

#ifndef _MURMURHASH3_H_
#define _MURMURHASH3_H_

//-----------------------------------------------------------------------------
// Platform-specific functions and macros

// Microsoft Visual Studio

#if defined(_MSC_VER) && (_MSC_VER < 1600)
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned __int64 uint64_t;

// Other compilers

#else    // defined(_MSC_VER)

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
//-----------------------------------------------------------------------------

void MurmurHash3_x86_32(const void * key, int len, uint32_t seed, void * out);

void MurmurHash3_x86_128(const void * key, int len, uint32_t seed, void * out);

void MurmurHash3_x64_128(const void * key, int len, uint32_t seed, void * out);

//-----------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif

#endif // _MURMURHASH3_H_
