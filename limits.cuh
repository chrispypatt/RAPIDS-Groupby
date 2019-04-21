// With reference to the numeric_limits header
// Macro with reference in following
// https://en.cppreference.com/w/cpp/types/climits

#include <cstdint>
#include <cfloat>

namespace cuda_custom {

  template <typename T>
  struct limits
  {
    __host__ __device__ static constexpr T
    min() noexcept {return T();}
    __host__ __device__ static constexpr T
    max() noexcept {return T();}
    __host__ __device__ static constexpr T
    lowest() noexcept {return T();}
  };
  
  template <>
  struct limits<int>
  {
    __host__ __device__ static constexpr int
    min() noexcept {return INT_MIN;}
    __host__ __device__ static constexpr int
    max() noexcept {return INT_MAX;}
    __host__ __device__ static constexpr int
    lowest() noexcept {return INT_MIN;}
  };

  template <>
  struct limits<unsigned int>
  {
    __host__ __device__ static constexpr unsigned int
    min() noexcept {return 0;}
    __host__ __device__ static constexpr unsigned int
    max() noexcept {return UINT_MAX;}
    __host__ __device__ static constexpr unsigned int
    lowest() noexcept {return 0;}
  };

  template <>
  struct limits<float>
  {
    __host__ __device__ static constexpr float
    min() noexcept {return FLT_MIN;}
    __host__ __device__ static constexpr float
    max() noexcept {return FLT_MAX;}
    __host__ __device__ static constexpr float
    lowest() noexcept {return -FLT_MAX;}
  };
}
