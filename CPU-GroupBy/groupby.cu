#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>
#include "cpuGroupby.h"
#include "HashFunc.h"
#include <thrust/iterator/permutation_iterator.h>

#define BLOCK_SIZE 2048

//Launch reduction kernels for each column based on their specified operation
template <typename T>
void groupby_GPU(T* key_columns, int num_key_columns, int num_key_rows,
	T* value_columns, int num_value_columns, int num_value_rows,
	reductionType* ops, int num_ops, T* output_keys, T* output_values)
{
	//Perform hashing
	dim3 dimGrid(ceil((float)num_key_columns/(float)BLOCK_SIZE),1,1);
	dim3 dimBlock(BLOCK_SIZE,1,1);

	uint32_t* hash_keys;
	cudaMalloc((void **) &hash_keys, num_key_rows * sizeof(uint32_t));

	//TODO: ADD hashing here. 

	thrust::device_ptr<uint32_t> d_hash_keys(hash_keys);
	thrust::fill(d_hash_keys, d_hash_keys + num_key_rows, (int) 0);


	//create index array for sorting. 
	thrust::device_vector<int> d_i(num_key_rows), key_locations(num_value_rows);
	thrust::device_vector<uint32_t> d_unique_keys(num_value_rows);
	thrust::sequence(thrust::host, d_i.begin(), d_i.end()); 

	//sort by key, also sort value indices. The result can be used to sort the actual data arrays later
	thrust::sort_by_key(d_hash_keys, d_hash_keys + num_key_columns, d_i);

	//Find count of unqiue keys - save location of where to find each key
	thrust::copy(d_hash_keys, d_hash_keys + num_key_rows,d_unique_keys.begin());
	thrust::copy(d_i.begin(), d_i.end(), key_locations.begin()); 
	thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<int>::iterator> end = thrust::unique_by_key(d_unique_keys.begin(), d_unique_keys.end(), key_locations.begin());
	
	int num_output_rows = *end.first;

	//setup output arrays
	output_keys = new T[num_output_rows*num_key_columns];
	output_values = new T[num_output_rows*num_value_columns];

	//copy back unique keys
	for (int i = 0; i<num_key_columns; i++){//i represents column of key output
		thrust::copy_n(thrust::make_permutation_iterator(key_columns + (i*num_output_rows), key_locations.begin()), num_output_rows, output_keys);
	}

	//iterate though all columns of the matrix. Perfrom the operation corresponding to that column
	for (int i = 0; i<num_ops; i++){//i represents column of output
		//get this column of data. copy does [first, last) 
		int start = i*num_value_rows;
		// int end = (i+1)*num_value_rows;
		//the column is not sorted yet so use d_i to sort!
		thrust::device_vector<T> sorted_col(num_value_rows);
		uint32_t* output_ptr;
		cudaMalloc((void **) &output_ptr, num_value_rows * sizeof(T));
		thrust::device_ptr<T> output(hash_keys);
		thrust::copy_n(thrust::make_permutation_iterator(value_columns + start, d_i.begin()), num_value_rows, sorted_col.begin());

		thrust::pair<int*,int*> n_end;
		thrust::equal_to<T> eq;
		thrust::minimum<T> mn;
		thrust::maximum<T> mx;
		thrust::plus<T> pls;
		switch(ops[i]){
			case rmax:
				n_end = thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_columns, thrust::make_constant_iterator(1), output_keys, output, eq, mx);
				break;
			case rmin:
				n_end = thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_columns, thrust::make_constant_iterator(1), output_keys, output, eq, mn);
				break;
			case rsum:
				n_end = thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_columns, sorted_col.begin(), output_keys, output);
				break;
			case rcount:
				n_end = thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_columns, thrust::make_constant_iterator(1), output_keys, output, eq, pls);
				break;
			case rmean:
				thrust::device_vector<T>  output_sums(num_output_rows);
				//get count for each key
				n_end = thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_columns, thrust::make_constant_iterator(1), output_keys, output, eq, pls);
				//Get sum for each key
				n_end = thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_columns, sorted_col.begin(), output_keys, output_sums);
				//Perform division: Sums/Counts
				thrust::divides<T> div;
				thrust::transform(output, output + n_end.first, output_sums, output, div);
				break;
		}
		int output_start = i*num_output_rows;
		thrust::copy(output, output + n_end.first, output_values + output_start);
	}
}