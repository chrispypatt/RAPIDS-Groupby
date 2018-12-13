#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <pair.h>
#include <thrust/iterator/permutation_iterator.h>

#define BLOCK_SIZE 2048

//Launch reduction kernels for each column based on their specified operation
template <typename T>
void groupby_GPU(T* key_columns[], int num_key_columns, int num_key_rows,
	T* value_columns[], int num_value_columns, int num_value_rows,
	reductionType ops[], int num_ops, T* output_keys[], T* output_values[])
{
	//Perform hashing
	dim3 dimGrid(ceil((float)num_key_columns/(float)BLOCK_SIZE),1,1);
	dim3 dimBlock(BLOCK_SIZE,1,1);

	uint32_t* d_hash_keys;

	cudaMemset(d_hash_keys, 0, num_key_rows*sizeof(uint32_t));

	MurmurHash3_x64_128_hash<<<dimGrid,dimBlock>>>(key_columns,
		num_key_columns,
		num_key_rows,
		MurmurHash3_x64_128_tab,
		d_hash_keys, 0);

	//create index array for sorting. 
	thrust::device_vector<int> d_i(num_key_rows), key_locations(num_value_rows);
	thrust::sequence(thrust::host, d_i.begin(), d_i.end()); 

	//sort by key, also sort value indices. The result can be used to sort the actual data arrays later
	thrust::sort_by_key(d_hash_keys, d_hash_keys + num_key_columns, d_i);

	//Find count of unqiue keys - save location of where to find each key
	thrust::copy(d_hash_keys, d_hash_keys + num_key_columns,d_unique_keys.begin());
	thrust::copy(d_i.begin(), d_i.end(), key_locations.begin()); 
	int num_output_rows = thrust::unique_by_key(d_unique_keys.begin(), d_unique_keys.end(), key_locations.begin()).first - d_hash_keys.begin();


	//setup output arrays
	output_keys = new T[num_output_rows*num_key_columns];
	output_values = new T[num_output_rows*num_value_columns];

	//copy back unique keys
	for (int i = 0; i<num_key_columns, i++){//i represents column of key output
		thrust::copy_n(thrust::make_permutation_iterator(key_columns + (i*num_output_rows), key_locations.begin()), num_output_rows, sorted_col.begin());
	}

	//iterate though all columns of the matrix. Perfrom the operation corresponding to that column
	for (int i = 0; i<num_ops, i++){//i represents column of output
		//get this column of data. copy does [first, last) 
		int start = i*num_value_rows;
		int end = (i+1)*num_value_rows;
		//the column is not sorted yet so use d_i to sort!
		thrust::device_vector<T> sorted_col(num_value_rows);
		thrust::copy_n(thrust::make_permutation_iterator(value_columns + start, d_i.begin()), num_value_rows, sorted_col.begin());

		thrust::copy(value_columns + start, value_columns + end,col.begin());

		switch(input.ops[i]){
			case max:
				thrust::equal_to<T> binary_pred;
				thrust::maximum<T> binary_op;
				thrust::pair<int*,int*> end = thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_columns, thrust::make_constant_iterator(1), output_keys.begin(), output_vector.begin(), binary_pred, binary_op);
				break;
			case min:
				thrust::equal_to<T> binary_pred;
				thrust::minimum<T> binary_op;
				thrust::pair<int*,int*> end = thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_columns, thrust::make_constant_iterator(1), output_keys.begin(), output_vector.begin(), binary_pred, binary_op);
				break;
			case sum:
				thrust::pair<int*,int*> end = thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_columns, sorted_col.begin(), output_keys.begin(), output_vector.begin());
				break;
			case count:
				thrust::equal_to<T> binary_pred;
				thrust::plus<T> binary_op;
				thrust::pair<int*,int*> end = thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_columns, thrust::make_constant_iterator(1), output_keys.begin(), output_vector.begin(), binary_pred, binary_op);
				break;
			case mean:
				thrust::device_vector<T>  output_sums(num_output_rows);
				thrust::equal_to<T> binary_pred;
				thrust::plus<T> binary_op;
				//get count for each key
				thrust::pair<int*,int*> end = thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_columns, thrust::make_constant_iterator(1), output_keys.begin(), output_vector.begin(), binary_pred, binary_op);
				//Get sum for each key
				thrust::pair<int*,int*> end1 = thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_columns, sorted_col.begin(), output_keys.begin(), output_sums.begin());
				//Perform division: Sums/Counts
				thrust::divides<T> op;
				thrust::transform(output_vector.begin(), output_vector.end(), output_counts, output_vector.begin(), op);
				break;
		}
		int output_start = i*num_output_rows;
		thrust::copy(output_vector.begin(), output_vector.end(),output_values + output_start;
	}
}