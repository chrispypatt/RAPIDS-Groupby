#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
enum reduction_op { max, min, sum, count, mean };

//Launch reduction kernels for each column based on their specified operation
template <typename T>
void perform_operators(T* key_columns[], int num_key_columns, int num_key_rows,
	T* value_columns[], int num_value_columns, int num_value_rows,
	reduction_op ops[], int num_ops, T* output_keys[], T* output_values[])
{
	//TODO: hashing here
	thrust::device_vector<T> d_hash_keys(num_key_rows), d_unique_keys(num_key_rows);

	//create index array for sorting. 
	thrust::device_vector<int> d_i(num_key_rows);
	thrust::sequence(thrust::host, d_i.begin(), d_i.end()); 

	//sort by key, also sort values. The result can be used to sort the actual data arrays later
	thrust::sort_by_key(d_hash_keys.begin(), d_hash_keys.end(), d_i);


	//Find count of unqiue keys
	thrust::copy(d_hash_keys.begin(), d_hash_keys.end(),d_unique_keys.begin());
	int num_output_rows = thrust::unique_by_key(d_unique_keys.begin(), d_unique_keys.end()).first - d_hash_keys.begin();


	//setup output arrays
	output_keys = new T[num_output_rows];
	output_values = new T[num_output_rows*num_value_columns];

	//copy back unique keys
	// thrust::copy(d_hash_keys.begin(), d_hash_keys.begin() + new_end, output_keys);


	//iterate though all columns of the matrix. Perfrom the operation corresponding to that column
	for (int i = 0; i<num_ops, i++){//i represents column of output
		//get this column of data. copy does [first, last) 
		int start = i*num_value_rows;
		int end = (i+1)*num_value_rows;
		//the column is not sorted yet so use d_i to sort!
		thrust::device_vector<T> sorted_col(num_value_rows);
		thrust::copy_n(thrust::make_permutation_iterator(value_columns + start, d_i.begin()), num_value_rows, sorted_col.begin());

		// thrust::device_vector<T> col(num_value_rows), sorted_col(num_value_rows);
		// thrust::device_vector<T> output_keys(num_output_rows), output_vector(num_output_rows);
		// thrust::copy(value_columns + start, value_columns + end,col.begin());

		//the column is not sorted yet so use d_i to sort!
		// thrust::copy_n(thrust::make_permutation_iterator(col.begin(), d_i.begin()), num_value_rows, sorted_col.begin());

		switch(input.ops[i]){
			case max:
				thrust::equal_to<T> binary_pred;
				thrust::maximum<T> binary_op;
				thrust::pair<int*,int*> end = thrust::reduce_by_key(d_hash_keys.begin(), d_hash_keys.end(), thrust::make_constant_iterator(1), output_keys.begin(), output_vector.begin(), binary_pred, binary_op);
				break;
			case min:
				thrust::equal_to<T> binary_pred;
				thrust::minimum<T> binary_op;
				thrust::pair<int*,int*> end = thrust::reduce_by_key(d_hash_keys.begin(), d_hash_keys.end(), thrust::make_constant_iterator(1), output_keys.begin(), output_vector.begin(), binary_pred, binary_op);
				break;
			case sum:
				thrust::pair<int*,int*> end = thrust::reduce_by_key(d_hash_keys.begin(), d_hash_keys.end(), sorted_col.begin(), output_keys.begin(), output_vector.begin());
				break;
			case count:
				thrust::equal_to<T> binary_pred;
				thrust::plus<T> binary_op;
				thrust::pair<int*,int*> end = thrust::reduce_by_key(d_hash_keys.begin(), d_hash_keys.end(), thrust::make_constant_iterator(1), output_keys.begin(), output_vector.begin(), binary_pred, binary_op);
				break;
			case mean:
				thrust::device_vector<T>  output_sums(num_output_rows);
				thrust::equal_to<T> binary_pred;
				thrust::plus<T> binary_op;
				//get count for each key
				thrust::pair<int*,int*> end = thrust::reduce_by_key(d_hash_keys.begin(), d_hash_keys.end(), thrust::make_constant_iterator(1), output_keys.begin(), output_vector.begin(), binary_pred, binary_op);
				//Get sum for each key
				thrust::pair<int*,int*> end1 = thrust::reduce_by_key(d_hash_keys.begin(), d_hash_keys.end(), sorted_col.begin(), output_keys.begin(), output_sums.begin());
				//Perform division: Sums/Counts
				thrust::divides<T> op;
				thrust::transform(output_vector.begin(), output_vector.end(), output_counts, output_vector.begin(), op);
				break;
		}
		int output_start = i*num_output_rows;
		thrust::copy(output_vector.begin(), output_vector.end(),output_values + output_start;
	}
}