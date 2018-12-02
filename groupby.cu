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
	thrust::device_vector<T> d_hash_keys(num_key_rows);

	//create index array for sorting. 
	thrust::device_vector<int> d_i(num_key_rows);
	thrust::sequence(thrust::host, d_i.begin(), d_i.end()); 

	//sort by key, also sort values. The result can be used to sort the actual data arrays later
	thrust::sort_by_key(d_hash_keys.begin(), d_hash_keys.end(), d_i);

	//Find the position of first key for each group
	thrust::device_vector<int> d_key_starts(num_key_rows);
	thrust::sequence(d_key_starts.begin(), d_key_starts.end()); //this sequence represents the index for each key
	int new_end  = (thrust::unique_by_key(d_hash_keys.begin(), d_hash_keys.end(), d_key_starts.begin())).first - d_hash_keys.begin();
	//after unique_by_key, d_key_starts holds the start of each group.

	//setup output arrays
	int num_output_rows = new_end;
	output_keys = new T[num_output_rows];
	output_values = new T[num_output_rows*num_value_columns];

	//copy back unique keys
	thrust::copy(d_hash_keys.begin(), d_hash_keys.begin() + new_end, output_keys;


	//iterate though all columns of the matrix. Perfrom the operation corresponding to that column
	for (int i = 0; i<num_ops, i++){//i represents column of output
		//get this column of data. copy does [first, last) 
		int start = i*num_value_rows;
		int end = (i+1)*num_value_rows;
		thrust::device_vector<T> col(num_value_rows), sorted_col(num_value_rows);
		thrust::copy(value_columns + start, value_columns + end,col.begin());

		//the column is not sorted yet so use d_i to sort!
		thrust::copy_n(thrust::make_permutation_iterator(col.begin(), d_i.begin()), num_value_rows, sorted_col.begin());
		for (int j = 0; j < new_end; j++){ //iterate over the groups of keys... j = output row
			int start = d_key_starts[j];
			int end;
			if (j < new_end-1){
				end = d_key_starts[j+1];
			}else{
				end = col.end();
			}
			T val;
			switch(input.ops[i]){
				case max:
					val = *(thrust::max_element(col.begin() + start, col.begin() + end));
					break;
				case min:
					val = *((thrust::min_element(col.begin() + start, col.begin() + end));
					break;
				case sum:
					val = ((T)thrust::reduce(col.begin() + start, col.begin() + end));
					break;
				case count:
					val = (T) (end-start)+1;
					break;
				case mean:
					T count = (T) (end-start)+1;
					val = ((T)thrust::reduce(col.begin() + start, col.begin() + end))/((T)count);
					break;
			}
			output_values[i*num_output_rows+j] = val;
		}
	}
}