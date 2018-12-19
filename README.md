# RAPIDS-Groupby
Repository for EE 5351 Applied Parallel Programming final project. This project is to implement RAPIDS Groupby function in CUDA.

Team member: Aaron Nightingale, Christopher Patterson, Jersin Nguetio, Menglu Liang, Tonglin Chen

The program is tested under nvcc 8.0 and GTX 1080 GPU.

To build the program, type 'make' in the root folder of the files.

Command line usage:

./groupby

will use the default setting: 100000 data entries, 2 key_columns, 3 row_columns, and 4 distinct key per column
			
./groupby <num_rows>

will use the default column settings and use the argument as number of data entries

./groupby <num_rows> <key_cols> <val_cols>

will use num_rows as data entries, key_cols as number of key columns and row_cols as number of row columns
while maintaining 4 distinct key per column

./groupby <num_rows> <key_cols> <val_cols> <distinct_keys_per_col>

will use all the parameters to populate the data

Notice: If the number of distinct keys in each column is m, n key_columns will generate m^n distinct keys.

The program will populate random data, compute on CPU and GPU then validate the results.

All rights reserved.

This repo is in progress.
Aaron wuz here