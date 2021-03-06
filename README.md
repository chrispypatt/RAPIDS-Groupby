# RAPIDS-Groupby
Repository for EE 5351 Applied Parallel Programming final project on sorting based Groupby and is being updated for EE 5355 Algorithmic Techniques for Scalable Many-core Computing final project on hashed based Groupby. This project is to implement RAPIDS Groupby function in CUDA.

Team members EE 5351: Aaron Nightingale, Christopher Patterson, Jersin Nguetio, Menglu Liang, Tonglin Chen

Team members EE 5355: Tonglin Chen, Tianming Cui, Christopher Patterson, Yadu Kiran

Compiling Options:

 - dbg:		Compile target using debug mode, 512 thread per block

 - NOPRINT:  	Target will not print the result from groupby

 - PRIV:	  	Experimental privatization without relaunches

 - TESLA:	  	Explicitly use 32KB shared memory per block. Should only be used with PRIV

 - CPU_SAMPLE:	Target will run CPU sampling to predict the number of unique keys

 - GPU_SAMPLE:	Target will run GPU sampling to predict the number of unique keys

Command line usage for sorting based GroupBy:
```
make
./groupby                                                           # Data Entries:  100k, key_columns: 2, row_columns: 3, unique keys: 4
./groupby <num_rows>                                                # Data Entries: num_rows, key_columns: 2, row_columns: 3, unique keys: 4
./groupby <num_rows> <key_cols> <val_cols>                          # Data Entries: num_rows, key_columns: key_cols, row_columns: val_cols, unique keys: 4
./groupby <num_rows> <key_cols> <val_cols> <distinct_keys>  # Data Entries: num_rows, key_columns: key_cols, row_columns: val_cols, unique keys : distinct_keys
```

Command line usage for hashed based GroupBy:
```
make groupby_hash
./groupby_hash                                                           # Data Entries:  100k, key_columns: 2, row_columns: 3, unique keys: 4, hashtable rows: 1003
./groupby_hash <num_rows>                                                # Data Entries: num_rows, key_columns: 2, row_columns: 3, unique keys: 4, hashtable rows: 1003
./groupby_hash <num_rows> <key_cols> <val_cols>                          # Data Entries: num_rows, key_columns: key_cols, row_columns: val_cols, unique keys: 4, hashtable rows: 1003
./groupby_hash <num_rows> <key_cols> <val_cols> <distinct_keys> # Data Entries: num_rows, key_columns: key_cols, row_columns: val_cols, unique keys : distinct_keys, hashtable rows: 1003
./groupby_hash <num_rows> <key_cols> <val_cols> <distinct_keys> <hash_table_size> # Data Entries: num_rows, key_columns: key_cols, row_columns: val_cols, unique keys : distinct_keys, hashtable rows: hash_table_size
```

Command line usage for hashed based GroupBy with hash table size prediction:
```
make groupby_hash <mode>												 # mode: CPU_SAMPLE=1 for cpu sampling, GPU_SAMPLE=1 for gpu sampling
./groupby_hash                                                           # Data Entries:  100k, key_columns: 2, row_columns: 3, unique keys: 4
./groupby_hash <num_rows>                                                # Data Entries: num_rows, key_columns: 2, row_columns: 3, unique keys: 4
./groupby_hash <num_rows> <key_cols> <val_cols>                          # Data Entries: num_rows, key_columns: key_cols, row_columns: val_cols, unique keys: 4
./groupby_hash <num_rows> <key_cols> <val_cols> <distinct_keys>  # Data Entries: num_rows, key_columns: key_cols, row_columns: val_cols, unique keys : distinct_keys
```

The program will populate random data, compute on CPU and GPU then validate the results.

All rights reserved.

