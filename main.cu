//
//  main.cpp
//  RAPIDS
//
//  Created by Aaron on 11/19/18.
//  Copyright © 2018 Aaron Nightingale. All rights reserved.
//
//  This is a simple CPU groupby class (currently only MAX is implemented)
//  one key column and one value column.

#include <iostream>
#include <time.h>
#include <algorithm>
#include <chrono>
#include "cpuGroupby.h"

#include "groupby.cu"

using namespace std;
int main(int argc, const char * argv[]) {
        

	using Time = std::chrono::high_resolution_clock;
	using fsec = std::chrono::duration<float>;

        int num_rows = 100000;
        int num_key_cols = 2;
        int num_val_cols = 3;
        int num_distinct_keys = 3;
        if (argc == 2){
                num_rows = atoi(argv[1]);
        }else if(argc ==4){
                num_rows = atoi(argv[1]);
                num_key_cols = atoi(argv[2]);
                num_val_cols = atoi(argv[3]);
        }else if(argc ==5){
                num_rows = atoi(argv[1]);
                num_key_cols = atoi(argv[2]);
                num_val_cols = atoi(argv[3]);
                num_distinct_keys = atoi(argv[4]);
        }
        // Setting up the CPU groupby
        cpuGroupby slowGroupby(num_key_cols, num_val_cols, num_rows);

        slowGroupby.fillRand(num_distinct_keys, num_rows);

        int *original_key_columns;
	cudaMallocHost((void**)&original_key_columns, sizeof(int)*num_key_cols*num_rows);
        int *original_value_columns;
	cudaMallocHost((void**)&original_value_columns, sizeof(int)*num_val_cols*num_rows);
        std::copy(slowGroupby.key_columns, slowGroupby.key_columns + num_key_cols*num_rows, original_key_columns);
        std::copy(slowGroupby.value_columns, slowGroupby.value_columns + num_val_cols*num_rows, original_value_columns);
        
        auto start = Time::now();

        slowGroupby.groupby();

        auto end = Time::now(); 
        fsec cpu_duration = end - start;

        // Insert GPU function calls here...
        int *gpu_output_keys, *gpu_output_values;
        int gpu_output_rows = 0;
        gpu_output_keys = (int *)malloc(slowGroupby.num_key_rows*slowGroupby.num_key_columns * sizeof(int));
        gpu_output_values = (int *)malloc(slowGroupby.num_value_rows*slowGroupby.num_value_columns * sizeof(int));

        start = Time::now();

        groupby_GPU(original_key_columns, slowGroupby.num_key_columns,
                slowGroupby.num_key_rows, original_value_columns, 
                slowGroupby.num_value_columns, slowGroupby.num_value_rows, 
                slowGroupby.ops, slowGroupby.num_ops,
                gpu_output_keys, gpu_output_values, gpu_output_rows);
        end = Time::now();
        
        slowGroupby.printGPUResults(gpu_output_keys, gpu_output_values);

        fsec gpu_duration = end - start;

        cout << "CPU time: " << cpu_duration.count() << " s" << endl;
        cout << "GPU time: " << gpu_duration.count() << " s" << endl;

        slowGroupby.validGPUResult(gpu_output_keys, gpu_output_values, gpu_output_rows);

        cudaFreeHost(original_value_columns);
        cudaFreeHost(original_key_columns);
        return 0;
}
