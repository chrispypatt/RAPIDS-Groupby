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
#include "cpuGroupby.h"

#include "groupby.cu"

using namespace std;
int main(int argc, const char * argv[]) {
    // Setting up the CPU groupby
    // 2 Key cols, 3 value cols, 100 rows
    cpuGroupby slowGroupby(2, 3, 100);
    // Filling arrays with 3 distinct keys, 100 distinct values
    slowGroupby.fillRand(3, 100);
    slowGroupby.groupby();

    
    // Insert GPU function calls here...
    int *gpu_output_keys, *gpu_output_values;
    gpu_output_keys = (int *)malloc(slowGroupby.num_key_rows*slowGroupby.num_key_columns * sizeof(int));
    gpu_output_values = (int *)malloc(slowGroupby.num_value_rows*slowGroupby.num_value_columns * sizeof(int));
    groupby_GPU(slowGroupby.key_columns, slowGroupby.num_key_columns,
                slowGroupby.num_key_rows, slowGroupby.value_columns, 
                slowGroupby.num_value_columns, slowGroupby.num_value_rows, 
                slowGroupby.ops, slowGroupby.num_ops,
                gpu_output_keys, gpu_output_values); 
    
    slowGroupby.printGPUResults(gpu_output_keys, gpu_output_values);
    
    // Validating the GPU Result
    // To - do
    //slowGroupby.validGPUResult(output_keys, output_values, output_rows);
    
    return 0;
}

/*
groupby( T* key_columns[], int num_key_columns, int num_key_rows,
        T* value_columns[], int num_value_columns, int num_value_rows,
        reduction_op ops[], int num_ops, T* output_keys[], T* output_values[]) {
    
    
}
*/
