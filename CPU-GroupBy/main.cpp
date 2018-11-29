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

using namespace std;
int main(int argc, const char * argv[]) {
    // insert code here...
    cout << "Hello, World!\n";
    
    cpuGroupby slowGroupby;
    slowGroupby.groupby();
    //slowGroupby.printResults();
    
    return 0;
}

/*
groupby( T* key_columns[], int num_key_columns, int num_key_rows,
        T* value_columns[], int num_value_columns, int num_value_rows,
        reduction_op ops[], int num_ops, T* output_keys[], T* output_values[]) {
    
    
}
*/
