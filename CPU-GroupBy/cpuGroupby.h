//
//  cpuGroupby.h
//  RAPIDS
//
//  Created by Aaron on 11/27/18.
//  Copyright © 2018 Aaron Nightingale. All rights reserved.
//

#ifndef cpuGroupby_h
#define cpuGroupby_h

#include <iostream>
#include <stdio.h>

#define SIZE 1000   //To Do: make size dynamic

using namespace std;

class cpuGroupby {
public:
    //Variables
    //To Do: update with dynamic key/value columns
    int key_columns[SIZE];
    int num_key_columns;
    int num_key_rows;
    int value_columns[SIZE];
    int num_value_rows;
    int num_ops;
    int* output_keys;           //Assynubg obe value for now
    int* output_values;         //Assuming one value for now
    
    // Aaron's custom data types
    int numGroups;
    
    //Functions
    void fillRand();
    void sort();
    void groupby();
    void getNumGroups();
    void rMax();
    
    void printData();
    void printResults();
    void allocResultArray();
    void freeResults();
};

//To Do:
// class destructor/contructors
// functionality of accepting pointer to data

// Model after this:
/*
 groupby( T* key_columns[], int num_key_columns, int num_key_rows,
 T* value_columns[], int num_value_columns, int num_value_rows,
 reduction_op ops[], int num_ops, T* output_keys[], T* output_values[]) {
 
 
 }
 */

#endif /* cpuGroupby_hpp */
