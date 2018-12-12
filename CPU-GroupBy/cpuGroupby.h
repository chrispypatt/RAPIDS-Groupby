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

enum reductionType { max, min, sum, count, mean };

class cpuGroupby {
public:
    // Custom Structures
    enum reductionType {rmin, rmax};    //To Do: use this and add other types
    
    //Variables
    int num_ops;        //unused at the moment - use to create array of "reductionType"
    int* output_keys;
    int* output_values;
    reductionType* ops;
    
    // ARBITRARY VERSION
    int* key_columns;
    int* value_columns;
    int num_key_columns;
    int num_value_columns;
    int num_key_rows;
    int num_value_rows;     //always the same as above...
    
    // Aaron's custom data types
    int numGroups;
    int* tempCol;   //Used for temporary storage of groupPtrs
    int* groupPtr;  //array of indicies of start of each group
    
    //Functions
    void fillRand();
    void sort();
    void groupby();
    void getNumGroups();
    void doReductionOps();
    void rMax(int valIdx);
    
    void printData();
    void printResults();
    void allocResultArray();
    void freeResults();
    
    // ARBITRARY FUNCTIONS
    void allocKeys();
    void allocValues();
    void printDataX();
    void fillRandX();
    bool nextKeyBigger(int cRow);
    void swapAtRow(int cRow);
    void getGroupPtr();
    void writeOutputKeys();
    
    //Constructor / destructor functions
    cpuGroupby();
    ~cpuGroupby();
    
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
