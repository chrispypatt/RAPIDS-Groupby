//
//  operation.hpp
//  
//
//  Created by menglu liang on 11/30/18.
//

#ifndef operation_h
#define operation_h
#include <stdio.h>
#include <iostream>
#include <stdio.h>

using namespace std;

class operation {
public:
    // Custom Structures
    enum reductionType {rmin, rmax,rmean,rcount};
    
    
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
    void getNumGroups();
    void doReductionOps();
    void rMax(int valIdx);
    void rMin(int valIdx);
    void rMean(int valIdx);
    void rConut(int valIdx);
    
    
    operation ();
    ~operation();
    
};

#endif /* operation_hpp */
