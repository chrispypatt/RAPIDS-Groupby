//
//  operation.cpp
//  
//
//  Created by menglu liang on 11/30/18.
//
#include <iostream>
#include "cpuGroupby.h"
#include <time.h>
#include "operation.h"
#include <bits/stdc++.h>
#include "my_matrix.h"

int num_key_rows=n;
int num_key_columns=m;
int num_value_rows=n;
int num_value_columns=j;

//taken from Aaron's code
void getNumGroups() {
    // Start with one group, each boundry is a "split" adding another group.
    numGroups = 1;
    // Store the start row of each group here
    tempCol[0] = 0;
    for (int cRow=1; cRow<num_key_rows; cRow++) {
        //Loop through all key columns
        for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
            if (key_columns[keyIdx*num_key_rows+cRow] != key_columns[keyIdx*num_key_rows+cRow-1]) {
                // New Group found
                tempCol[numGroups] = cRow;
                numGroups++;
                keyIdx = num_key_columns+1; //break the for loop
            }
        }
    }
    cout << "numGroups: " << numGroups << endl;
}

void doReductionOps() {
    for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
        switch (ops[valIdx]) {
            case rmax:
                rMax(valIdx);
                break;
            case rmin:
                rMin(valIdx);
                break;
            case rmean:
                rMean(valIdx);
                break;
            case rcount:
                rCount(valIdx);
                break;
            default:
                rMax(valIdx);
                break;
        }
    }
}

void rMax(int valIdx) {
    int maximum = -999999999;
    int tempVal;
    
    for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
        maximum = -999999999;
        for (int subIdx=0; subIdx<groupPtr[groupIdx]-groupPtr[groupIdx-1]; subIdx++) {
            tempVal = value_columns[ valIdx*num_value_rows + groupPtr[groupIdx-1]+subIdx ];
            if (tempVal>maximum) {
                maximum = tempVal;
            }
        }
        // Copy values to the output array
        output_values[valIdx*numGroups + groupIdx-1] = maximum;
    }
}

    void rMin(int valIdx) {
        int minimum = 999999999;
        int tempVal;
        
        for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
            minimum = 999999999;
            for (int subIdx=0; subIdx<groupPtr[groupIdx]-groupPtr[groupIdx-1]; subIdx++) {
                tempVal = value_columns[ valIdx*num_value_rows + groupPtr[groupIdx-1]+subIdx ];
                if (tempVal<minimum) {
                    minimum = tempVal;
                }
            }
            // Copy values to the output array
            output_values[valIdx*numGroups + groupIdx-1] = minimum;
        }
    }
    void rMean(int valIdx) {
        float sum=0;
        float mean=0;
            for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
                sum = 0;
                for (int subIdx=0; subIdx<groupPtr[groupIdx]-groupPtr[groupIdx-1]; subIdx++) {
                    sum = 0;
                    sum += value_columns[valIdx*num_value_rows + groupPtr[groupIdx-1]+subIdx ];
                    mean=sum/groupPtr[groupIdx]-groupPtr[groupIdx-1];
                }
                // Copy values to the output array
                output_values[valIdx*numGroups + groupIdx-1] = mean;
            }
    }
void rCount(int valIdx) {
            int count = 0;
            for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
                count = 0;
                for (int subIdx=0; subIdx<groupPtr[groupIdx]-groupPtr[groupIdx-1]; subIdx++) {
                    count =groupPtr[groupIdx]-groupPtr[groupIdx-1];
                }
                // Copy values to the output array
                output_values[valIdx*numGroups + groupIdx-1] = count;
            }
}
