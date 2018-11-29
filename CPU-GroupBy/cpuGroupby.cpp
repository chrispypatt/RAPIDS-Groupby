//
//  cpuGroupby.cpp
//  RAPIDS
//
//  Created by Aaron on 11/27/18.
//  Copyright © 2018 Aaron Nightingale. All rights reserved.
//

#include <iostream>
#include "cpuGroupby.h"
#include <time.h>

// Memory Allocation (put all in one function)
void cpuGroupby::allocKeys() {
    key_columns = (int *)malloc(sizeof(int)*num_key_columns*num_key_rows);
}

void cpuGroupby::allocValues() {
    value_columns = (int *)malloc(sizeof(int)*num_value_columns*num_value_rows);
    tempCol = (int *)malloc(sizeof(int)*num_value_rows);
}

void cpuGroupby::fillRandX() {
    srand((unsigned int)time(NULL));
    for (int cRow=0; cRow<num_key_rows; cRow++) {
        //Fill the key columns in a row
        for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
            key_columns[keyIdx*num_key_rows + cRow] = rand() % (3+1);
        }
        //Fill the value columns in a row
        for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
            value_columns[valIdx*num_value_rows + cRow] = rand() % (100+1);
        }
    }
}

void cpuGroupby::allocResultArray() {
    output_keys = (int *)malloc(sizeof(int)*numGroups*num_key_columns);
    output_values = (int *)malloc(sizeof(int)*numGroups*num_value_columns);
}

void cpuGroupby::freeResults() {
    free(output_keys);
    free(output_values);
}

// Sorting Functions
void cpuGroupby::sort() {
    for (int i = 0; i < num_key_rows-1; i++) {
        // Last i elements are already in place
        for (int j = 0; j < num_key_rows-i-1; j++) {
            if (!nextKeyBigger(j)) {
                // Swap j and j+1
                swapAtRow(j);
            }
        }
    }
}

bool cpuGroupby::nextKeyBigger(int cRow) {
    int keyIdx=0;
    for (keyIdx=0; key_columns[keyIdx*num_key_rows + cRow] == key_columns[keyIdx*num_key_rows + cRow+1] && keyIdx<num_key_columns; keyIdx++);
    //see if key at cRow is greater than key at cRow+1
    if (key_columns[keyIdx*num_key_rows + cRow] > key_columns[keyIdx*num_key_rows + cRow+1]) {
        return false;
    } else {
        return true;
    }
}

void cpuGroupby::swapAtRow(int cRow) {
    int tempVal;
    //Swap the keys
    for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
        tempVal = key_columns[keyIdx*num_key_rows + cRow];
        key_columns[keyIdx*num_key_rows + cRow] = key_columns[keyIdx*num_key_rows + cRow+1];
        key_columns[keyIdx*num_key_rows + cRow+1] = tempVal;
    }
    //Swap the values
    for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
        tempVal = value_columns[valIdx*num_value_rows + cRow];
        value_columns[valIdx*num_value_rows + cRow]=value_columns[valIdx*num_value_rows + cRow+1];
        value_columns[valIdx*num_value_rows + cRow+1] = tempVal;
    }
}

void cpuGroupby::getNumGroups() {
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

void cpuGroupby::getGroupPtr() {
    // Allocate the group pointer array
    groupPtr = (int*) malloc(sizeof(int)*numGroups);
    
    // Fill it
    for (int i=0; i<numGroups; i++) {
        groupPtr[i] = tempCol[i];
    }
    
    // To Do: Get rid of the tempCol Array?
    //free(tempCol);
}

// Groupby functions
void cpuGroupby::groupby() {
    //max, min, sum, count, and arithmetic mean
    // Just do the max for testing now.
    // Setup / test parameters
    num_key_rows = 100;     // These are always equal
    num_value_rows = 100;
    num_key_columns = 3;
    num_value_columns = 3;
    
    //Init the list of reduction operations
    num_ops = 3;    //This is always the same as num_value_columns
    ops = (reductionType*)malloc(sizeof(reductionType)*num_ops);
    ops[0] = rmax;
    ops[1] = rmax;
    ops[2] = rmax;
    
    allocKeys();
    allocValues();
    fillRandX();
    sort();
    getNumGroups();
    printDataX();
    
    allocResultArray();
    getGroupPtr();
    writeOutputKeys();
    doReductionOps();
    printResults();
}

void cpuGroupby::doReductionOps() {
    for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
        switch (ops[valIdx]) {
            case rmax:
                rMax(valIdx);
                break;
                
            default:
                rMax(valIdx);
                break;
        }
    }
}

void cpuGroupby::rMax(int valIdx) {
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
    
    //Handeling the final group
    maximum = -999999999;
    for (int subIdx=groupPtr[numGroups-1]; subIdx<num_value_rows; subIdx++) {
        tempVal = value_columns[ valIdx*num_value_rows + subIdx ];
        if (tempVal>maximum) {
            maximum = tempVal;
        }
    }
    // Copy values to the output array
    output_values[valIdx*numGroups + numGroups-1] = maximum;
}

void cpuGroupby::writeOutputKeys() {
    // Copy each unique key to the output.
    int rowIdx;
    for (int groupIdx=0; groupIdx<numGroups; groupIdx++) {
        rowIdx = groupPtr[groupIdx];
        for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
            output_keys[keyIdx*numGroups + groupIdx] = key_columns[keyIdx*num_key_rows + rowIdx];
        }
    }
}


// Debug / Printing Functions

void cpuGroupby::printData() {
    cout << "Printing Data..." << endl;
    for (int i=0; i<SIZE; i++) {
        cout << key_columns[i] << ":" << value_columns[i] << endl;
    }
    cout << "End Printing Data" << endl;
}

void cpuGroupby::printDataX() {
    cout << "Printing Data..." << endl;
    
    //To Do: is num_key_rows always the same as num_value_rows?
    for (int cRow=0; cRow<num_key_rows; cRow++) {
        //print keys for a row
        for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
            if (keyIdx == 0) {
                cout << "{";
            }
            cout << key_columns[num_key_rows*keyIdx + cRow];
            if(keyIdx != num_key_columns-1) {
                cout << ":";
            } else {
                cout << "}:";
            }
        }
        //To Do: print values for a row
        for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
            if (valIdx == 0) {
                cout << "{";
            }
            cout << value_columns[num_value_rows*valIdx + cRow];
            if(valIdx != num_value_columns-1) {
                cout << ":";
            } else {
                cout << "}";
            }
        }
        cout << endl;
    }
    
    cout << "End Printing Data" << endl;
}

void cpuGroupby::printResults() {
    cout << "Printing Results..." << endl;
    
    //To Do: is num_key_rows always the same as num_value_rows?
    for (int cRow=0; cRow<numGroups; cRow++) {
        //print keys for a row
        for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
            if (keyIdx == 0) {
                cout << "{";
            }
            cout << output_keys[numGroups*keyIdx + cRow];
            if(keyIdx != num_key_columns-1) {
                cout << ":";
            } else {
                cout << "}:";
            }
        }
        //To Do: print values for a row
        for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
            if (valIdx == 0) {
                cout << "{";
            }
            cout << output_values[numGroups*valIdx + cRow];
            if(valIdx != num_value_columns-1) {
                cout << ":";
            } else {
                cout << "}";
            }
        }
        cout << endl;
    }
    
    cout << "End Printing Data" << endl;
}


// Contructor & Destructor Funcitons
cpuGroupby::~cpuGroupby() {
    /*
    //Free the allocated memory
    for (int i=0; i<num_key_columns; i++) {
        free( *(key_columns + i) );
    }
    for (int i=0; i<num_value_columns; i++) {
        //To Do - fix this bug
        //free( *(value_columns + i) );
    }
    */
    free(key_columns);
    free(value_columns);
    
}

cpuGroupby::cpuGroupby() {
    //not sure what to put in here...
}

