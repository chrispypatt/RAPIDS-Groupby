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

// Memory Allocation
void cpuGroupby::fillRand() {
    srand(time(NULL));
    for (int i=0; i<SIZE; i++) {
        key_columns[i] = rand() % (100+1);
        value_columns[i] = rand() % (100+1);
    }
}

void cpuGroupby::allocResultArray() {
    output_keys = (int *)malloc(sizeof(int)*numGroups);
    output_values = (int *)malloc(sizeof(int)*numGroups);
}

void cpuGroupby::freeResults() {
    free(output_keys);
    free(output_values);
}

// Sorting Functions
void cpuGroupby::sort() {
    for (int i = 0; i < SIZE-1; i++) {
        // Last i elements are already in place
        for (int j = 0; j < SIZE-i-1; j++) {
            if (key_columns[j] > key_columns[j+1]) {
                // Swao j and j+1
                int temp = key_columns[j];
                key_columns[j] = key_columns[j+1];
                key_columns[j+1] = temp;
            }
        }
    }
}

void cpuGroupby::getNumGroups() {
    // Start with one group, each boundry is a "split" adding another group.
    numGroups = 1;
    for (int i=1; i<SIZE; i++) {
        if (key_columns[i-1] != key_columns[i]) {
            numGroups++;
        }
    }
    cout << "numGroups: " << numGroups << endl;
}

// Groupby functions
void cpuGroupby::groupby() {
    //max, min, sum, count, and arithmetic mean
    // Just do the max for testing now.
    fillRand();
    sort();
    printData();
    getNumGroups();
    allocResultArray();
    rMax();
    printResults();
    freeResults();
}

void cpuGroupby::rMax() {
    int maximum = -999999999;
    int groupIdx = 0;
    
    int firstKey = key_columns[0];
    for (int inputIdx = 0; inputIdx < SIZE; inputIdx++) {
        if (firstKey != key_columns[inputIdx]) {
            //Start of a new group
            output_values[groupIdx] = maximum;
            output_keys[groupIdx] = firstKey;
            groupIdx++;
            
            firstKey = key_columns[inputIdx];
            maximum = -999999999;
        } else {
            if (value_columns[inputIdx] > maximum) {
                maximum = value_columns[inputIdx];
            }
        }
    }
    //Handling the last group
    output_values[groupIdx] = maximum;
    output_keys[groupIdx] = firstKey;
}


// Debug / Printing Functions

void cpuGroupby::printData() {
    cout << "Printing Data..." << endl;
    for (int i=0; i<SIZE; i++) {
        cout << key_columns[i] << ":" << value_columns[i] << endl;
    }
    cout << "End Printing Data" << endl;
}

void cpuGroupby::printResults() {
    cout << "Printing Results..." << endl;
    for (int i=0; i<numGroups; i++) {
        cout << output_keys[i] << ":" << output_values[i] << endl;
    }
    cout << "End Printing Data" << endl;
}



