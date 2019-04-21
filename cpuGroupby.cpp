//
//  cpuGroupby.cpp
//  RAPIDS
//
//  Created by Aaron on 11/27/18.
//  Copyright © 2018 Aaron Nightingale. All rights reserved.
//

#include <iostream>
#include "cpuGroupby.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <limits>

std::random_device rd;
std::mt19937 gen(rd());

void cpuGroupby::fillRand(int distinctKeys, int distinctVals) {
  const int maxKeyVal = 65535;
  std::uniform_int_distribution<> keyVals(0, maxKeyVal);
  std::uniform_int_distribution<> keys(0, distinctKeys - 1);
  std::uniform_int_distribution<> vals(0, distinctVals - 1);
  // Key array
  std::vector<std::vector<int>> keyArray;
  int currKey = 0;
  while (currKey < distinctKeys) {
    std::vector<int> random_key(num_key_columns);
    for (auto& i: random_key) {
      i = keyVals(gen);
    }
    auto result = std::find(std::begin(keyArray), std::end(keyArray), random_key);
    if (result == std::end(keyArray)) {
      keyArray.push_back(random_key);
      currKey++;
    }
  }
  for (int cRow=0; cRow<num_key_rows; cRow++) {
    int useKey = keys(gen);
    //Fill the key columns in a row
    for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
      key_columns[keyIdx*num_key_rows + cRow] = keyArray[useKey][keyIdx];
    }
    //Fill the value columns in a row
    for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
      value_columns[valIdx*num_value_rows + cRow] = vals(gen);
    }
  }
}

void cpuGroupby::allocResultArray() {
    output_keys = new int[numGroups*num_key_columns];
    output_values = new int[numGroups*num_value_columns];
}

void cpuGroupby::libsort() {
  std::vector<int> idx(num_key_rows);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), 
	    [=] (const int idx1, const int idx2) {
	      for (int i = 0; i < num_key_columns; ++i) {
		int data1 = key_columns[i * num_key_rows + idx1];
		int data2 = key_columns[i * num_key_rows + idx2];
		if (data1 > data2) return false;
		if (data1 < data2) return true;
	      }
	      return false;
	    });
  std::vector<int> new_row(num_key_rows);
  for (int i = 0; i < num_key_columns; ++i) {
    for (int j = 0; j < num_key_rows; ++j) {
      new_row[j] = key_columns[i * num_key_rows + idx[j]];
    }
    std::copy(new_row.begin(), new_row.end(), key_columns + i * num_key_rows);
  }
  for (int i = 0; i < num_value_columns; ++i) {
    for (int j = 0; j < num_value_rows; ++j) {
      new_row[j] = value_columns[i * num_value_rows + idx[j]];
    }
    std::copy(new_row.begin(), new_row.end(), value_columns + i * num_value_rows);
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
    std::cout << "numGroups: " << numGroups << std::endl;
}

void cpuGroupby::getGroupPtr() {
    // Allocate the group pointer array
    groupPtr = (int*) malloc(sizeof(int)*numGroups);
    
    // Fill it
    for (int i=0; i<numGroups; i++) {
        groupPtr[i] = tempCol[i];
    }
}

std::string opName(reductionType A) {
  if (A == rmin) return "rmin";
  else if (A == rmax) return "rmax";
  else if (A == rmean) return "rmean";
  else if (A == rcount) return "rcount";
  else if (A == rsum) return "rsum";
  else return "None";
}

// Groupby functions
void cpuGroupby::groupby() {
    //max, min, sum, count, and arithmetic mean
    //Init reduction operation list (rmax for testing)
  std::vector<reductionType> allOps{rmin, rmax, rmean, rcount, rsum};
  std::uniform_int_distribution<> dis(0, 4);
  std::cout << "Operations:";
  for (int i=0; i<num_value_columns; i++) {
    ops[i] = allOps[dis(gen)];
    std::cout << opName(ops[i]) << " ";
  }
  std::cout << std::endl;

    //sort();
    //quickSort(key_columns, 0, num_key_rows);
    libsort();
    
    getNumGroups();
    // printData();
    
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
            case rmin:
                rMin(valIdx);
                break;
            case rmean:
                rMean(valIdx);
                break;
            case rcount:
                rCount(valIdx);
                break;
            case rsum:
                rSum(valIdx);
                break;
            default:
                rMax(valIdx);
                break;
        }
    }
}

void cpuGroupby::rMax(int valIdx) {
  int maximum = std::numeric_limits<int>::lowest();
    int tempVal;
    
    for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
        maximum = std::numeric_limits<int>::lowest();
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
    maximum = std::numeric_limits<int>::lowest();
    for (int subIdx=groupPtr[numGroups-1]; subIdx<num_value_rows; subIdx++) {
        tempVal = value_columns[ valIdx*num_value_rows + subIdx ];
        if (tempVal>maximum) {
            maximum = tempVal;
        }
    }
    // Copy values to the output array
    output_values[valIdx*numGroups + numGroups-1] = maximum;
}

void cpuGroupby::rMin(int valIdx) {
    int minimum = std::numeric_limits<int>::max();
    int tempVal;
    
    for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
        minimum = std::numeric_limits<int>::max();
        for (int subIdx=0; subIdx<groupPtr[groupIdx]-groupPtr[groupIdx-1]; subIdx++) {
            tempVal = value_columns[ valIdx*num_value_rows + groupPtr[groupIdx-1]+subIdx ];
            if (tempVal<minimum) {
                minimum = tempVal;
            }
        }
        // Copy values to the output array
        output_values[valIdx*numGroups + groupIdx-1] = minimum;
    }
    
    //Handeling the final group
    minimum = std::numeric_limits<int>::max();
    for (int subIdx=groupPtr[numGroups-1]; subIdx<num_value_rows; subIdx++) {
        tempVal = value_columns[ valIdx*num_value_rows + subIdx ];
        if (tempVal<minimum) {
            minimum = tempVal;
        }
    }
    // Copy values to the output array
    output_values[valIdx*numGroups + numGroups-1] = minimum;
}

void cpuGroupby::rMean(int valIdx) {
    int sum=0;
    int mean=0;
    for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
        sum = 0;
        for (int subIdx=0; subIdx<groupPtr[groupIdx]-groupPtr[groupIdx-1]; subIdx++) {
            sum += value_columns[valIdx*num_value_rows + groupPtr[groupIdx-1]+subIdx ];
        }
        // Copy values to the output array
        mean = sum / (groupPtr[groupIdx]-groupPtr[groupIdx-1]);
        output_values[valIdx*numGroups + groupIdx-1] = mean;
    }
    
    //Handeling the final group
    sum = 0;
    for (int subIdx=groupPtr[numGroups-1]; subIdx<num_value_rows; subIdx++) {
        sum += value_columns[ valIdx*num_value_rows + subIdx ];
    }
    // Copy values to the output array
    mean = sum / (num_value_rows-groupPtr[numGroups-1]);
    output_values[valIdx*numGroups + numGroups-1] = mean;
}

void cpuGroupby::rSum(int valIdx) {
    int sum=0;
    for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
        sum = 0;
        for (int subIdx=0; subIdx<groupPtr[groupIdx]-groupPtr[groupIdx-1]; subIdx++) {
            sum += value_columns[valIdx*num_value_rows + groupPtr[groupIdx-1]+subIdx ];
        }
        // Copy values to the output array
        output_values[valIdx*numGroups + groupIdx-1] = sum;
    }
    
    //Handeling the final group
    sum = 0;
    for (int subIdx=groupPtr[numGroups-1]; subIdx<num_value_rows; subIdx++) {
        sum += value_columns[ valIdx*num_value_rows + subIdx ];
    }
    // Copy values to the output array
    output_values[valIdx*numGroups + numGroups-1] = sum;
}

void cpuGroupby::rCount(int valIdx) {
    int count = 0;
    for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
        count = groupPtr[groupIdx]-groupPtr[groupIdx-1];
        
        // Copy values to the output array
        output_values[valIdx*numGroups + groupIdx-1] = count;
    }
    
    // Handling the final group
    count = num_value_rows-groupPtr[numGroups-1];
    output_values[valIdx*numGroups + numGroups-1] = count;
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
  std::cout << "Printing Data..." << std::endl;
    
  for (int cRow=0; cRow<num_key_rows; cRow++) {
    //print keys for a row
    for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
      if (keyIdx == 0) {
	std::cout << "{";
      }
      std::cout << key_columns[num_key_rows*keyIdx + cRow];
      if(keyIdx != num_key_columns-1) {
	std::cout << ":";
      } else {
	std::cout << "}:";
      }
    }
    // Print values for a row
    for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
      if (valIdx == 0) {
	std::cout << "{";
      }
      std::cout << value_columns[num_value_rows*valIdx + cRow];
      if(valIdx != num_value_columns-1) {
	std::cout << ":";
      } else {
	std::cout << "}";
      }
    }
    std::cout << std::endl;
  }
    
  std::cout << "End Printing Data" << std::endl << std::endl;
}

void cpuGroupby::printResults() {
  std::cout << "Printing Results..." << std::endl;
    
    for (int cRow=0; cRow<numGroups; cRow++) {
        //print keys for a row
        for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
            if (keyIdx == 0) {
	      std::cout << "{";
            }
	    std::cout << output_keys[numGroups*keyIdx + cRow];
            if(keyIdx != num_key_columns-1) {
	      std::cout << ":";
            } else {
	      std::cout << "}:";
            }
        }
        // Print values for a row
        for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
            if (valIdx == 0) {
	      std::cout << "{";
            }
	    std::cout << output_values[numGroups*valIdx + cRow];
            if(valIdx != num_value_columns-1) {
	      std::cout << ":";
            } else {
	      std::cout << "}";
            }
        }
	std::cout << std::endl;
    }
    
    std::cout << "End Printing Results" << std::endl;
}

void cpuGroupby::printGPUResults(int* GPU_output_keys, int* GPU_output_values){
  std::cout << "Printing GPU Results..." << std::endl;
    
    for (int cRow=0; cRow<numGroups; cRow++) {
        //print keys for a row
        for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
            if (keyIdx == 0) {
	      std::cout << "{";
            }
	    std::cout << GPU_output_keys[numGroups*keyIdx + cRow];
            if(keyIdx != num_key_columns-1) {
	      std::cout << ":";
            } else {
	      std::cout << "}:";
            }
        }
        // Print values for a row
        for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
            if (valIdx == 0) {
	      std::cout << "{";
            }
	    std::cout << GPU_output_values[numGroups*valIdx + cRow];
            if(valIdx != num_value_columns-1) {
	      std::cout << ":";
            } else {
	      std::cout << "}";
            }
        }
	std::cout << std::endl;
    }
    
    std::cout << "End GPU Printing Results" << std::endl;
}

bool cpuGroupby::validGPUResult(int* GPUKeys, int* GPUValues, int GPUOutputRows) {
    //ASSUMING THE GPU RESULT IS SORTED
    if (GPUOutputRows != numGroups) {
      std::cout << "FAILED - CPU Rows: " << numGroups << " GPU Rows: " << GPUOutputRows << std::endl;
        return false;
    }
    // cout << "GPU:CPU"<<endl;
    for (int i=0; i<num_value_columns*numGroups; i++) {
        // cout << GPUValues[i] << ":" << output_values[i] << endl;
        if (GPUValues[i] != output_values[i]) {
	  std::cout << "FAILED - CPU data != GPU data " << std::endl;
	  return false;
        }
    }
    std::cout << "PASSED - CPU data == GPU data " << std::endl;   
    return true;
}

// Contructor & Destructor Funcitons
cpuGroupby::~cpuGroupby() {
    //Free the allocated memory
    delete [] key_columns;
    delete [] value_columns;
    delete [] tempCol;
    delete [] ops;
    delete [] output_keys;
    delete [] output_values;
}

cpuGroupby::cpuGroupby(int numKeys, int numValues, int numRows) {
    // Save the arguments
    num_key_columns = numKeys;
    num_key_rows = numRows;
    num_value_columns = numValues;
    num_value_rows = numRows;
    num_ops = numValues;
    
    // Allocate key & value arrays
    key_columns = new int[num_key_columns*num_key_rows];
    value_columns = new int[num_value_columns*num_value_rows];
    tempCol = new int[num_value_rows];
    ops = new reductionType[num_ops];
}



