//
//  vector sum.cpp
//  
//
//  Created by menglu liang on 11/19/18.
//

#include "vector sum.hpp"
#include <iostream>
#include <vector>
#include <numeric>
using namespace std;

// User defined function that returns sum of
// arr[] using accumulate() library function.
int arraySum(vector<int> &v)
{
    int initial_sum  = 0;
    return accumulate(v.begin(), v.end(), initial_sum);
} 
