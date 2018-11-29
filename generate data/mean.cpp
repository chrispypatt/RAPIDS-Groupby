//
//  mean.cpp
//  
//
//  Created by menglu liang on 11/19/18.
//

#include "mean.hpp"
// CPP program to find mean and median of
// an array
#include <bits/stdc++.h>
using namespace std;

// Function for calculating mean
double findMean(int a[], int n)
{
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
    
    return (double)sum/(double)n;
}

// Function for calculating median
double findMedian(int a[], int n)
{
    // First we sort the array
    sort(a, a+n);
    
    // check for even case
    if (n % 2 != 0)
        return (double)a[n/2];
    
    return (double)(a[(n-1)/2] + a[n/2])/2.0;
} 
