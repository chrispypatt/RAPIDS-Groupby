//
//  max min operation.cpp
//  
//
//  Created by menglu liang on 11/19/18.
//

#include "max min operation.hpp"

#include <bits/stdc++.h>

using namespace std;
/* For a given array arr[], returns the maximum j – i such that
 arr[j] > arr[i] */
int maxIndexDiff(int arr[], int n)
{
    int maxDiff = -1;
    int i, j;
    
    for (i = 0; i < n; ++i)
    {
        for (j = n-1; j > i; --j)
        {
            if(arr[j] > arr[i] && maxDiff < (j - i))
                maxDiff = j - i;
        }
    }
    
    return maxDiff;
}

int minIndexDiff(int arr[], int n)
{
    int minDiff = 1;
    int i, j;
    
    for (i = 0; i < n; ++i)
    {
        for (j = n-1; j > i; --j)
        {
            if(arr[j] < arr[i] && minDiff > (j - i))
                minDiff = j - i;
        }
    }
    
    return minDiff;
}

// CPP for finding minimum operation required
#include<bits/stdc++.h>
using namespace std;

// function for finding array sum
int arraySum (int arr[], int n)
{
    int sum = 0;
    for (int i=0; i<n; sum+=arr[i++]);
    return sum;
}

// function for finding smallest element
int smallest (int arr[], int n)
{
    int small = INT_MAX;
    for (int i=0; i<n; i++)
        if (arr[i] < small)
            small = arr[i];
    return small;
}

// function for finding min operation
int minOp (int arr[], int n)
{
    // find array sum
    int sum = arraySum (arr, n);
    
    // find the smallest element from array
    int small = smallest (arr, n);
    
    // calculate min operation required
    int minOperation = sum - (n * small);
    
    // return result
    return minOperation;
} 
