
//
//  count.cpp
//  
//
//  Created by menglu liang on 11/19/18.
//

#include "count.hpp"
// arrays as parameters
#include <iostream>
using namespace std;

void printarray (int arg[], int length) {
    for (int n=0; n<length; ++n)
        cout << arg[n] << ' ';
    cout << '\n';
}
