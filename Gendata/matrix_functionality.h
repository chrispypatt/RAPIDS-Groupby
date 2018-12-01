#ifndef MATRIX_FUNCTIONALITY_H
#define MATRIX_FUNCTIONALITY_H

#include <iostream>
#include <new>
#include <time.h>

using namespace std;

template <typename T>
int MatrixDisplay(T *mtrx_ptr, size_t size_x, size_t size_y)
{
  if (mtrx_ptr == NULL) {
  cout << "Error: no matrix passed!" << endl;
  return -1;
  }
  
  cout << "Matrix elements: " << endl;
  for (size_t i = 0; i < size_x; i++) {
	for (size_t j = 0; j < size_y; j++) {
	  cout << mtrx_ptr[i * size_y + j] << " ";
	}
	cout << endl;
  }

  return 0;
}


#endif // !MATRIX_FUNCTIONALITY_H


