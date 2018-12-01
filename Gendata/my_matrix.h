#ifndef MY_MATRIX_H
#define MY_MATRIX_H

#include <iostream>
#include <new>
#include <time.h>


using namespace std;

class MyMatrix
{
  public:
	MyMatrix(size_t x, size_t y);

	~MyMatrix() {
	  cout << "Destructor function called" << endl;
	  delete[] matrix_data;
	}

	void MatrixDisplay();

  private:
	size_t n_row;
	size_t n_col;
	int *matrix_data;

	void Randamize();
};

#endif // !MY_MATRIX_H
