#include "my_matrix.h"

MyMatrix::MyMatrix(size_t x = 5, size_t y = 5){
  if (x <= 0 || y <= 0) {
	cout << "x and y <=0, reset to 5x5" << endl;
	x = 5;
	y = 5;
  }
  n_row = x;
  n_col = y;
  cout << "Constructor initlizing instance to (" << n_row << " X " << n_col
	<< ") matrix" << endl;
  matrix_data = new int[n_row * n_col];
  Randamize();
}

void MyMatrix::MatrixDisplay()
{
  if (matrix_data == NULL) {
	cout << "Error: no matrix passed!" << endl;
  }

  cout << "Matrix elements: " << endl;
  for (int i = 0; i < n_row; i++) {
	for (int j = 0; j < n_col; j++) {
	  cout << matrix_data[i * n_row + j] << " ";
	}
	cout << endl;
  }
}

void MyMatrix::Randamize()
{
  cout << "Randomize fucntion called!" << endl;
  srand(time(NULL));
  for (int i = 0; i < n_row * n_col; i++)
	matrix_data[i] = rand() % 100;
}
