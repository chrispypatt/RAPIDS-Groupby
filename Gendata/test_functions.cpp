#include "test_functions.h"

int test_function0()
{

  srand(time(NULL));
  auto a = rand() % 100;
  auto b = rand() % 100;
  auto c = rand() % 100;
  printf("Initialized numbers: %d, %d, %d\n", a, b, c);

  unsigned int n = 20;
  unsigned int m = 4;
  unsigned int j = 100;
  unsigned int size = n*m;

  auto Key = new int[size];
  auto Value = new int[size];
    
    
  for (size_t i = 0; i < size; i++) {
	Key[i] = rand() % 100;
	Value[i] = rand() % 100;
  }

  
MatrixDisplay(Key, n, m);
MatrixDisplay(Value, n, j);
  cout << " random matrix generate completed!" << endl;

  return 0;
}
