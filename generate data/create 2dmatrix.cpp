//
//  create 2dmatrix.cpp
//  
//
//  Created by menglu liang on 11/28/18.
//

#include <iostream>

using namespace std;

int main()
{
    //initialize the matrix
    int** Matrix;                        //A pointer to pointers to an int.
    int rows,columns;
    cout<<"Enter number of rows: ";
    cin>>rows;
    cout<<"Enter number of columns: ";
    cin>>columns;
    Matrix=new int*[rows];                //Matrix is now a pointer to an array of 'rows' pointers.
    
    //define the matrix
    for(int i=0;i<rows;i++)
    {
        Matrix[i]=new int[columns];        //the ith array is initialized
        for(int j=0;j<columns;j++)        //the i,jth element is defined
        {
            cout<<"Enter element in row "<<(i+1)<<" and column "<<(j+1)<<": ";
            cin>>Matrix[i][j];
        }
    }
    
    //Print the matrix
    cout<<"The matrix you have input is:\n";
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<columns;j++)
            cout<<Matrix[i][j]<<"\t";    //tab between each element
        cout<<"\n";                        //new row
    }
    
    //now, we have to free up the memory we took by releasing each vector:
    for(int i=0;i<rows;i++)
        delete[] Matrix[i];   
}

