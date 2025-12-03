#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include <vector>

class Matrix{
    int rows_, columns_;

    void rando();
    public:
        std::vector<double> data_;

        Matrix(int m, int n, bool randomize = true);
        Matrix(const Matrix& inputMatrix );
        
        double& operator()(int i, int j);
        const double& operator()(int i, int j) const;


        const int getRows() const {return rows_;}
        const int getColumns() const {return columns_;}

        void printMatrix();
        
};
#endif
