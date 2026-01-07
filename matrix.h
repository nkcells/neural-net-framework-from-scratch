#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include <vector>
#include <random>



class Matrix{
    int rows_, columns_;
    double mean, variance;

    void rando(double mean, double variance);
    void initToZeroFunc();
    public:
        std::vector<double> data_;
        Matrix(int m, int n, bool initToZero, bool neededAnExtraParameter);
        Matrix(int m, int n, bool randomize = true, double mean = 1, double variance = 0.25);
        Matrix(const Matrix& inputMatrix );
        
        double& operator()(int i, int j);
        const double& operator()(int i, int j) const;


        const int getRows() const {return rows_;}
        const int getColumns() const {return columns_;}

        void printMatrix();
        void printDimensionz();
    
        
};
#endif
