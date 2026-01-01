#include "matrix.h"
#include <iostream>
#include <random>

// 1. Set up the engine (Mersenne Twister is standard)
std::random_device rd;
std::mt19937 gen(rd());

void Matrix::rando(){
    for (int i =0; i< rows_; i++){
        for (int j = 0; j < columns_; j++){
            std::normal_distribution<double> dist(1.0, 1.0);
            data_[i * columns_ + j] = dist(gen) * 0.01;
        }
    }
}

void Matrix::initToZeroFunc(){
    for (int i =0; i< rows_; i++){
        for (int j = 0; j < columns_; j++){
            data_[i * columns_ + j] = 0;
        }
    }
}
Matrix::Matrix(int m, int n, bool initToZero, bool extraUselessParameter) : data_(m*n), rows_(m), columns_(n){
    if (initToZero){
        initToZeroFunc();
    }
}
Matrix::Matrix(int m, int n, bool randomize) : data_(m*n), rows_(m), columns_(n){
    if (randomize){
        rando();
    }
}
Matrix::Matrix(const Matrix& inputMatrix ){ // copy constructor
    data_ = inputMatrix.data_;
    rows_ = inputMatrix.rows_;
    columns_ = inputMatrix.columns_;

}
double& Matrix::Matrix::operator()(int i, int j){
    return data_[i * columns_ + j]; //does not have bounds checking
}

const double& Matrix::operator()(int i, int j) const{
    return  data_[i * columns_ + j]; //does not have bounds checking
}

void Matrix::printMatrix(){

    for (int i =0; i< rows_; i++){
        for (int j = 0; j < columns_; j++){
            std::cout << data_[i * columns_ + j] << ',';
        }

        std::cout << std::endl;
    } 
    
    std::cout << std::string(2*columns_,'-') << std::endl;
}

void Matrix::printDimensionz(){
    std::cout << rows_ << 'x' << columns_ << std::endl;
}


