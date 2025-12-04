#include "matrix.h"
#include <iostream>

void Matrix::rando(){
    for (int i =0; i< rows_; i++){
        for (int j = 0; j < columns_; j++){
            data_[i * columns_ + j] = i;
        }
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


