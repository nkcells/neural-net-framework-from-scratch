#ifndef NODE_H
#define NODE_H
#include <vector>
#include <iostream>
#include <memory> // for std shared_ptr

#include "matrix.h"

class Node{
    const std::vector<int>* layers_;
    public:
    std::shared_ptr<Matrix> input_; //this is specifically for first layer
    std::shared_ptr<Matrix> weights_;
    std::shared_ptr<Matrix> biases_;
    std::shared_ptr<Node> prev_;
    std::shared_ptr<Node> next_;

    std::shared_ptr<Matrix> aOut_; //1xnum of ouputs
    std::shared_ptr<Matrix> sigPrimeOutput; //mx1
    std::shared_ptr<Matrix> residuals_; //1xnum of outputs

    std::shared_ptr<Matrix> gradients_weights; //num of weightsx num of outputs
    std::shared_ptr<Matrix> gradients_biases;
    std::shared_ptr<Matrix> dels_;
    std::shared_ptr<Matrix> z; //wx plus b
    double totalLoss_;

    bool isOutputlayer;
    
    // std::shared_ptr<Matrix> gradients



    Node(std::shared_ptr<Matrix>& weights,std::shared_ptr<Matrix>& biases);
    void printNodes();
    
    void printDimensions();

};
#endif
