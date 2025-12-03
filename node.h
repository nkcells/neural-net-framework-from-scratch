#ifndef NODE_H
#define NODE_H
#include <vector>
#include <iostream>

#include "matrix.h"

class Node{
    const std::vector<int>* layers_;
    public:
    std::shared_ptr<Matrix> weights_;
    std::shared_ptr<Matrix> biases_;
    std::shared_ptr<Node> prev_{nullptr};
    std::shared_ptr<Node> next_{nullptr};

    std::shared_ptr<Matrix> aOut_; //1xnum of ouputs
    std::shared_ptr<Matrix> residuals_; //1xnum of outputs

    std::shared_ptr<Matrix> gradients_weights; //num of weightsx num of outputs
    
    std::shared_ptr<Matrix> dels_;
    std::shared_ptr<Matrix> z; //wx plus b
    double totalLoss_;

    bool isOutputlayer{false};
    
    // std::shared_ptr<Matrix> gradients

    Node(std::shared_ptr<Matrix>& weights,std::shared_ptr<Matrix>& biases);
    void printNodes();
    
    void printDimensions();

};
#endif
