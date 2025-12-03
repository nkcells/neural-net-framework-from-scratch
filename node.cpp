#include "node.h"
Node::Node(std::shared_ptr<Matrix>& weights,std::shared_ptr<Matrix>& biases) : weights_(weights), biases_(biases){
    }
void Node::printNodes(){
    // weights_->printMatrix();
    // std::cout << "....." << std::endl;

    std::shared_ptr<Node> nextOne = this->next_->prev_; //starting with head
    // std::shar nextOne& = this;
    while (nextOne != nullptr){
        nextOne->weights_->printMatrix();
        std::cout << "~" << std::endl;
        nextOne->biases_->printMatrix();

        std::cout << "....." << std::endl;
        nextOne = nextOne->next_;
    }
}

void Node::printDimensions(){
    std::shared_ptr<Node> nextOne = this->next_->prev_; //starting with head
    while (nextOne != nullptr){
        
        std::cout << nextOne->weights_->getRows() << "x" <<
        nextOne->weights_->getColumns()
        << "+" <<
        nextOne->biases_->getRows() << "x" <<
        nextOne->biases_->getColumns()
        << std::endl;
        
        nextOne = nextOne->next_;
    }
}




