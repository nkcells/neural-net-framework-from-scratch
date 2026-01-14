#include "node.h"
Node::Node(std::shared_ptr<Matrix>& weights,std::shared_ptr<Matrix>& biases) : weights_(weights), biases_(biases){
    gradients_weights = std::make_shared<Matrix>(weights->getRows(),weights->getColumns()); //ggradient for every weight
    dels_ = std::make_shared<Matrix>(1,weights_->getColumns()); // we should have a del for every neuron
    next_ = nullptr;
    // prev_ = nullptr;
    isOutputlayer = false;
    isInputLayer = false;
    }
void Node::printNodes(){
    // weights_->printMatrix();
    // std::cout << "....." << std::endl;

    std::shared_ptr<Node> nextOne = this->next_->prev_.lock(); //starting with head
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
    // std::shared_ptr<Node> nextOne = &this; //starting with head error prone if we do for last layer
    // while (nextOne != nullptr){ // iterates through the whole network this is bad should only print dimenson for self
        
        std::cout << "weights: " << this->weights_->getRows() << "x" <<
        this->weights_->getColumns()
        << " + " <<
        "biases: " << this->biases_->getRows() << "x" <<
        this->biases_->getColumns()
        << std::endl;
        
        // nextOne = nextOne->next_;
    // }
}




