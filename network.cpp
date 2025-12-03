
    // matrix of weights
    // matrix of biases

    // apply activation to activations

    // each neuron in a layer has a vector of weights, the layer as a whole is a matrix

    
    // rows x columns
    //left   matrix  mxr 
    //right  matrix  rxn
    //result matrix  mxn

    // user can speicify num of epochs
    // user can specify num of hidden layers
    // user can specify training data
#include <iostream>
#include <vector>
#include <memory>

// #include <algorithm> //for std::transform
#include <functional> //std::multiplies
#include <cmath> // std::exp
#define euler static_cast<double>(std::exp(1)) // is this float or double?
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

    Node(std::shared_ptr<Matrix>& weights,std::shared_ptr<Matrix>& biases) : weights_(weights), biases_(biases){
    }

    void printNodes(){
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
    
    void printDimensions(){
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



};
class network{
    
    const std::vector<int>* layers_;
    // Node head(Matrix yo(2,3));
    // Node* nextNode = head.next;
    std::shared_ptr<Node> myNet;
    std::shared_ptr<Matrix> targetValue_;
    std::shared_ptr<Node> create_network(){
   
        std::shared_ptr<Matrix> temp_weights = std::make_shared<Matrix>((*layers_).at(0),(*layers_).at(1)); 
        std::shared_ptr<Matrix> temp_biases  = std::make_shared<Matrix>(1,(*layers_).at(1));
        auto head = std::make_shared<Node>(temp_weights,temp_biases);
        
        
        temp_weights = std::make_shared<Matrix>((*layers_).at(1),(*layers_).at(2));
        temp_biases  = std::make_shared<Matrix>(1,(*layers_).at(2));
        auto nextLeaf = std::make_shared<Node>(temp_weights,temp_biases);

        head->next_ = nextLeaf;
        nextLeaf->prev_ = head;
        // nextLeaf->prev_ = std::make_shared<Node>head;
        // head.next_ = nextLeaf;
        for (int i = 3; i < layers_->size()-1; i++){ // minus one since the last layer doesn't have w/b
            temp_weights = std::make_shared<Matrix>((*layers_).at(i-1),(*layers_).at(i));
            temp_biases = std::make_shared<Matrix>(1,(*layers_).at(i+1));

            nextLeaf->next_ = std::make_shared<Node>(temp_weights,temp_biases);
            nextLeaf->next_->prev_ = nextLeaf;
            nextLeaf = nextLeaf->next_;
            
         
        }
        (*nextLeaf).isOutputlayer = true;
 

        // head->printNodes();

        return head;
    }
    void generate_gradients(std::shared_ptr<Node> theNet){
        
    }
    public:
        network(const std::vector<int>& layers) : layers_(&layers){
            myNet = create_network();

            targetValue_ = std::make_shared<Matrix>(1,layers_->at(layers_->size() -1)); 
            auto t = std::make_shared<Matrix>(1,3);

            /*
            input
            0
            0
            0 
            */

           /*
           weights1
           0    0
           1    1
           2    2
           
           */
          /*bias1
          0  0
          */
   
            forwardPass(t);

            // myNet->printDimensions();

        }

        std::shared_ptr<Matrix> multiply(const Matrix& matrix1, const Matrix& matrix2){ //activation, weights
            std::shared_ptr<Matrix> productMatrix;
            if (matrix1.getColumns() == matrix2.getRows()){
                productMatrix = std::make_shared<Matrix>(matrix1.getRows(),matrix2.getColumns(),false);
                
                for (int which_row = 0; which_row < matrix1.getRows(); which_row++){
                    
                    for (int which_column = 0; which_column < matrix2.getColumns(); which_column++){
                        double cell_total = 0;

                        for(int i = 0;i<matrix2.getRows(); i++){ //could also be matrix1.getColumns()
                            cell_total += (matrix1(which_row,i)) * matrix2(i,which_column);
                            // std::cout << cell_total <<std::endl;

                        }

                        


                        (*productMatrix)(which_row, which_column) = cell_total;
                    }
                }

            } else{
                std::cerr << "Error! Trying to multiply a " << matrix1.getRows() << "x" 
                  << matrix1.getColumns() << "*" << matrix2.getRows() << "x" << matrix2.getColumns()
                  << std::endl;
            
            }   

            return productMatrix;
        }
        std::shared_ptr<Matrix> add(const Matrix& matrix1, const Matrix& matrix2){
            auto sum = std::make_shared<Matrix>(matrix1.getRows(),matrix1.getColumns());
            std::transform(matrix1.data_.begin(),matrix1.data_.end(),matrix2.data_.begin(),
            sum->data_.begin(), std::plus<double>());

            return sum;
        }
        void relu(Matrix& matrix1){
            for (int i =0; i < matrix1.getColumns(); i++){
                if (matrix1(0,i) < 0){
                    matrix1(0,i) = 0;
                }
            }
        }
        
        void sigmoid(Matrix& matrix1){
            for (int j=0; j<matrix1.getColumns(); j++){
                matrix1(0,j) = 1 / ( 1 + static_cast<double>(std::pow(std::exp(1), -1 * matrix1(0,j) )));
            }
            // std::cout << 1 / ( 1 + static_cast<double>(std::pow((std::exp(1)), (-1 * matrix1(0,j))))<<std::endl;
        }

        int sigmoid(int num){
           
            num = 1 / ( 1 + static_cast<double>(std::pow(std::exp(1), -1 * num )));
            return num;
            // std::cout << 1 / ( 1 + static_cast<double>(std::pow((std::exp(1)), (-1 * matrix1(0,j))))<<std::endl;
        }

        std::shared_ptr<Matrix> sigmoid(Matrix& matrix1,bool shouldReturn){
        auto returnMatrix = std::make_shared<Matrix>(matrix1.getRows(),matrix1.getColumns());
        for (int j=0; j<matrix1.getColumns(); j++){
            (*returnMatrix)(0,j) = 1 / ( 1 + static_cast<double>(std::pow(std::exp(1), -1 * matrix1(0,j) )));
        }

        return returnMatrix;
        // std::cout << 1 / ( 1 + static_cast<double>(std::pow((std::exp(1)), (-1 * matrix1(0,j))))<<std::endl;
    }
        int ssr(Matrix& predicted_matrix, Matrix& target_matrix){
            auto difference = std::make_shared<Matrix>(predicted_matrix.getRows(),predicted_matrix.getColumns());
            int sum{0};
            std::transform(predicted_matrix.data_.begin(),predicted_matrix.data_.end(),target_matrix.data_.begin(),
            difference->data_.begin(), std::minus<int>());

            std::transform((*difference).data_.begin(),(*difference).data_.end(),(*difference).data_.begin(),
            difference->data_.begin(), std::multiplies<int>());
            
            for (int j = 0; j < (*difference).getColumns(); j++){
                (*difference)(0,j) *= 1/2;
                sum += (*difference)(0,j);
                
            }
            

            return sum;
        }
        
        void forwardPass(std::shared_ptr<Matrix> input){
            std::shared_ptr<Matrix> product = input;

            std::shared_ptr<Node> nextOne = myNet;
            bool stopNext{false}; 
            while (1){
                std::cout << "Trying to multiply a " << (*product).getRows() << "x" 
                  << (*product).getColumns() << "*" << (*(nextOne->weights_)).getRows() << "x" << (*(nextOne->weights_)).getColumns()
                  << std::endl;
                (*product).printMatrix();
                (nextOne->weights_)->printMatrix();
                std::cout<<"multiply" <<std::endl;

                product = multiply(*product,*(nextOne->weights_));
                std::cout<<"done"<<std::endl;

                (*product).printMatrix();
                std::cout<<std::endl;

                product = add(*product,*(nextOne->biases_));
                nextOne->z = std::make_shared<Matrix>(*product);
                (*product).printMatrix();
                
                // relu(*product);
                sigmoid(*product);
                nextOne->aOut_ = std::make_shared<Matrix>(*product);
                // this saves the activation to aOut for a layer
                (*product).printMatrix();

                // nextOne->printNodes();
                if (stopNext) {break;}
                if (nextOne->next_ != nullptr) {stopNext=true;} //go through
                 //whole next one but stop after it

                nextOne = nextOne->next_;
            } 
            std::cout << 101 << std::endl;
            (*product).printMatrix();
            int residualSum = ssr(*product,*targetValue_);

            nextOne->residuals_ = std::make_shared<Matrix>(*product);
            nextOne->totalLoss_ = residualSum;
            std::cout << 201 << std::endl;

        }
        // for last hidden compute gradient
        // general function after that to pass it back

        // for any hidden
        /*
        del of all neurons in the prev layer * sig`(z) * weight connecting them and sum it together
        */

       int dotProd(const Matrix& matrix1, const Matrix& matrix2, int rowToIterate1=0, int rowToIterate2=0){
            // if (matrix1.getRows() != 1 || matrix2.getRows() != 1){
            //     std::cerr << "Error: dotProd input matrices #rows != 1" << std::endl;
            // } else 
            
            if (matrix1.getColumns() != matrix2.getColumns()){
                std::cerr << "Error: dotProd input matrices do not have same length" << std::endl;
            }
            
            int total{0};
            for (int j = 0; j < matrix1.getColumns(); j++){
                total += matrix1(rowToIterate1,j) * matrix2(rowToIterate2,j);
            }
            return total;
       }

       

       void outputLayerGrad(Node& outputlayer){
          outputlayer.dels_ = multiplyVector(*outputlayer.residuals_,*(sigPrime(*outputlayer.z)));// (a_j - y_j) * sig`(z_j)
          outputlayer.gradients_weights = multiplyVector(*outputlayer.dels_,*outputlayer.aOut_);// dL/w_i,j = (del_j) * a_i
       
       }

       void hiddenLayerGrad(Node& hidLayer){
        double learningRate{0.02};
        double summation{0};
        std::shared_ptr<Node> currHiddenLayer = hidLayer.next_->prev_;
        std::shared_ptr<Node> nextLayer;
        while (currHiddenLayer->prev_ != nullptr){ // this prob wont work since we need a_out of input layer
        for (int i=0; i<currHiddenLayer->weights_->getColumns(); i++){
            nextLayer = currHiddenLayer->next_;

            for (int j =0; j<nextLayer->weights_->getColumns(); j++){
                summation = dotProd(*nextLayer->dels_, *currHiddenLayer->weights_,0,j);
                (*currHiddenLayer->gradients_weights)(i,j) = (*nextLayer->dels_)(i,j) * (*currHiddenLayer->aOut_)(0,j) ; // no point in keeping track of this    
                (*currHiddenLayer->weights_)(i,j) = (*currHiddenLayer->weights_)(i,j) - (learningRate * (*currHiddenLayer->gradients_weights)(i,j));

            }
            // (*nextLayer)
            (*currHiddenLayer->dels_)(i,0) = summation * sigPrime((*currHiddenLayer->z)(i,0));
        
        }   
        currHiddenLayer = currHiddenLayer->prev_;
       }
       }


       /*void hiddenGrad(Node& hiddenLayer){
            auto mat = multiplyVector(hiddenLayer.next_->dels_,*(sigPrime(hiddenLayer.z)));
            for (int i = 0; i < (hiddenLayer.weights_)->getColumns(); i++){
                (*hiddenLayer.gradients_weights)(0,i) = dotProd(*mat, //specififc row of     *hiddenLayer.weights_);
            }
            mat = multiplyVector(*mat, *hiddenLayer.weights_);
       }*/
        void computeGrad(Node& lol){
            if (lol.isOutputlayer){
                auto delLast = multiplyVector(*(lol.aOut_), *(oneMinus(*lol.aOut_)));
                delLast = multiplyVector(*(lol.residuals_),*delLast);
                //((a_out - y) * a_out(1-a_out))
                lol.gradients_weights = multiplyVector(*delLast,*(lol.prev_->aOut_));
                
                // multiplyVector(lol.residuals_,sigPrime()).residuals_
            
        }
    }
        std::shared_ptr<Matrix> multiplyVector(const Matrix& matrix1, const Matrix& matrix2){
            auto product = std::make_shared<Matrix>(matrix1.getRows(),matrix1.getColumns());
            std::transform(matrix1.data_.begin(),matrix1.data_.end(),matrix2.data_.begin(),
            product->data_.begin(), std::multiplies<double>());

            return product;
        }
        std::shared_ptr<Matrix> oneMinus(const Matrix& sigZ1){
            auto oneMinusMatrixPtr = std::make_shared<Matrix>(sigZ1.getRows(),sigZ1.getColumns());
            
            for (int j =0; j < sigZ1.getRows(); j++){
                (*oneMinusMatrixPtr)(0,j) =1 - (sigZ1)(0,j);

            }
            return oneMinusMatrixPtr;
        }
        std::shared_ptr<Matrix> sigPrime(Matrix& z1){ //z1 should always be 1xcolumns
            // auto sigPtr = std::make_shared<Matrix>(z1.getRows(),z1.getColumns());
            std::shared_ptr<Matrix> sigPtr = sigmoid(z1,true);

            Matrix oneMinus(z1.getRows(),z1.getColumns());
            
            for (int j =0; j < z1.getRows(); j++){
                oneMinus(0,j) =1 - (*sigPtr)(0,j);

            }
            std::shared_ptr<Matrix> product = std::make_shared<Matrix>(*multiplyVector(*sigPtr, oneMinus));
            return product;
            
        }

        double sigPrime(double z1){ //z1 should always be 1xcolumns
            // auto sigPtr = std::make_shared<Matrix>(z1.getRows(),z1.getColumns());
            double val = sigmoid(z1) * (1-sigmoid(z1));
            
            return val;
            
        }
};


int main(){
    std::vector<int> myVe = {3,2,1};
    network myNetwork(myVe);
    
    return 0;
}

