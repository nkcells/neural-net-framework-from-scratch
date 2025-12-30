
    // matrix of weights
    // matrix of biases

    // apply activation to activations

    // each neuron in a layer has a vector of weights, the layer as a whole is a matrix

    // 
    
    // rows x columns
    //left   matrix  mxr 
    //right  matrix  rxn
    //result matrix  mxn

    // user can speicify num of epochs
    // user can specify num of hidden layers
    // user can specify training data
#include <iostream>
#include <vector>
#include <memory> // for std shared_ptr
#include <algorithm> // for std transform
// #include <algorithm> //for std::transform
#include <functional> //std::multiplies
#include <cmath> // std::exp
#define euler static_cast<double>(std::exp(1)) // is this float or double?
#include "matrix.h"
#include "node.h"

class network{
    
    const std::vector<int>* layers_;
    // Node head(Matrix yo(2,3));
    // Node* nextNode = head.next;
    std::shared_ptr<Node> myNet;
    std::shared_ptr<Matrix> targetValue_;
    std::shared_ptr<Node> lastLayerPtr;
    std::shared_ptr<Node> create_network(){
        
        std::shared_ptr<Matrix> temp_weights = std::make_shared<Matrix>((*layers_).at(1),(*layers_).at(0)); 
        std::shared_ptr<Matrix> temp_biases  = std::make_shared<Matrix>((*layers_).at(1),1);
        auto head = std::make_shared<Node>(temp_weights,temp_biases);
        head->isInputLayer = true;
        
        
        temp_weights = std::make_shared<Matrix>((*layers_).at(2),(*layers_).at(1));
        temp_biases  = std::make_shared<Matrix>((*layers_).at(2),1);
        auto nextLeaf = std::make_shared<Node>(temp_weights,temp_biases);

        head->next_ = nextLeaf;
        nextLeaf->prev_ = head;
        //2x3x2x4
        // nextLeaf->prev_ = std::make_shared<Node>head;
        // head.next_ = nextLeaf;
        for (int i = 3; i < layers_->size(); i++){ // minus one since the last layer doesn't have w/b
            temp_weights = std::make_shared<Matrix>((*layers_).at(i),(*layers_).at(i-1));
            temp_biases = std::make_shared<Matrix>((*layers_).at(i),1);

            nextLeaf->next_ = std::make_shared<Node>(temp_weights,temp_biases);
            nextLeaf->next_->prev_ = nextLeaf; //next leaf's prev will point to node we on now
            nextLeaf = nextLeaf->next_;
            
            std::cout << "a new layer " << i << std::endl;
         
        }

        (*nextLeaf).isOutputlayer = true;
        lastLayerPtr = nextLeaf;

        // head->printNodes();

        return head;
    }
    void generate_gradients(std::shared_ptr<Node> theNet){
        
    }
    public:
        network(const std::vector<int>& layers) : layers_(&layers){
            myNet = create_network();

            targetValue_ = std::make_shared<Matrix>(layers_->at(layers_->size() -1),1); 
            auto t = std::make_shared<Matrix>(layers.at(0),1);


            // // should add argument for num of times so we can have mini batches/ batches
            forwardPass(t);
            // lastLayerPtr->weights_->printMatrix();
            
            outputLayerGrad(*lastLayerPtr);
            // std::cout << "f\nf\nn\n";
            hiddenLayerGrad(*lastLayerPtr->prev_.lock());

            
            // myNet->printDimensions();

        }
        std::shared_ptr<Matrix> multiplyTransposem1(const Matrix& matrix1, const Matrix& matrix2){ //appplies transpose to matrix1 before multiplication
            std::shared_ptr<Matrix> productMatrix;
            if (matrix1.getRows() == matrix2.getRows()){
                productMatrix = std::make_shared<Matrix>(matrix1.getColumns(),matrix2.getColumns(),false);
                //4x2
                for (int which_row = 0; which_row < matrix1.getColumns(); which_row++){ 
                    
                    for (int which_column = 0; which_column < matrix2.getColumns(); which_column++){ 
                        double cell_total = 0;

                        for(int i = 0;i<matrix2.getRows(); i++){ //could also be matrix1.getRows() 
                            cell_total += (matrix1(i,which_row)) * matrix2(i,which_column);
                
                        }
                        (*productMatrix)(which_row, which_column) = cell_total;
                    }
                }

            } else{
                std::cerr << "Error: {multiplyTransposem1}! Trying to multiply a " << matrix1.getRows() << "x" 
                  << matrix1.getColumns() << "*" << matrix2.getRows() << "x" << matrix2.getColumns()
                  << std::endl;
            
            }   

            return productMatrix;
        }
        std::shared_ptr<Matrix> multiplyTransposem2(const Matrix& matrix1, const Matrix& matrix2){ //applies transpose to matrix2 before multiplication
            std::shared_ptr<Matrix> productMatrix;
            if (matrix1.getColumns() == matrix2.getColumns()){
                productMatrix = std::make_shared<Matrix>(matrix1.getRows(),matrix2.getRows(),false);
                
                for (int which_row = 0; which_row < matrix1.getRows(); which_row++){ //4
                    
                    for (int which_column = 0; which_column < matrix2.getRows(); which_column++){ //2
                        double cell_total = 0;
                        //1
                        for(int i = 0;i<matrix2.getColumns(); i++){ //could also be matrix1.getColumns() //1
                            cell_total += (matrix1(which_row,i)) * matrix2(which_column,i);
                            // std::cout << cell_total <<std::endl;

                        }

                        


                        (*productMatrix)(which_row, which_column) = cell_total;
                    }
                }

            } else{
                std::cerr << "Error: {multiplyTransposem2}!\nTrying to multiply a " << matrix1.getRows() << "x" 
                  << matrix1.getColumns() << "*" << matrix2.getColumns() << "x" << matrix2.getRows()
                  << "\nNote the dimension printed above is the transposed dimension of matrix 2" << std::endl;
            
            }   

            return productMatrix;
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
            for (int j=0; j<matrix1.getRows(); j++){
                matrix1(j,0) = 1 / ( 1 + static_cast<double>(std::pow(std::exp(1), -1 * matrix1(j,0) )));
            }
            // std::cout << 1 / ( 1 + static_cast<double>(std::pow((std::exp(1)), (-1 * matrix1(0,j))))<<std::endl;
        }

        double sigmoid(double num){
           
            num = 1 / ( 1 + static_cast<double>(std::pow(std::exp(1), -1 * num )));
            return num;
            // std::cout << 1 / ( 1 + static_cast<double>(std::pow((std::exp(1)), (-1 * matrix1(0,j))))<<std::endl;
        }

        std::shared_ptr<Matrix> sigmoid(const Matrix& matrix1,bool shouldReturn){ //should always be nx1
            auto returnMatrix = std::make_shared<Matrix>(matrix1.getRows(),matrix1.getColumns());
                for (int j=0; j<matrix1.getRows(); j++){
                    (*returnMatrix)(j,0) = 1 / ( 1 + static_cast<double>(std::pow(std::exp(1), -1 * matrix1(j,0) )));
                }

                return returnMatrix;
                // std::cout << 1 / ( 1 + static_cast<double>(std::pow((std::exp(1)), (-1 * matrix1(0,j))))<<std::endl;
        }
        double ssr(Matrix& predicted_matrix, Matrix& target_matrix){
            auto difference = std::make_shared<Matrix>(predicted_matrix.getRows(),predicted_matrix.getColumns());
            double sum{0};
            std::transform(predicted_matrix.data_.begin(),predicted_matrix.data_.end(),target_matrix.data_.begin(),
            difference->data_.begin(), std::minus<double>());

            std::transform((*difference).data_.begin(),(*difference).data_.end(),(*difference).data_.begin(),
            difference->data_.begin(), std::multiplies<double>());
            
            for (int j = 0; j < (*difference).getColumns(); j++){
                (*difference)(0,j) *= 1/2;
                sum += (*difference)(0,j);
                
            }
            

            return sum;
        }
        
        // overloaded function for when we pass in the outputLayer too
        // used so we can ammend the residuals for the output layer 
        void ssr(Matrix& predicted_matrix, Matrix& target_matrix, bool isALayerGiven, std::shared_ptr<Node> givenLayerPtr){
            (givenLayerPtr)->residuals_ = std::make_shared<Matrix>(predicted_matrix.getRows(),predicted_matrix.getColumns());
            double sum{0};
            std::transform(predicted_matrix.data_.begin(),predicted_matrix.data_.end(),target_matrix.data_.begin(),
            (givenLayerPtr)->residuals_->data_.begin(), std::minus<double>());

            auto lossMatrix = std::make_shared<Matrix>(predicted_matrix.getRows(),predicted_matrix.getColumns());

            std::transform((givenLayerPtr)->residuals_->data_.begin(),(givenLayerPtr)->residuals_->data_.end(),(givenLayerPtr)->residuals_->data_.begin(),
            lossMatrix->data_.begin(), std::multiplies<double>());
            
            for (int j = 0; j < (*lossMatrix).getColumns(); j++){
                (*lossMatrix)(0,j) *= 1/2;
                
                sum += (*lossMatrix)(0,j);
                
            }
            givenLayerPtr->totalLoss_ = sum;
        }
        void forwardPass(std::shared_ptr<Matrix> input){
            std::shared_ptr<Matrix> product = input;

            std::shared_ptr<Node> nextOne = myNet;
            nextOne->input_ = std::make_shared<Matrix>(*input);
            while (1){
                std::cout << "Trying to multiply a " << (*(nextOne->weights_)).getRows() << "x" << (*(nextOne->weights_)).getColumns()
                 << "*" << 
                 (*product).getRows() << "x"  << (*product).getColumns()
                  << std::endl;
                
                (nextOne->weights_)->printMatrix();
                (*product).printMatrix();
                
                // std::cout<<"multiply" <<std::endl;

                product = multiply(*(nextOne->weights_),*product);
                // std::cout<<"done"<<std::endl;
                std::cout << "Product after matrix multiplication:" << std::endl;
                (*product).printMatrix();
                std::cout<<std::endl;

                product = add(*product,*(nextOne->biases_));
                std::cout << "After addition: "<<std::endl;
                (*product).printMatrix();
                nextOne->z = std::make_shared<Matrix>(*product);
                std::cout << "{" << (*nextOne->z).getRows() << 'x' << (*nextOne->z).getColumns() << "}" << std::endl;
                // (*product).printMatrix();
                
                // relu(*product);
                nextOne->sigPrimeOutput = sigPrime(*product);
                sigmoid(*product);
                std::cout << "After sigmoid: " << std::endl;

                (*product).printMatrix();

                nextOne->aOut_ = std::make_shared<Matrix>(*product);
                
                // this saves the activation to aOut for a layer
                // (*product).printMatrix();

                if (nextOne->next_ == nullptr) {break;} //go through
                 //whole next one but stop after it

                nextOne = nextOne->next_;
            } 
            std::cout << "Output:  " << std::endl;
            (*product).printMatrix();
            
            std::cout << "\n";
            ssr(*product,*targetValue_,true,nextOne);

            // nextOne->residuals_ = std::make_shared<Matrix>(*product);
            // nextOne->totalLoss_ = residualSum;
            std::cout << "------------done with forward------------------" << std::endl;

        }
        // for last hidden compute gradient
        // general function after that to pass it back

        // for any hidden
        /*
        del of all neurons in the prev layer * sig`(z) * weight connecting them and sum it together
        */
       
        double dotProdRows(const Matrix& matrix1, const Matrix& matrix2, int columnToIterate1=0, int columnToIterate2=0){ //given two matrices it will return the dot prod of a specified column in them
            // if (matrix1.getRows() != 1 || matrix2.getRows() != 1){
            //     std::cerr << "Error: dotProd input matrices #rows != 1" << std::endl;
            // } else 
            
            if (matrix1.getRows() != matrix2.getRows()){
                std::cerr << "Error{dotProdRows}: input matrices do not have same length" << std::endl;
            }
            std::cout<<"oi" <<std::endl;

            double total{0};
            // double first{99};
            // double secon{88};
            for (int j = 0; j < matrix1.getRows(); j++){

                total += matrix1(j,columnToIterate1) * matrix2(j,columnToIterate2);
            }
            std::cout<<"oi" <<std::endl;
            return total;
        }

        double dotProd(const Matrix& matrix1, const Matrix& matrix2, int rowToIterate1=0, int rowToIterate2=0){
            // if (matrix1.getRows() != 1 || matrix2.getRows() != 1){
            //     std::cerr << "Error: dotProd input matrices #rows != 1" << std::endl;
            // } else 
            
            if (matrix1.getColumns() != matrix2.getColumns()){
                std::cerr << "Error: dotProd input matrices do not have same length" << std::endl;
            }
            std::cout<<"oi" <<std::endl;

            double total{0};
            // double first{99};
            // double secon{88};
            for (int j = 0; j < matrix1.getColumns(); j++){

                total += matrix1(rowToIterate1,j) * matrix2(rowToIterate2,j);
            }
            std::cout<<"oi" <<std::endl;
            return total;
       }

       void outputLayerGrad(Node& outputlayer){
        //   (*outputlayer.residuals_).printMatrix();
            std::shared_ptr<Matrix> sigPrimeOutput = ((sigPrime(*outputlayer.z))); //fixx
            // (myMat).printMatrix();
            // std::cout << (*outputlayer.prev_->z).getRows() << "x" << (*outputlayer.prev_->z).getColumns() << std::endl;

            std::cout << (*outputlayer.residuals_).getRows() << "x" << (*outputlayer.residuals_).getColumns() << std::endl;

            std::cout << (*sigPrimeOutput).getRows() << "x" << (*sigPrimeOutput).getColumns() << std::endl;
          
          outputlayer.dels_ = multiplyVector(*outputlayer.residuals_,*sigPrimeOutput);// (a_j - y_j) * sig`(z_j)
          outputlayer.gradients_weights = multiplyTransposem2(*outputlayer.dels_,*outputlayer.prev_.lock()->aOut_);// dL/w_i,j = (del_j) * a_i (transpose a_i)
          

            // std::cout << (*outputlayer.dels_).getRows() << "x" << (*outputlayer.dels_).getColumns() << std::endl;
            // std::cout << (*outputlayer.gradients_weights).getRows() << "x" << (*outputlayer.gradients_weights).getColumns() << std::endl;

            std::cout << "-------------done with outputlayer grad---------------" << std::endl;
       }

       void scaleFirstColumn(Matrix& myVec, double scalar){
        for (int j = 0; j < myVec.getRows(); j++){
            myVec(j,0) = scalar * myVec(j,0);
        }
       }

       void hiddenLayerGrad(Node& hidLayer ){
        double learningRate{0.02};
        double summation{0};
        bool stopNext{false}; 

               
        std::cout << "didn't made here1" << std::endl;

        std::shared_ptr<Node> currHiddenLayer = hidLayer.next_->prev_.lock(); // this is fine but in this case but error prone if we do .next->prev on last layer since it doesn't exist
        std::cout << "didn't make here2" << std::endl;
        std::shared_ptr<Node> nextLayer;
        (*currHiddenLayer).printDimensions();
        std::cout << "ma" << std::endl;

        while (1){ // this prob wont work since we need a_out of input layer
            std::cout << "ahahaasdfa" << std::endl;
            nextLayer = currHiddenLayer->next_; //5x3

            nextLayer->printDimensions();

            /**for (int i=0; i<currHiddenLayer->weights_->getColumns(); i++){
                std::cout << "ding" << std::endl;
                for (int j =0; j<nextLayer->weights_->getColumns(); j++){
                    std::cout << "murphy-start" <<std::endl;
                    (nextLayer->dels_->printDimensionz());
                    (currHiddenLayer->weights_->printDimensionz());
                    summation = dotProd(*nextLayer->dels_, *currHiddenLayer->weights_,0,j); // 0:dels is a nx1
                    std::cout << "murphy-0" <<std::endl;
                    // auto hey = std::make_shared<M
                    // this hasn't been initializized yet.
                    (*currHiddenLayer->gradients_weights)(i,j) = (*nextLayer->dels_)(i,j) * (*currHiddenLayer->aOut_)(0,j) ; // no point in keeping track of this    
                    std::cout << "murphy-1" <<std::endl;

                    (*currHiddenLayer->weights_)(i,j) = (*currHiddenLayer->weights_)(i,j) - (learningRate * (*currHiddenLayer->gradients_weights)(i,j));
                    std::cout << "murphy-2" <<std::endl;

                }
                

                // *currHiddenLayer->gradients_weights = multiplyVector(*next)
                // (*nextLayer)

                //not sur why we index by i
                std::cout << "m" << std::endl;
                double cat = sigPrime((*currHiddenLayer->z)(i,0));
                std::cout << "n" <<std::endl;
                (*(currHiddenLayer->dels_))(0,i) = summation * cat; 
                std::cout << "o" << std::endl;
                std::cout << (*currHiddenLayer->dels_).getRows() << "xx" << (*currHiddenLayer->dels_).getColumns() << std::endl;
            }   **/
            std::cout << "hidden grad time" << std::endl;
            
            auto meh = multiplyTransposem1(*nextLayer->weights_,*nextLayer->dels_);
            meh->printDimensionz();

            currHiddenLayer->dels_ = multiplyVector(*meh,*currHiddenLayer->sigPrimeOutput);
            currHiddenLayer->dels_->printDimensionz();
            if (currHiddenLayer->isInputLayer){
                multiplyTransposem2(*currHiddenLayer->dels_,*currHiddenLayer->input_); //since i don't have an 'input layer' necessarily i cant use curr->prev
            }else{
                (currHiddenLayer->gradients_weights) = multiplyTransposem2(*currHiddenLayer->dels_,*currHiddenLayer->prev_.lock()->aOut_);
            }

            currHiddenLayer->gradients_weights->printDimensionz();


            scaleFirstColumn((*currHiddenLayer->gradients_weights),learningRate);
            currHiddenLayer->gradients_weights->printDimensionz();




            if (currHiddenLayer->weights_->data_.size() != currHiddenLayer->gradients_weights->data_.size()) {
                std::cerr << "SIZE MISMATCH: Weights=" << currHiddenLayer->weights_->data_.size() 
                    << " Grads=" << currHiddenLayer->gradients_weights->data_.size()
                     << std::endl;
                currHiddenLayer->weights_->printDimensionz();
                currHiddenLayer->gradients_weights->printDimensionz();

                exit(1);
            }        
        
            std::transform((currHiddenLayer->weights_->data_).begin(),(currHiddenLayer->weights_->data_).end(),(currHiddenLayer->weights_->data_).begin(),
            (currHiddenLayer->gradients_weights->data_).begin(), std::minus<double>());
            // not sure if it will work
            // std::cout << " four " << std::endl;
            // (currHiddenLayer->dels_->printDimensionz());


            
            // if (currHiddenLayer->prev_ != nullptr){
            //     std::cout << "can keep going" << std::endl;
            //     currHiddenLayer->input_->printDimensionz();
            //     std::cout << "u" << std::endl;

            // }else{
            //     // ****** this doesn't work and i don't know why note to self
            // std::cout << "we have reached the last hidden layer in hiddenLayerGrad" << std::endl;
            // currHiddenLayer->gradients_weights = multiplyVector(*currHiddenLayer->dels_,*currHiddenLayer->input_);
            // std::cout << "did that";
            // // (currHiddenLayer->prev_->weights_->printDimensionz());
            // }
            
            std:: cout << "five" << std::endl;
            // if (stopNext) {break;}
            if (currHiddenLayer->prev_.lock() == nullptr) {
                std::cout << "stop" << std::endl;
                break;
            } else{
            currHiddenLayer = currHiddenLayer->prev_.lock();
            }
        }
        std::cout << "done " << std::endl;
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
                lol.gradients_weights = multiplyVector(*delLast,*(lol.prev_.lock()->aOut_));
                
                // multiplyVector(lol.residuals_,sigPrime()).residcluals_
            
        }
        }
        std::shared_ptr<Matrix> multiplyVector(const Matrix& matrix1, const Matrix& matrix2){ // for an nxn * nxn
            if (matrix1.data_.size() != matrix2.data_.size()){
                std::cerr << "Error: multiplyVector trying to multiply a vector of size " 
                << matrix1.getRows() << "x" << matrix1.getColumns()
                << "by a " << matrix2.getRows() << "x" << matrix2.getColumns() << std::endl;
            }
            auto product = std::make_shared<Matrix>(matrix1.getRows(),matrix1.getColumns());
            std::transform(matrix1.data_.begin(),matrix1.data_.end(),matrix2.data_.begin(),
            product->data_.begin(), std::multiplies<double>());

            return product;
        }
        
        std::shared_ptr<Matrix> elementWiseColumnMultiplication(const Matrix& matrix1, const Matrix& matrix2, int columnToIterate1=0, int columnToIterate2=0){ //given two matrices it will return the product of a specified column in them
            // if (matrix1.getRows() != 1 || matrix2.getRows() != 1){
            //     std::cerr << "Error: dotProd input matrices #rows != 1" << std::endl;
            // } else 
            
            if (matrix1.getRows() != matrix2.getRows()){
                std::cerr << "Error{dotProdRows}: input matrices do not have same length" << std::endl;
            }
            std::cout<<"oi" <<std::endl;
            std::shared_ptr<Matrix> product = std::make_shared<Matrix>(matrix1.getRows(),1);
            // double total{0};
            // double first{99};
            // double secon{88};
            for (int j = 0; j < matrix1.getRows(); j++){

                (*product)(j,0) = matrix1(j,columnToIterate1) * matrix2(j,columnToIterate2);
            }
            std::cout<<"boi" <<std::endl;
            return product;
        }

        std::shared_ptr<Matrix> oneMinus(const Matrix& sigZ1){
            auto oneMinusMatrixPtr = std::make_shared<Matrix>(sigZ1.getRows(),sigZ1.getColumns());
            
            for (int j =0; j < sigZ1.getRows(); j++){
                (*oneMinusMatrixPtr)(0,j) =1 - (sigZ1)(0,j);

            }
            return oneMinusMatrixPtr;
        }
        std::shared_ptr<Matrix> sigPrime(const Matrix& z1){ //z1 should always be nx1
            // auto sigPtr = std::make_shared<Matrix>(z1.getRows(),z1.getColumns());
            std::shared_ptr<Matrix> sigPtr = sigmoid(z1,true);

            Matrix oneMinus(z1.getRows(),z1.getColumns());
            
            for (int j =0; j < z1.getRows(); j++){
                oneMinus(j,0) =1 - (*sigPtr)(j,0);

            }
            std::shared_ptr<Matrix> product = std::make_shared<Matrix>(*multiplyVector(*sigPtr, oneMinus));
            return product;
            
        }

        double sigPrime(double z1){
            // auto sigPtr = std::make_shared<Matrix>(z1.getRows(),z1.getColumns());
            double val = sigmoid(z1) * (1-sigmoid(z1));
            
            return val;
            
        }
};


int main(){
    std::vector<int> myVe = {2,3,5,7,1
    };
    network myNetwork(myVe);
    
    return 0;
}

