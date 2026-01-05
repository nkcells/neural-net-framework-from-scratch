
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
#include <random>
std::random_device rdn;
std::mt19937 genn(rdn());

class network{
    
    const std::vector<int>* layers_;
    // Node head(Matrix yo(2,3));
    // Node* nextNode = head.next;
    std::shared_ptr<Node> myNet; 
    std::shared_ptr<Matrix> targetValue_;
    std::shared_ptr<Node> lastLayerPtr;

    void (network::*activationFuncPtr)(Matrix&) = &network::relu;
    std::shared_ptr<Matrix> (network::*activationDerivativeFuncPtr)(const Matrix&) = &network::reluPrime;



    double learningRate{.01};


    std::shared_ptr<Node> create_network(){

        std::shared_ptr<Matrix> temp_weights = std::make_shared<Matrix>((*layers_).at(1),(*layers_).at(0),true); 
        std::shared_ptr<Matrix> temp_biases  = std::make_shared<Matrix>((*layers_).at(1),1,true,true);
        
        auto head = std::make_shared<Node>(temp_weights,temp_biases);
        head->isInputLayer = true;
        
        
        temp_weights = std::make_shared<Matrix>((*layers_).at(2),(*layers_).at(1));
        temp_biases  = std::make_shared<Matrix>((*layers_).at(2),1,true,true);
        auto nextLeaf = std::make_shared<Node>(temp_weights,temp_biases);

        head->next_ = nextLeaf;
        nextLeaf->prev_ = head;
        //2x3x2x4
        // nextLeaf->prev_ = std::make_shared<Node>head;
        // head.next_ = nextLeaf;
        for (int i = 3; i < layers_->size(); i++){ // minus one since the last layer doesn't have w/b
            temp_weights = std::make_shared<Matrix>((*layers_).at(i),(*layers_).at(i-1));
            temp_biases = std::make_shared<Matrix>((*layers_).at(i),1,true,true);

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
            
            std::normal_distribution<double> dist(50, 10); //average 0, variance .5

            // activationFuncPtr = &maf;{}

            for (int i = 1; i < 1000; i+=1){
                double randoDouble = dist(genn);
                // std::cout << randoDouble << std::endl;
                t->data_[0] = randoDouble;
                targetValue_->data_[0] = 0.5 * randoDouble;
               
                // // should add argument for num of times so we can have mini batches/ batches
                forwardPass(t);
                // lastLayerPtr->weights_->printMatrix();
                
                outputLayerGrad(*lastLayerPtr);
                // std::cout << "f\nf\nn\n";
                hiddenLayerGrad(*lastLayerPtr->prev_.lock());
            }
            
            t->data_[0] = 45;
            forwardPass(t);

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
            std::transform(matrix1.data_.begin(),matrix1.data_.end(),matrix2.data_.begin(), // good
            sum->data_.begin(), std::plus<double>());

            return sum;
        }
        void relu(Matrix& matrix1){
            for (int i =0; i < matrix1.getRows(); i++){
                if (matrix1(i,0) <= 0){
                    matrix1(i,0) = 0;
                }
            }
        }

        std::shared_ptr<Matrix> reluPrime(const Matrix& matrix1){
            std::shared_ptr<Matrix> derivativeRelu = std::make_shared<Matrix>(matrix1.getRows(),matrix1.getColumns(),true,true);
            //Matrix::Matrix(int m, int n, bool initToZero, bool extraUselessParameter)
            for (int i =0; i< matrix1.getRows(); i++){
                if (matrix1(i,0) > 0){
                    (*derivativeRelu)(i,0) = 1;
                }
                //since derivativeRelu was auto initialized to zero dont need an else
            }
            
            return derivativeRelu;
        }
        
        void sigmoid(Matrix& matrix1){
            for (int j=0; j<matrix1.getRows(); j++){
                matrix1(j,0) = 1 / ( 1 + static_cast<double>(std::pow(std::exp(1), -1 * matrix1(j,0) )));
            }
            // std::cout << 1 / ( 1 + static_cast<double>(std::pow((std::exp(1)), (-1 * matrix1(0,j))))<<std::endl;
        }


  

    

        /*std::shared_ptr<Matrix> sigmoid(const Matrix& matrix1,bool shouldReturn){ //should always be nx1
            auto returnMatrix = std::make_shared<Matrix>(matrix1.getRows(),matrix1.getColumns());
                for (int j=0; j<matrix1.getRows(); j++){
                    (*returnMatrix)(j,0) = 1 / ( 1 + static_cast<double>(std::pow(std::exp(1), -1 * matrix1(j,0) )));
                }

                return returnMatrix;
                // std::cout << 1 / ( 1 + static_cast<double>(std::pow((std::exp(1)), (-1 * matrix1(0,j))))<<std::endl;
        }*/
        double ssr(Matrix& predicted_matrix, Matrix& target_matrix){
            auto difference = std::make_shared<Matrix>(predicted_matrix.getRows(),predicted_matrix.getColumns());
            double sum{0};
            std::transform(predicted_matrix.data_.begin(),predicted_matrix.data_.end(),target_matrix.data_.begin(),
            difference->data_.begin(), std::minus<double>()); // good

            std::transform((*difference).data_.begin(),(*difference).data_.end(),(*difference).data_.begin(),
            difference->data_.begin(), std::multiplies<double>()); // good
            
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
            (givenLayerPtr)->residuals_->data_.begin(), std::minus<double>()); // good

            auto lossMatrix = std::make_shared<Matrix>(predicted_matrix.getRows(),predicted_matrix.getColumns());

            std::transform((givenLayerPtr)->residuals_->data_.begin(),(givenLayerPtr)->residuals_->data_.end(),(givenLayerPtr)->residuals_->data_.begin(),
            lossMatrix->data_.begin(), std::multiplies<double>()); //good
            
            for (int j = 0; j < (*lossMatrix).getRows(); j++){ // **** this may be wrong and should actually be by column
                (*lossMatrix)(j,0) *= 1/2;
                
                sum += (*lossMatrix)(j,0);
                
            }
            givenLayerPtr->totalLoss_ = sum;
        }
        void forwardPass(std::shared_ptr<Matrix> input){
            std::shared_ptr<Matrix> product = input;

            std::shared_ptr<Node> nextOne = myNet;
            nextOne->input_ = std::make_shared<Matrix>(*input);
            while (1){
                std::cout << "<-------- Forward pass time -------->" << std::endl;

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
                (this->*activationFuncPtr)(*product);
                nextOne->activationOut = (this->*activationDerivativeFuncPtr)(*product);
                


                //to skip activation on output layer
                /*if (nextOne->isOutputlayer == false){
                    sigmoid(*product);
                    std::cout << "After sigmoid: " << std::endl;
                    (*product).printMatrix();
                } else {
                    std::cout << "Simoid skipped {output layer}" << std::endl;
                }*/
                
                

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
             std::cout << "--------OUTPUTlayerGrad begin-------" << std::endl;
        //   (*outputlayer.residuals_).printMatrix();
            // std::shared_ptr<Matrix> activationOut = ((sigPrime(*outputlayer.z))); //fixx
            // (myMat).printMatrix();
            // std::cout << (*outputlayer.prev_->z).getRows() << "x" << (*outputlayer.prev_->z).getColumns() << std::endl;

            // std::cout << (*outputlayer.residuals_).getRows() << "x" << (*outputlayer.residuals_).getColumns() << std::endl;

            // std::cout << (*activationOut).getRows() << "x" << (*activationOut).getColumns() << std::endl;
            // std::cout << (*outputlayer.activationOut).getRows() << "x" << (*outputlayer.activationOut).getColumns() << std::endl;
            // exit(0);
          outputlayer.dels_ = multiplyVector(*outputlayer.residuals_,*outputlayer.activationOut);// (a_j - y_j) * sig`(z_j)
            //  outputlayer.dels_ =outputlayer.residuals_;// (a_j - y_j) * sig`(z_j) this is when it is linear thus no sigmoid for output layer

          outputlayer.gradients_weights = multiplyTransposem2(*outputlayer.dels_,*outputlayer.prev_.lock()->aOut_);// dL/w_i,j = (del_j) * a_i (transpose a_i)
          

            if (outputlayer.weights_->data_.size() != outputlayer.gradients_weights->data_.size()) {
                std::cerr << "SIZE MISMATCH: Weights=" << outputlayer.weights_->data_.size() 
                    << " Grads=" << outputlayer.gradients_weights->data_.size()
                     << std::endl;
                outputlayer.weights_->printDimensionz();
                outputlayer.gradients_weights->printDimensionz();

                exit(1);
            }        
        
            std::transform((outputlayer.weights_->data_).begin(),(outputlayer.weights_->data_).end(),
            (outputlayer.gradients_weights->data_).begin(),(outputlayer.weights_->data_).begin(),
             [this](double i, double j) { return (i - (j * this->learningRate )); }); // good

            
            std::transform(outputlayer.biases_->data_.begin(),outputlayer.biases_->data_.end(),
            outputlayer.dels_->data_.begin(),outputlayer.biases_->data_.begin(),
            [this](double i, double j) {return (i - (j * this->learningRate));});
            // std::cout << (*outputlayer.dels_).getRows() << "x" << (*outputlayer.dels_).getColumns() << std::endl;
            // std::cout << (*outputlayer.gradients_weights).getRows() << "x" << (*outputlayer.gradients_weights).getColumns() << std::endl;

            std::cout << "-------------done with outputlayer grad---------------" << std::endl;
       }

       void scaleFirstColumn(Matrix& myVec, double scalar){
        for (int j = 0; j < myVec.getRows(); j++){
            myVec(j,0) = scalar * myVec(j,0);
        }
       }
       void scaleAll(Matrix& myVec, double scalar){
            for (int i = 0; i < myVec.getColumns(); i++){
                for (int j = 0; j < myVec.getRows(); j++){

                myVec(j,i) = scalar * myVec(j,i);
                }
            }
       }

       void hiddenLayerGrad(Node& hidLayer ){
        double summation{0};
        bool stopNext{false}; 

               
        std::cout << "---------HIDDENlayerGrad begin--------------" << std::endl;

        std::shared_ptr<Node> currHiddenLayer = hidLayer.next_->prev_.lock(); // this is fine but in this case but error prone if we do .next->prev on last layer since it doesn't exist
        std::shared_ptr<Node> nextLayer;
        (*currHiddenLayer).printDimensions();

        while (1){ // this prob wont work since we need a_out of input layer
            nextLayer = currHiddenLayer->next_; 

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

            currHiddenLayer->dels_ = multiplyVector(*meh,*currHiddenLayer->activationOut);
            currHiddenLayer->dels_->printDimensionz();
            if (currHiddenLayer->isInputLayer){
                multiplyTransposem2(*currHiddenLayer->dels_,*currHiddenLayer->input_); //since i don't have an 'input layer' necessarily i cant use curr->prev
            }else{
                (currHiddenLayer->gradients_weights) = multiplyTransposem2(*currHiddenLayer->dels_,*currHiddenLayer->prev_.lock()->aOut_);
            }

            currHiddenLayer->gradients_weights->printDimensionz();


            // scaleAll((*currHiddenLayer->gradients_weights),learningRate); //switched to scaleAll instead for scale first column may be wrong
            currHiddenLayer->gradients_weights->printDimensionz();




            if (currHiddenLayer->weights_->data_.size() != currHiddenLayer->gradients_weights->data_.size()) {
                std::cerr << "SIZE MISMATCH: Weights=" << currHiddenLayer->weights_->data_.size() 
                    << " Grads=" << currHiddenLayer->gradients_weights->data_.size()
                     << std::endl;
                currHiddenLayer->weights_->printDimensionz();
                currHiddenLayer->gradients_weights->printDimensionz();

                exit(1);
            }        

            //0.00808702040489917 - 0.00023051553810784256= 0.0078
            std::transform((currHiddenLayer->weights_->data_).begin(),(currHiddenLayer->weights_->data_).end(),
             (currHiddenLayer->gradients_weights->data_).begin(), (currHiddenLayer->weights_->data_).begin(),
             [this](double i, double j) { return (i - (j * this->learningRate )); });//good

            std::transform(currHiddenLayer->biases_->data_.begin(), currHiddenLayer->biases_->data_.end(),
            currHiddenLayer->dels_->data_.begin(), currHiddenLayer->biases_->data_.begin(),
            [this](double i, double j) { return (i - (j * this->learningRate )); });
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
            std::transform(matrix1.data_.begin(),matrix1.data_.end(),matrix2.data_.begin(), // good
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
            /*
                should
                    takes a matri specifically a_out 
                    a_out * (1 - a_out)
                    which would fix the fact we use sigmoid(z1,true);
                    which is the only call to that overloaded function that returns a matri
                    which allows of us to get rid of the overloaded function
                    which lets simplifies the function pointers since there is only one activation function
            */

            // auto sigPtr = std::make_shared<Matrix>(z1.getRows(),z1.getColumns());
            // std::shared_ptr<Matrix> sigPtr = sigmoid(z1,true);

            std::shared_ptr  oneMinus = std::make_shared<Matrix>(z1.getRows(),z1.getColumns());
            
            for (int j =0; j < z1.getRows(); j++){
                // a (1- a)
                (*oneMinus)(j,0) = (z1)(j,0) * (1 - (z1)(j,0));

            }
            // std::shared_ptr<Matrix> product = std::make_shared<Matrix>(*multiplyVector(*sigPtr, oneMinus));
            return oneMinus;
            
        }


};
      
 
int main(){
    std::vector<int> myVe = {1,10,8,6,1};
    network myNetwork(myVe);


    return 0;
}

