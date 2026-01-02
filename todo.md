Check math in hiddenLayerGrad specifically indexes

More than 1 hidden layer

Training input


I am getting inconcisistencies during runtime sometimes works sometimes doesn't




Weight matrix shape = (number of neurons in next layer) × (number of neurons in previous layer)

Bias vector shape = (number of neurons in that layer) × 1

mini-batches
![alt text](image.png)


We are not updating the biases at all
no sigmoid last layergt5er :done
must fix std::transform

check void ssr(Matrix& predicted_matrix, Matrix& target_matrix, bool isALayerGiven, std::shared_ptr<Node> givenLayerPtr)
    for loop



![alt text](image-1.png)



should have one point in the code that i use to change the act fun
