cd .\neural-net-framework-from-scratch\cudaPortion\
nvcc .\vectorAddition.cu -o app.exe
nsys profile --stats=true ./app.exe
g++ matrix.cpp network.cpp node.cpp  -o app -std=c++20
