import neural_network_functions as nnf
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()
print(train_X.shape)

W1, b1, W2, b2 = nnf.init_params()
iterations = 100



for i in range(iterations):
    Z1, A1, Z2, A2 = nnf.forward_prop(W1, b1, W2, b2, train_X)
    dW1, db1, dW2, db2 = nnf.back_prop(Z1, A1, Z2, A2, W2, train_X, train_y)
    W1, b1, W2, b2 = nnf.update_params(Z1, b1, W2, b2, dW1, db1, dW2, db2, .1)
    if i % 50 == 0:
        print("iteration: ", i)
        print("Accuracy: ", nnf.get_accuracy(nnf.get_predictions(A2), train_y))
        

