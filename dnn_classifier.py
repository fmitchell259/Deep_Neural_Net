import numpy as np 
from dnn_object import DeepNeuralNet
from one_hot import one_hot
import pandas as pd 

def main():

    # Here in our main method we simply load and scale the
    # data, one hot encode the labels and then initialise
    # our deep neural net. 

    # Below we test our network with both activation functions,
    # 'ReLU' and 'Tanh'.

    train_data = pd.read_csv('mnist_train.csv')
    test_data = pd.read_csv('mnist_test.csv')

    train_labels = train_data.iloc[:,0]
    test_labels = test_data.iloc[:,0]
    train_labels = one_hot(train_labels, 10)

    train_data = train_data.iloc[:, 1:] / 255.00
    test_data = test_data.iloc[:, 1:] / 255.00

    train_data = train_data.to_numpy()
    test_data = test_data.to_numpy()

    dnn_relu = DeepNeuralNet(0.001, 784, 2, 28, 10, 5, "relu", "MEGATRON!")

    dnn_relu.fit(train_data, train_labels)

    predictions = []

    for item in test_data:
        p = list(dnn_relu.predict(item))
        predictions.append(p.index(max(p)))

    dnn_relu.accuracy_scores(predictions, test_labels)

    print("\n\n=============\n\n")

    dnn_tanh = DeepNeuralNet(0.001, 784, 2, 28, 10, 5, "tanh", "MEGATRON!")

    dnn_tanh.fit(train_data, train_labels)

    predictions = []

    for item in test_data:
        p = list(dnn_tanh.predict(item))
        predictions.append(p.index(max(p)))

    dnn_tanh.accuracy_scores(predictions, test_labels)


main()
