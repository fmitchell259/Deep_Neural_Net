import numpy as np 
from layer import Layer
import time

CONSTANT = np.ones((1, 1), dtype='float32')

class DeepNeuralNet:
    def __init__(self, lr, inputs, hidden, nodes, classes, epochs, activ, name):
        self.name = name
        self.lr = lr
        self.epochs = epochs
        self.inputs = inputs
        self.hidden = hidden
        self.nodes = nodes
        self.classes = classes
        self.deep_net = [None for _ in range(hidden + 2)]
        self.activation_function  = activ
        self.initalise()

    def __repr__(self):
        return f"I AM {self.name}, YOUR LORD AND MASTER"


    def initalise(self):

        print(f"[+] Initialising Deep Neural Net with '{self.activation_function}' activation")

        self.deep_net[0] = Layer(self.nodes, self.inputs, "in", 42)
        
        print("\t[+] Input layer added\n")
        
        for _ in range(self.hidden):

            self.deep_net[_+ 1] = Layer(self.nodes, self.nodes, f"Hidden {_ + 1}", 42)
            print(f"\t\t[+] Hidden Layer {_ + 1} added\n")
            

        self.deep_net[-1] = Layer(self.classes, self.nodes, "out", 42)

        print("\t[+] Output layer added\n")

        print("[+] Deep Neural Net Initialised\nThank you for your service.\n")

    def forward_pass(self, X):

        # Each instance is passed through the network
        # multiplying each feature by every hidden
        # node. 

        # Final output are class probabilities.

        inp = X
        for layer in self.deep_net:
            if layer.name == 'out':
                layer.do_dot(inp)
                if self.activation_function == 'relu':
                    layer.set_activation_output(layer.do_relu(layer.dot_product[:, np.newaxis]))
                
                elif self.activation_function == 'tanh':
                    layer.set_activation_output(layer.do_tanh(layer.dot_product[:, np.newaxis]))
                
                return layer.activ_output

            else:
                layer.do_dot(inp)
                if self.activation_function == 'relu':
                    layer.set_activation_output(layer.do_relu(layer.dot_product[:, np.newaxis]))
                
                elif self.activation_function == 'tanh':
                    layer.set_activation_output(layer.do_tanh(layer.dot_product[:, np.newaxis]))
                
                inp = layer.activ_output

    def backward_pass(self, X, y):
        
        # Here we do a backward pass through the network
        # and update the error terms and gradient vector.
        
        for i, layer in enumerate(reversed(self.deep_net)):

            # As soon as we enter the loop we set up our
            # partial derivatives and assign one of these
            # to an object attribute, depending on the user
            # requirement. 

            re_der = (layer.activ_output > 0) * 1.0
            tan_der = 1-np.tanh(layer.activ_output)**2

            if self.activation_function == "relu":
                self.partial_derivative = re_der
            
            elif self.activation_function == "tanh":
                self.partial_derivative = tan_der
            
            if layer.name == "out":

                # Then we set the error term for this layer. by multuplying
                # the error by the partial derivative. 

                layer.set_error_term(layer.activ_output - y[:, np.newaxis] * self.partial_derivative)
                 
                # With the error term set we can use this value to assign a 
                # gradient vector to the layer. 

                layer.set_gradient_vector(np.dot(np.append(self.deep_net[::-1][i + 1].activ_output, CONSTANT)[:, np.newaxis], 
                                                          layer.error_term.T))

                # The loop continues until all gradient vectors have been 
                # assigned. 

            else:

                # An extra step we need to take is setting the dot product
                # of the hidden payer. This is because we use the layer + 1
                # error term and weights (along with the this layers partial
                # derivative) to calculate this layer's error term.

                layer.set_hidden_dot(np.dot(self.deep_net[abs((self.hidden + 2) - i)].error_term.T, 
                                            self.deep_net[abs((self.hidden + 2) - i)].weights[:, 1:]))


                # With all those values assigned we use the layer method
                # to assign this layers error term. 

                layer.set_error_term(layer.hidden_dot.T * self.partial_derivative)

                # An IF-ELSE is required here as the INPUT layer updates the
                # the gradient vector based on the INPUT, and all layers 
                # use the OUTPUT from the previous layer. Seen as the INPUT
                # layer is the first layer in the network, this is not
                # possible. 

                if layer.name == 'in':

                    # Gradient vector updated by the input multiplied by the error term. 

                    layer.set_gradient_vector(np.append(X, CONSTANT)[:, np.newaxis] * layer.error_term.T)
                else:

                    # Here we need to add a constant to account for the bias, and then an 
                    # element-wise multiplication takes place to update the gradient vector.  

                    inpu_plus_constant = np.append(self.deep_net[::-1][i + 1].activ_output, CONSTANT)[:, np.newaxis]

                    layer.set_gradient_vector(inpu_plus_constant * layer.error_term.T)
         

    def fit(self, X, y):

        # Forward pass runs all data through the net and saves
        # net input and layer output. 

        for _ in range(self.epochs):

            print(f"Epoch {_ + 1}")

            for xi, yi in zip(X, y):

                # counter=0
                # for xi, yi in zip(X, y):

                self.forward_pass(xi)

                # Now we do the back-propogation.

                # Error vector and gradient vector are both saved within
                # the layer object and used to update weights.

                self.backward_pass(xi, yi)

                # Now with each layers gradient vector ready we can update
                # all weights at once. 
                
                self.update_weights()

                # counter = counter+1
                # if counter%100==0:
                #     self.test(X[1],y[1])
        

    def test(self,x,y):
        print (self.predict(x))
        print (y)

    def update_weights(self):

        # Cool, so gradient vectors now sorted for all layers
        # We just need to update! 

        for l in self.deep_net:
            l.update_those_weights(self.lr)
    
    def predict(self, X):
        
        # Predict simply runs a forward pass and
        # outputs the class probability list. 

        p = self.forward_pass(X)
        return p
    
    def accuracy_scores(self, pred, lab):
        
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for pred, lab in zip(pred, lab):
            if lab == 0:
                if pred == 0:
                    tn += 1
                else:
                    fp += 1
            else:
                if pred == 0:
                    fn += 1
                else:
                    tp += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * (tp + fp + fn))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(f"\n[+] Deep Neural Net with '{self.activation_function}' activation accuracy scores:\n")
        print(f"\t[+] Model Precision: {precision}")
        print(f"\t[+] Model Recall: {recall}")
        print(f"\t[+] Model F1 Score: {f1}")
        print(f"\t[+] Model Accuracy Score: {accuracy}")

