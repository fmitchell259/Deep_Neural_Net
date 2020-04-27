import numpy as np 

class Layer:
    def __init__(self, nodes_x, nodes_y, name, rand_state):
        self.name = name
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.random_gen = np.random.RandomState(rand_state)
        self.weights = self.random_gen.normal(loc=0.0,
                                              scale=0.1,
                                              size=(nodes_x, nodes_y + 1)) 
    
    def __repr__(self):
        return f"Layer {self.name}"

    def set_error_term(self, term):
        self.error_term = term


    def set_activation_output(self, a):
        self.activ_output = a

    def set_gradient_vector(self, grad_vec):
        self.gradient_vector = grad_vec

    def do_tanh(self, t):
        return np.tanh(t)


    def set_dot(self, dot):
        self.dot_product = dot
        self.net_input = dot

    def do_relu(self, X):
        return np.maximum(0, X)

    def do_dot(self, inp):

        # Add the bias before the dot product. 
        
        bias = np.ones((1, 1))

        # Assign the layer dot product from within
        # the method. 
        
        self.dot_product = np.dot(self.weights, np.append(inp, bias).T) 

    
    def do_sigmoid(self, X):
        return  1/(1 + np.exp(-X)) 

    def do_error(self):
        return False

    def do_gradient(self):
        return False

    def update_those_weights(self, lr):
        self.weights -= lr*self.gradient_vector.T

    def set_hidden_dot(self, dot):
        self.hidden_dot = dot



