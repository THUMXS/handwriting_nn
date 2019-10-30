"""

Handwritten number identification using a feedforward neural net.
Using the MNIST set of images and labels for training and testing.

Kelvin Peng
September 2019

"""

import numpy as np
import random
import copy
import math
import datetime 

from mnist import MNIST
    
class Network(object):

    def __init__(self, arg):
        # initialize with random weights and biases based on size list arg
        if type(arg) == list:
            self.sizes = arg # layer sizes
            self.num_layers = len(self.sizes) # number of layers in net

            # bias and weight lists
            self.biases = [[random.random() * 2 - 1 for _ in range(self.sizes[i + 1])] for i in range(self.num_layers - 1)]
            self.weights = [[[random.random() * 2 - 1 for _ in range(self.sizes[i + 1])] for _ in range(self.sizes[i])] for i in range(self.num_layers - 1)]
        
        # initialize with existing weights in biases from file with name arg
        elif type(arg) == str:
            f = open("states/" + arg, "r")
            
            # read in layer sizes of net
            self.sizes = [int(_) for _ in f.readline().split(" ")[0:-1]]
            self.num_layers = len(self.sizes)

            self.biases = []
            self.weights = []

            # read in saved biases
            for i in range(self.num_layers - 1):
                self.biases.append([float(_) for _ in f.readline().split(" ")[0:-1]])

            # read in saved weights
            for i in range(self.num_layers - 1):
                a = []
                for _ in range(self.sizes[i]):
                    a.append([float(_) for _ in f.readline().split(" ")[0:-1]])
                self.weights.append(a)
            
            f.close()

    # classify image a
    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(a, w) + b)
        return a
    
    # produce adjustment to biases and weights
    # for lower cost for image "a" and label "l"
    # a: 1xn list of [0, 1] float values representing image
    # l: integer label classifying a
    def backprop(self, a, l):
        # create target array (ideal result)
        y = [0] * 10
        y[l] = 1
        
        a_s = [a] # each layer's activations
        z_s = [] # each layer's z values (z = w dot a + b)
        
        # feed forward
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b # current layer's z
            z_s.append(z) # track z
            a = sigmoid(z) # current layer's activation
            a_s.append(a) # track a
        
        # delta for biases and weights for this backprop
        del_b = [[] for _ in range(self.num_layers - 1)]
        del_w = [[] for _ in range(self.num_layers - 1)]

        # modify last layer (L)
        delta = [m * n for m, n in zip([2 * (i - j) for i, j in zip(a_s[-1], y)], sigmoid_prime(z_s[-1]))]
        
        del_b[-1] = delta
        del_w[-1] = np.outer(a_s[-2], delta)

        # modify layers 0 to L-1
        for L in range(2, self.num_layers):
            delta = [i * j for i, j in zip(sigmoid_prime(z_s[-L]), np.dot(self.weights[-L+1], delta))]

            del_b[-L] = delta
            del_w[-L] = np.outer(a_s[-L-1], delta)

        return (del_b, del_w)
    
    # backprop for each (image, label) tuple in batch
    # adjust weights and biases based on average of all backprops
    def processBatch(self, batch):
        # lists for summing backprop result
        sigma_del_b = [[0] * len(self.biases[i]) for i in range(len(self.biases))]
        sigma_del_w = [[[0] * len(self.weights[i][j]) for j in range(len(self.weights[i]))] for i in range(len(self.weights))]

        # run training on each training pair
        for a, l in batch:
            # backprop
            delta_n_b, delta_n_w = self.backprop(a, l)

            # sum backprop result to running total
            sigma_del_b = [[i + j for i, j in zip(m, n)] for m, n in zip(sigma_del_b, delta_n_b)]
            sigma_del_w = [[[i + j for i, j in zip(m, n)] for m, n in zip(a, b)] for a, b in zip(sigma_del_w, delta_n_w)]

        # adjust biases and weights based on backprop derivative
        # add average of backprop results to bias and weight lists
        self.biases = [[i - j * 1.0 / len(batch) for i, j in zip(m, n)] for m, n in zip(self.biases, sigma_del_b)]
        self.weights = [[[i - j * 1.0 / len(batch) for i, j in zip(m, n)] for m, n in zip(a, b)] for a, b in zip(self.weights, sigma_del_w)]
    
    # save network's state (biases and weights) in file
    def save_state(self):
        print("saving network state...")

        now = datetime.datetime.now()
        fn = "states/bw_%d-%d_%d-%d-%d" % (now.year, now.month, now.day, now.hour, now.minute)

        with open(fn, "w") as f:
            for _ in self.sizes:
                f.write("%i " %(_))
            f.write("\n")

            for r in self.biases:
                for n in r:
                    f.write("%1.8f " %(n))
                f.write("\n")

            for l in self.weights:
                for r in l:
                    for n in r:
                        f.write("%1.8f " %(n))
                    f.write("\n")
            
            f.close()

        print("network state saved")

# calculate sigmoid on "a"
def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))

# calculate derivative of sigmoid on "a"
def sigmoid_prime(a):
    return sigmoid(a) * (1 - sigmoid(a))

# calculate cost for output "y" from feedforward of image of class "label"
def cost(y, label):
    y[label] = 1 - y[label]
    return sum(v ** 2 for v in y)

# load mnist data
def load_data(type):
    print("loading data...")
    
    mndata = MNIST('samples')
    
    if type == "testing":
        images, labels = mndata.load_testing()
    elif type == "training":
        images, labels = mndata.load_training()
    else:
        return None
    
    data = [([v / 255.0 for v in i], l) for i, l in zip(images, labels)]
    
    print("finished loading data")
    
    return data

# evaluate network accuracy with mnist testing data
def test_net():
    data = load_data("testing")

    num_correct = 0
    for t in data:
        y = net.feedforward(t[0])
        if np.argmax(y) == t[1]:
            num_correct += 1
    
    # print % correct
    print("%3.2f percent correct" %(num_correct * 100.0 / len(data)))

def train_net(iter):
    data = load_data("training")

    for i in range(iter):
        random.shuffle(data)
        
        print("starting iteration %d/%d of training" %(i, iter))
        
        for f in range(int(len(data) // batch_size)):
            net.processBatch(data[f * batch_size : (f + 1) * batch_size - 1])
        
        print("iteration %d/%d complete" %(i, iter))
        
        net.save_state()

# create net
#net = Network([784, 24, 16, 10])
net = Network("bw_2019-10_30-0-58")
batch_size = 1000

#train_net(1)
test_net()
# bw_21-57_13-9-2019 got 94.57%