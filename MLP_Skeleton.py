"""
Neural network with one hidden layer from scratch.

Author: Saeed Khoram. 18 Feb, 2018.
"""


from __future__ import division
from __future__ import print_function

import cPickle
import numpy as np

# epsilon
eps = 10e-8

# This is a class for a LinearTransform layer which takes an input
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, w, b):
        """
        Initializes the linear affine layer using weight and bias.
        :param w: The weight. - first layer --> [M, H]
        :param b: The bias. - first layer --> (H,)
        """
        self.w = w
        self.b = b
        self.x = None

    def forward(self, x):
        """
        Calculates the forward pass in the linear affine.
        :param x: The input to the affine layer. - first layer --> [batch_size, M]
        :return: The multiply of input and the wight added with the bias.
        """
        self.x = x
        output = np.dot(self.x, self.w) + self.b

        return output

    def backward(self, grad_output):
        """
        Calculates the gradient backward in the layer.
        :param grad_output: The gradient coming back from the next layer.
        :return: The sub gradient with respect to the input, weight, and bias.
        """
        dx = np.dot(grad_output, self.w.T)
        dw = np.dot(self.x.T, grad_output)
        db = np.sum(grad_output, axis=0)

        return dx, dw, db

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        Calculates the output of ReLU layer, given an inpu x.
        :param x: The input to the ReLU layer.
        :return: Rectified input.
        """
        self.out = np.maximum(0, x)

        return self.out

    def backward(self, grad_output):
        """
        Calculates the gradient backward in the ReLU layer.
        :param grad_output: The gradient coming back from the next layer.
        :return: The gradient with respect to the input.
        """
        self.out[self.out > 0] = 1
        dx = np.multiply(self.out, grad_output)

        return dx

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):

    def __init__(self):
        self.y = None
        self.sigx = None

    def _forward(self, x):
        """
        Calculates the output of the sigmoid function, given and input x.
        :param x: The output of the last linear affine layer. (batch,)
        :return: The logits.
        """
        self.sigx = 1.0 / (1.0 + np.exp(-x))

        return self.sigx

    def _backward(self, y, grad_output):
        """
        Calculates the gradient of the (sigmoid + cross-entropy loss) with respect to its input.
        :param y: The ground truth.
        :param grad_output:  The gradient coming back from the next layer.
        :return: The gradient of the layer with respect to its input.
        """
        self.y = y
        delta = grad_output * (self.sigx - self.y)

        return delta

    def _loss(self, y):
        """
        Calculates the value of loss in cross-entropy layer.
        :param y: The ground truth.
        :return: Loss value.
        """
        self.y = y  # todo
        loss = -1 * (self.y * np.log(self.sigx + eps) + (1.0 - self.y) * np.log(1.0 - self.sigx + eps))
        return loss


# This is a class for one hidden layer neural network
class NN(object):

    def __init__(self, input_dims, hidden_units=512, output_units=1):
        """
        Initializes the network.
        :param input_dims: The dimension of the input.
        :param hidden_units: The number of the hidden units in the network.
        :param output_units: The number of the output unit in the network.
        """
        # the network units
        self.input_dims = input_dims
        self.hidden_units = hidden_units
        self.output_units = output_units

        # the loss and accuracy corresponding to each batch.
        self.batch_loss = None
        self.batch_acc = None

        # the batch_size and L2 regularization parameter
        self.batch_size = None
        self.l2_lambda = None

        # weight and bias initialization
        self.w1, self.b1 = self._initialize_weights(input_dims, hidden_units)
        self.w2, self.b2 = self._initialize_weights(hidden_units, output_units)

        # tranform, relu, and sigmoid-cross-entropy layers
        self.LinT1 = LinearTransform(self.w1, self.b1)
        self.LinT2 = LinearTransform(self.w2, self.b2)
        self.Relu = ReLU()
        self.SigCE = SigmoidCrossEntropy()

        # momentum update
        self.mw1 = 0
        self.mw2 = 0
        self.mb1 = 0
        self.mb2 = 0

    def forward(self, x_batch):
        """
        Calculates the output by going forward layer by layer in the network.
        :param x_batch: The input batch.
        :return: The logits.
        """
        z1 = self.LinT1.forward(x_batch)
        a1 = self.Relu.forward(z1)
        z2 = self.LinT2.forward(a1)
        pred = self.SigCE._forward(z2)

        return pred

    def backward(self, y_batch):
        """
        Propagates the gradient of the loss with respect to model parameters layer by layer back to the input.
        :param y_batch: The ground truth labels corresponding to the input  bacth.
        :return: The gradients of the loss with respect to the model parameters.
        """
        delta = self.SigCE._backward(y_batch, 1) # returns the (pred - y) which is the derivative of the SigmoidCE
        dx2, dw2, db2 = self.LinT2.backward(delta)
        dx2_r = self.Relu.backward(dx2)
        dx1, dw1, db1 = self.LinT1.backward(dx2_r)

        return dw1, db1, dw2, db2

    def error(self, y_batch, l2_lambda):
        """
        Calculates the loss for each batch.
        :param y_batch: The ground truth labels of each batch.
        :param l2_lambda: The l2 regularization parameter - lambda.
        :return: The loss value.
        """
        loss = self.SigCE._loss(y_batch)
        # adding regulations
        loss += l2_lambda/2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))

        return np.mean(loss)

    def train(
            self,
            x_train,
            y_train,
            x_val,
            y_val,
            num_epochs=10,
            batch_size=128,
            learning_rate=0.001,
            momentum=0.8,
            l2_lambda=0.0):
        """
        Trains the model using gradient descent.
        :param x_train: The training data.
        :param y_train: The training labels.
        :param x_val: The validation data.
        :param y_val: The validation labels.
        :param num_epochs: The number of epochs for training.
        :param batch_size: The number of batches in each iteration.
        :param learning_rate: The learning rate for parameter update.
        :param momentum: The momentum coefficient.
        :param l2_lambda: The l2 regularization parameter - lambda.
        :return: The loss and accuracy in each epoch for training and validation data.
        """

        self.batch_size = batch_size
        self.l2_lambda = l2_lambda

        num_examples = x_train.shape[0]

        print('start training...')
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        # start training
        for epoch in xrange(num_epochs):
            # shuffle data
            rnd_ind = np.arange(num_examples)
            np.random.shuffle(rnd_ind)

            train_x = x_train[rnd_ind]
            train_y = y_train[rnd_ind]

            batch_loss = []
            batch_acc = []
            # num_batches
            num_batches = num_examples // self.batch_size
            for b in xrange(num_batches):
                # extract x and y batches
                x_batch = train_x[b * self.batch_size: (b + 1) * self.batch_size, :]
                y_batch = train_y[b * self.batch_size: (b + 1) * self.batch_size, :]

                # go forward in the layers
                self.forward(x_batch)

                # backpropagate the error
                dw1, db1, dw2, db2 = self.backward(y_batch)

                # add l2 reg gradient # todo
                dw1 += self.l2_lambda * self.w1
                dw2 += self.l2_lambda * self.w2

                # momentum
                self.mw1 = momentum * self.mw1 - learning_rate * dw1
                self.mb1 = momentum * self.mb1 - learning_rate * db1
                self.mw2 = momentum * self.mw2 - learning_rate * dw2
                self.mb2 = momentum * self.mb2 - learning_rate * db2

                # update parameters
                self.w1 += self.mw1
                self.b1 += self.mb1
                self.w2 += self.mw2
                self.b2 += self.mb2

                # calculate the loss
                loss_b = self.error(y_batch, self.l2_lambda)

                # training loss and accuracy after each batch
                y_p = np.round(self.forward(x_batch))
                acc_b = 100 * np.mean((y_p == y_batch))

                batch_loss.append(loss_b)
                batch_acc.append(acc_b)

            # training evaluation
            train_loss.append(np.mean(batch_loss))
            train_acc.append(np.mean(batch_acc))

            # validation evaluation
            v_loss, v_acc = self.evaluate(x_val, y_val)
            val_loss.append(v_loss)
            val_acc.append(v_acc)

            print("[Epoch {}/{}] \t train.Loss = {:.3f}          Train.Accuracy = {:.3f}          "\
                  "Validation.Loss = {:.3f}          Validation.Accuracy = {:.3f}"
                  .format(epoch+1, num_epochs, train_loss[epoch], train_acc[epoch], v_loss, v_acc))

        return train_loss, train_acc, val_loss, val_acc


    def _initialize_weights(self, dim_in , dim_out):
        """
        Initializes the weights and biases using random uniform distribution.
        :param dim_in: The input dimension of the affine layer.
        :param dim_out: The output dimension of the affine layer.
        :return: The randomly initialized weight and bias.
        """
        w = 10e-5 * np.random.uniform(-1.0, 1.0, size=(dim_in, dim_out))
        b = 10e-5 * np.random.uniform(-1.0, 1.0, size=(1, dim_out))

        return w, b

    def evaluate(self, x, y):
        """
        Evaluates the model performance for validation/test data.
        :param x: The input.
        :param y: The ground truth labels.
        :return: The loss and accuracy for the input.
        """
        num_exp = x.shape[0]
        rnd_ind = np.arange(num_exp)
        np.random.shuffle(rnd_ind)

        eval_x = x[rnd_ind]
        eval_y = y[rnd_ind]

        num_batches = num_exp // self.batch_size

        loss = []
        acc = []
        for b in xrange(num_batches):
            x_batch = eval_x[b * self.batch_size: (b + 1) * self.batch_size, :]
            y_batch = eval_y[b * self.batch_size: (b + 1) * self.batch_size, :]

            y_pred = np.round(self.forward(x_batch))

            loss.append(self.error(y_batch, self.l2_lambda))
            acc.append(100 * np.mean(y_pred == y_batch))

        return np.mean(loss), np.mean(acc)

    def predict(self, x):
        """
        Calculates the output of the sigmoid function in the last layer.
        :param x: The input at the first layer.
        :return: The logits.
        """
        return self.forward(x)


def preprocess_data(data_path):
    """
    Preprocessing on the data set.
    :param data_path: The path to the data.
    :return: The training data, training labels, the testing data, the resting labels, and the dimension of the data.
    """
    print("loading the data...")
    data = cPickle.load(open(data_path, 'rb'))
    x_train = data['train_data'].astype(np.float64)
    y_train = data['train_labels'].astype(np.float64)
    x_test = data['test_data'].astype(np.float64)
    y_test = data['test_labels'].astype(np.float64)

    print("normalizing the data...")
    # min-max normalization on the images
    x_max = np.max(x_train, axis=0)
    x_train = x_train / x_max
    x_test = x_test / x_max

    x_mean = np.mean(x_train, axis=0)
    x_train = (x_train - x_mean)
    x_test = (x_test - x_mean)

    data_dimension = x_train.shape[1]

    return x_train, y_train, x_test, y_test, data_dimension


if __name__ == '__main__':
    # path to data
    data_path = 'cifar-2class-py2/cifar_2class_py2.p'

    # load the data
    x_train, y_train, x_test, y_test, data_dimension = preprocess_data(data_path)

    # build the model
    nn = NN(input_dims=data_dimension, hidden_units=512, output_units=1)

    # train the model
    train_loss, train_Acc, val_loss, val_acc = nn.train(
                                                        x_train=x_train,
                                                        y_train=y_train,
                                                        x_val=x_test,
                                                        y_val=y_test,
                                                        num_epochs=10,
                                                        batch_size=128,
                                                        learning_rate=0.001,
                                                        momentum=0.8,
                                                        l2_lambda=0.0
                                                        )

    # evaluate the trained model
    test_loss, test_acc = nn.evaluate(x_test, y_test)
    print("\nTest.Loss = {}          Test.Accuracy = {}".format(test_loss, test_acc))



