import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        dp = PerceptronModel.run(self, x)
        if nn.as_scalar(dp) >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        accuracy = 0.0
        batch_size = 1
        while accuracy < 1:
            num_correct = 0
            num_samples = 0
            for x, y in dataset.iterate_once(batch_size):
                num_samples += 1
                predx = self.get_prediction(x)
                predy = nn.as_scalar(y)              
                if predx == predy:
                    num_correct += 1
                else:
                    nn.Parameter.update(self=self.get_weights(), direction=x, multiplier=predy)
            accuracy = num_correct / num_samples

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 200
        self.learning_rate = 0.005
        self.hidden_size = 512
        self.w1 = nn.Parameter(1, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, 1)
        self.b2 = nn.Parameter(1, 1)
        self.parameters = [self.w1, self.b1, self.w2, self.b2]
        

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        xw1 = nn.Linear(x, self.w1)
        xw1b1 = nn.AddBias(xw1, self.b1)
        relu = nn.ReLU(xw1b1)
        relu_w2 = nn.Linear(relu, self.w2)
        relu_w2b2 = nn.AddBias(relu_w2, self.b2)
        return relu_w2b2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        scaLoss = 1
        while scaLoss > 0.02:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                scaLoss = nn.as_scalar(loss)
                grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                self.w1.update(grad_wrt_w1, -self.learning_rate)
                self.b1.update(grad_wrt_b1, -self.learning_rate)
                self.w2.update(grad_wrt_w2, -self.learning_rate)
                self.b2.update(grad_wrt_b2, -self.learning_rate)
            

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 100
        self.learning_rate = 0.5
        self.hidden_size = 200
        
        self.w1 = nn.Parameter(784, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, 10)
        self.b2 = nn.Parameter(1, 10)
        self.parameters = [self.w1, self.b1, self.w2, self.b2]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        xw1 = nn.Linear(x, self.w1)
        xw1b1 = nn.AddBias(xw1, self.b1)
        relu = nn.ReLU(xw1b1)
        relu_w2 = nn.Linear(relu, self.w2)
        relu_w2b2 = nn.AddBias(relu_w2, self.b2)
        return relu_w2b2
        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                self.w1.update(grad_wrt_w1, -self.learning_rate)
                self.b1.update(grad_wrt_b1, -self.learning_rate)
                self.w2.update(grad_wrt_w2, -self.learning_rate)
                self.b2.update(grad_wrt_b2, -self.learning_rate)
            
            # Check validation accuracy after each epoch
            if dataset.get_validation_accuracy() >= 0.98:  # Stopping threshold
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.
    """
    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.batch_size = 25
        self.hidden_size = 350

        # Initialize model parameters
        self.w = nn.Parameter(self.num_chars, self.hidden_size)
        self.w_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        self.output_w = nn.Parameter(self.hidden_size, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.
        """
        # Initialize hidden state with first character
        h = nn.ReLU(nn.Linear(xs[0], self.w))

        # Update hidden state for each subsequent character
        for x in xs[1:]:
            h = nn.Add(nn.ReLU(nn.Linear(x, self.w)), nn.ReLU(nn.Linear(h, self.w_hidden)))

        # Compute output
        output = nn.Linear(h, self.output_w)
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.
        """
        output = self.run(xs)
        return nn.SoftmaxLoss(output, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        learning_rate = -0.09
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.w, self.w_hidden, self.output_w])
                learning_rate = min(-0.004, learning_rate)

                # Update parameters
                self.w.update(gradients[0], learning_rate)
                self.w_hidden.update(gradients[1], learning_rate)
                self.output_w.update(gradients[2], learning_rate)

            learning_rate += 0.002
            if dataset.get_validation_accuracy() >= 0.89:
                break