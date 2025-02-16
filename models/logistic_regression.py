import numpy as np



def initialize_weights(n_features): # initializes weights and bias
    """
    PURPOSE: 
    Initialize weights (W) and bias (b) to small values (usually zero).
    W should have the same number of elements as the input features.

    INPUT:
    n_features: Number of input features.

    OUTPUT:
    W: A numpy array of shape (n_features,), initialized to small values (e.g., zeros)
    b: A scalar value (usually initialized to 0) ==> Bias
    """ 
    W = np.zeros((n_features, ))
    b = 0

    return W, b


def sigmoid(z): # computes the sigmoid function
    """
    PURPOSE: (mapping real-valued numbers to values between 0 and 1)
    Compute the sigmoid activation function for a given input z.
    Used to squash values into the range (0,1) for probability output.  

    INPUT:
    z: A scalar or numpy array.

    OUTPUT:
    sigmoid(z): A scalar or numpy array of the same shape as z, with values between 0 and 1.
    """ 

    return 1 / (1 + np.exp(-z)) # using np.exp() instead of math.pow() or math.exp() because these works only for scalar Thus error when Numpy array


def predict(X, W, b): # computes output probabilities
    """
    PURPOSE:
    Compute the predicted probabilities using the logistic regression formula:
    ^
    y = σ(XW + b)

    INPUTS:
    X: NumPy array of shape (m, n_features) → input features.
    W: NumPy array of shape (n_features,) → model weights.
    b: Scalar → bias term.

    OUTPUT:
    NumPy array of shape (m,) → Predicted probabilities (values between 0 and 1).

    NOTE:
    In machine learning, m represents the number of training examples (data points) in the dataset
    """ 

    # let me compute z = XW + b first and after get its sigmoid value
    Z = np.dot(X, W) + b
    y = sigmoid(Z)

    return y


def compute_loss(y_true, y_pred): # computes binary cross entropy loss
    """
    A loss function in machine learning helps models learn by quantifying the difference between predicted and actual values
    it's based on PROBALITY DISTRIBUTION AND THEIR SURPRISES

    PURPOSE:
    Compute the binary cross-entropy loss.
    Measures how well the model predicts labels.
    if the predicted probability for a label is close to the correct label, the loss is small
    if the predicted probability for a label is far from the correct label, the loss is large
    the goal is to minimize this loss so predictions match actual labels

    INPUT:
    y_true: Ground truth labels, a numpy array of shape (m,) (values: 0 or 1).
    y_pred: Model-predicted probabilities, a numpy array of shape (m,).

    OUTPUT:
    loss: A single scalar value (average binary cross-entropy).
    """ 

    # number of total row in the dataset
    m = y_true.shape[0]

    sum = 0

    for i in range(m):
        sum += (y_true[i] * np.log(y_pred[i])) + ((1 - y_true[i]) * np.log(1 - y_pred[i]))

    loss = (-1 / m) * sum  # binary cross-entropy loss formula
    
    return loss 


# i will implement Mini-Batch Gradient Descent as it is the middle ground between Batch 
# Gradient Descent (BGD) and Stochastic Gradient Descent (SGD)
def compute_gradients(X, y_true, y_pred, W, b, batch_size = 32, shuffle = True): # computes gradients
    """
    PURPOSE:
    Compute the gradients (derivatives) of the loss with respect to W and b
    Needed for updating parameters using gradient descent

    INPUT:
    X: Feature matrix of shape (m, n_features)
    y_true: True labels of shape (m,)
    y_pred: Predicted probabilities of shape (m,)
    batch_size: (
        if batch_size = m then it's equivalent to Batch GD
        if batch_size = 1 then it's equivalent to Stochastic GD
    )
    shuffle:

    OUTPUT:
    dW: A numpy array of shape (n_features,) (gradient for W)
    db: A scalar (gradient for b)
    """ 
    m = X.shape[0]

    if batch_size <= m:
        """  
        NOTE: Actually for me there is no need for shuffling or selecting a subset when batch_size is equal to m
        """
        # reconstructing the total dataset with their y_true
        stack_X_Y = np.hstack((X, y_true.reshape(-1, 1))) 

        if shuffle:
            shuffle_X = np.random.permutation(stack_X_Y) # it shuffles along the indices
            # splitting shuffle subset of the dataset accordingly to the batch size
            batch = shuffle_X[:batch_size]
        else:
            batch = stack_X_Y[:batch_size]
        X_batch = batch[:, :-1]
        y_batch = batch[:, -1].flatten() # assuring to be on 1D form by using flatten

        y_pred_batch = predict(X_batch, W, b)
        loss_batch = compute_loss(y_batch, y_pred_batch)

        dZ = y_pred_batch - y_batch # computing error term
        XT = X_batch.T # transposing X(feature matrix)
        dW = (1 / batch_size) * np.dot(XT, dZ) # following standard implementations by reshaping it to column vector
        db = (1 / batch_size) * np.sum(dZ)
    else:
        print("Error: check the batch size and see, should be(batch_size <= m)")
        return None, None

    return dW, db



def update_parameters(W, b, dW, db, alpha): # updates weights
    pass 


def train(X_train, y_train, epochs, alpha): # runs gradient descent
    pass 


def evaluate(X_test, y_test, W, b): # Measures model performance
    pass 




""" no_features = 4
res = initialize_weights(no_features)
print(res) """


""" z_scalar = 2
z_array = np.array([-1, 0, 1, 2])
res1 = sigmoid(z_scalar)
res2 = sigmoid(z_array)
print(res1)
print(res2) """

""" X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0], 
    [7.0, 8.0, 9.0],  
    [10.0, 11.0, 12.0],  
    [13.0, 14.0, 15.0],
])  
n_features = X.shape[1]
W, b = initialize_weights(n_features)
res = predict(X, W, b)
print(f" predicted: {res}") """


""" X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0], 
    [7.0, 8.0, 9.0],  
    [10.0, 11.0, 12.0],  
    [13.0, 14.0, 15.0],
])

y_true = np.array([1, 0, 0, 1, 1])

n_features = X.shape[1]
W, b = initialize_weights(n_features)
y_pred = predict(X, W, b)
print(f"actual: {y_true}")
print(f"predicted: {y_pred}")

loss = compute_loss(y_true, y_pred)
print(f"loss: {loss}") """


X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0], 
    [7.0, 8.0, 9.0],  
    [10.0, 11.0, 12.0],  
    [13.0, 14.0, 15.0],
])

y_true = np.array([1, 0, 0, 1, 1])

n_features = X.shape[1]
W, b = initialize_weights(n_features)
y_pred = predict(X, W, b)
print(f"actual: {y_true}")
print(f"predicted: {y_pred}")

loss = compute_loss(y_true, y_pred)
print(f"loss: {loss}")

dW, db = compute_gradients(X, y_true, y_pred, W, b, 2)
print(f"dW: {dW}, db: {db}")