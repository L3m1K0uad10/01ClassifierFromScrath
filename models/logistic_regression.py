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
    # W = np.random.randn(n_features) * 0.01  # randomizing
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
    m, n_features = X.shape

    if batch_size > m:
        raise ValueError("Batch size must be <= number of samples (m)")
    
    if shuffle and batch_size < m:
        indices = np.random.permutation(m)
        X, y_true = X[indices], y_true[indices]

    # splitting shuffle subset of the dataset accordingly to the batch size
    X_batch = X[:batch_size]
    y_batch = y_true[:batch_size]

    y_pred_batch = predict(X_batch, W, b)

    dZ = y_pred_batch - y_batch # computing error term
    XT = X_batch.T # transposing X(feature matrix)
    dW = (1 / batch_size) * np.dot(XT, dZ) # following standard implementations by reshaping it to column vector
    db = (1 / batch_size) * np.sum(dZ)

    return dW, db



def update_parameters(W, b, dW, db, alpha = 0.01): # updates weights
    """  
    PURPOSE:
    Update W and b using gradient descent.
    W = W - alpha * dW
    b = b - alpha * db

    INPUT:
    W: Current weight vector (n_features,).
    b: Current bias scalar.
    dW: Gradient for W (n_features,).
    db: Gradient for b (scalar).
    alpha: Learning rate (scalar)

    OUTPUT:
    W: Updated weight vector (n_features,).
    b: Updated bias scalar
    """ 

    W = W - alpha * dW 
    b = b - alpha * db
    
    return W, b 


def train(X_train, y_train, epochs = 8, alpha = 0.01, batch_size = 2): # runs gradient descent
    """  
    PURPOSE:
    Train the model using gradient descent.
    Iteratively update W and b for epochs iterations.

    INPUT:
    X_train: Training data (m, n_features).
    y_train: Training labels (m,).
    epochs: Number of training iterations (scalar).
    alpha: Learning rate (scalar).

    OUTPUT:(PRINT IN THIS CASE FOR MONITORING TRAINING PROGRESS)
    W: Final trained weight vector (n_features,).
    b: Final trained bias scalar.
    Optionally, you can also return the loss history for analysis.
    """ 
    m, n_features = X_train.shape
    W, b = initialize_weights(n_features)

    for epoch in range(epochs):
        y_pred = predict(X_train, W, b)
        loss = compute_loss(y_train, y_pred)
        dW, db = compute_gradients(X_train, y_train, y_pred, W, b, batch_size)
        W, b = update_parameters(W, b, dW, db, alpha)

        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, W = {W}, b = {b}")



def evaluate(X_test, y_test, W, b): # Measures model performance
    """  
    PURPOSE:
    Measure the model's accuracy on test data.
    Convert probabilities into binary predictions (0 or 1).
    Compare with true labels.

    INPUT:
    X_test: Test data (m, n_features).
    y_test: True labels (m,).
    W: Trained weight vector (n_features,).
    b: Trained bias scalar.

    OUTPUT:
    accuracy: A scalar value representing the percentage of correct predictions.
    """ 




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
print(f"loss: {loss}")

dW, db = compute_gradients(X, y_true, y_pred, W, b, 2)
print(f"dW: {dW}, db: {db}") """


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
""" epochs = 10  # Number of updates before stopping
alpha = 0.01  # Learning rate

for epoch in range(epochs):
    y_pred = predict(X, W, b)
    loss = compute_loss(y_true, y_pred)
    dW, db = compute_gradients(X, y_true, y_pred, W, b, 2)
    W, b = update_parameters(W, b, dW, db, alpha)
    
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, W = {W}, b = {b}") """

train(X, y_true)