import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



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
    # W = np.zeros((n_features, ))
    W = np.random.randn(n_features) * 0.01
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
    z = np.clip(z, -500, 500) # prevent extreme large values

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
    epsilon = 1e-10 # small value to prevent log(0)

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    sum = 0

    for i in range(m):
        sum += (y_true[i] * np.log(y_pred[i])) + ((1 - y_true[i]) * np.log(1 - y_pred[i]))

    loss = (-1 / m) * sum  # binary cross-entropy loss formula
    
    return loss 


# i will implement Mini-Batch Gradient Descent as it is the middle ground between Batch 
# Gradient Descent (BGD) and Stochastic Gradient Descent (SGD)
def compute_gradients(X_batch, y_batch, y_pred_batch, W, b, batch_size = 32, shuffle = True): # computes gradients
    """
    PURPOSE:
    Compute the gradients (derivatives) of the loss with respect to W and b
    Needed for updating parameters using gradient descent

    INPUT:
    X_batch: Feature matrix of shape (m, n_features)
    y_batch: True labels of shape (m,)
    y_pred_batch: Predicted probabilities of shape (m,)
    batch_size: (
        if batch_size = m then it's equivalent to Batch GD
        if batch_size = 1 then it's equivalent to Stochastic GD
    )
    shuffle:

    OUTPUT:
    dW: A numpy array of shape (n_features,) (gradient for W)
    db: A scalar (gradient for b)
    """ 
    batch_size = X_batch.shape[0] 

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


def train(X_train, y_train, alpha = 0.01, epochs = 20, batch_size = 2): # runs gradient descent
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
        indices = np.arange(m)
        np.random.shuffle(indices) # Shuffle dataset at the start of each epoch
        X_train, y_train = X_train[indices], y_train[indices]

        epoch_loss = 0  
        for i in range(0, m, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            y_pred_batch = predict(X_batch, W, b)
            batch_loss = compute_loss(y_batch, y_pred_batch)
            epoch_loss += batch_loss

            dW, db = compute_gradients(X_batch, y_batch, y_pred_batch, W, b)
            W, b = update_parameters(W, b, dW, db, alpha)

        avg_loss = epoch_loss / (m / batch_size)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, W[:5] = {W[:5] if len(W) > 5 else W}, b = {b:.4f}")

    return W, b


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

    y_pred_prob = sigmoid(np.dot(X_test, W) + b)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


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

train(X, y_true, 0.005) """


""" X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0], 
    [7.0, 8.0, 9.0],  
    [10.0, 11.0, 12.0],  
    [13.0, 14.0, 15.0],
])

y_true = np.array([1, 0, 0, 1, 1])

# Feature normalization (standardization)
def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

X_normalized = normalize(X)

# Initialize weights
n_features = X.shape[1]
W, b = initialize_weights(n_features)

# Training execution
if __name__ == "__main__":
    train(X_normalized, y_true, 0.005)   """