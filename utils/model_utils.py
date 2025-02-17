import joblib



# Function to save weights, biases, or any model-related data
def save_model(weights, biases, filename):
    """Save trained weights and biases."""
    model_data = {"weights": weights, "biases": biases}
    joblib.dump(model_data, filename)
    print(f"Model saved to {filename}")


# Function to load weights and biases
def load_model(filename):
    """Load the trained model data."""
    model_data = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model_data["weights"], model_data["biases"]
