from utils.preprocessing import load_and_split_data
from models.logistic_regression import train, evaluate
from utils.model_utils import save_model



file_path = "data/data_banknote_authentication.csv"
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(file_path)

print(f"Training Set: {X_train.shape}, {y_train.shape}")
print(f"Validation Set: {X_val.shape}, {y_val.shape}")
print(f"Test Set: {X_test.shape}, {y_test.shape}")


trained_Weights, trained_biases = train(X_train, y_train, 0.001, 16, 32)

save_model(trained_Weights, trained_biases, "banknote_auth_Model.pkl")

evaluate(X_test, y_test, trained_Weights, trained_biases)