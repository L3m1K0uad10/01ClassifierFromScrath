import pandas as pd 
from sklearn.model_selection import train_test_split



def load_and_split_data(filepath, test_size = 0.15, val_size = 0.15, random_state = 42):
    df = pd.read_csv(filepath)

    # the last column is the labels
    X = df.iloc[: :-1].values # .values for directly convert to Numpy Array
    y = df.iloc[:, -1].values 

    # temp for (validation + test) for further split them
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, 
        y, 
        test_size = test_size + val_size, 
        random_state = random_state,
        stratify = y
    )

    # splitting temp data
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp, 
        test_size = test_size + val_size,
        random_state = random_state,
        stratify = y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test