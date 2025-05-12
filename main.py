# import pandas as pd
# import joblib
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split

# file_path = 'injury_data.csv'




# def load_data(file_path):
#     return pd.read_csv(file_path)


# def preprocess_data(df, task='classification'):
#     df = df.copy()

#     # Drop rows with missing values
#     df.dropna(inplace=True)

#     # Define features and target
#     if task == 'classification':
#         X = df.drop(['Injury', 'Recovery_Time'], axis=1)
#         y = df['Injury']
#     else:
#         X = df.drop(['Injury', 'Recovery_Time'], axis=1)
#         y = df['Recovery_Time']

#     # Identify numerical and categorical columns
#     num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     cat_cols = X.select_dtypes(include=['object']).columns.tolist()

#     # Create preprocessing pipeline
#     num_pipeline = Pipeline([
#         ('scaler', StandardScaler())
#     ])

#     cat_pipeline = Pipeline([
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))
#     ])

#     preprocessor = ColumnTransformer([
#         ('num', num_pipeline, num_cols),
#         ('cat', cat_pipeline, cat_cols)
#     ])

#     X_processed = preprocessor.fit_transform(X)

#     return X_processed, y, preprocessor, num_cols, cat_cols

# def train_model(X, y, task='classification'):
#     if task == 'classification':
#         model = RandomForestClassifier(random_state=42)
#     else:
#         model = RandomForestRegressor(random_state=42)

#     model.fit(X, y)
#     return model

# def main():
#     # LOAD DATA
#     file_path = 'injury_data.csv'
#     df = load_data(file_path)

#     # PREPROCESS AND TRAIN CLASSIFIER
#     X_clf, y_clf, preprocessor_clf, num_cols_clf, cat_cols_clf = preprocess_data(df, task='classification')
#     clf_model = train_model(X_clf, y_clf, task='classification')
#     joblib.dump(clf_model, 'classifier_model.pkl')
#     joblib.dump(preprocessor_clf, 'preprocessor_clf.pkl')

#     # PREPROCESS AND TRAIN REGRESSOR
#     X_reg, y_reg, preprocessor_reg, num_cols_reg, cat_cols_reg = preprocess_data(df, task='regression')
#     reg_model = train_model(X_reg, y_reg, task='regression')
#     joblib.dump(reg_model, 'regressor_model.pkl')
#     joblib.dump(preprocessor_reg, 'preprocessor_reg.pkl')

#     # USER INPUT PREDICTION
#     print("\n" + "="*50)
#     print("PREDICTION FOR USER-ENTERED INPUT")
#     print("="*50)

#     # Collect input from user
#     input_data = {
#         'Player_Age': float(input("Enter Player Age: ")),
#         'Player_Weight': float(input("Enter Player Weight (kg): ")),
#         'Player_Height': float(input("Enter Player Height (cm): ")),
#         'Previous_Injuries': int(input("Enter Number of Previous Injuries: ")),
#         'Training_Intensity': input("Enter Training Intensity (Low/Medium/High): ")
#         # Add more input prompts here if you have more features
#     }

#     user_df = pd.DataFrame([input_data])

#     # Load models and preprocessors
#     clf_model = joblib.load('classifier_model.pkl')
#     reg_model = joblib.load('regressor_model.pkl')
#     preprocessor_clf = joblib.load('preprocessor_clf.pkl')
#     preprocessor_reg = joblib.load('preprocessor_reg.pkl')

#     # Transform user input
#     user_X_clf = preprocessor_clf.transform(user_df)
#     user_X_reg = preprocessor_reg.transform(user_df)

#     # Predict
#     user_clf_pred = clf_model.predict(user_X_clf)[0]
#     user_reg_pred = reg_model.predict(user_X_reg)[0]

#     # Show output
#     print(f"\nPredicted Likelihood of Injury: {'Yes' if user_clf_pred == 1 else 'No'}")
#     print(f"Predicted Recovery Time (in days): {user_reg_pred:.2f}")

# if __name__ == '__main__':
#     main()


import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

file_path = 'injury_data.csv'

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, task='classification'):
    df = df.copy()

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Define features and target
    if task == 'classification':
        X = df.drop(['Likelihood_of_Injury', 'Recovery_Time'], axis=1)
        y = df['Likelihood_of_Injury']  # Fixed: Changed 'Injury' to 'Likelihood_of_Injury'
    else:
        X = df.drop(['Likelihood_of_Injury', 'Recovery_Time'], axis=1)
        y = df['Recovery_Time']

    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Create preprocessing pipeline
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor, num_cols, cat_cols

def train_model(X, y, task='classification'):
    if task == 'classification':
        model = RandomForestClassifier(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    model.fit(X, y)
    return model

def main():
    # LOAD DATA
    file_path = 'injury_data.csv'
    df = load_data(file_path)

    # PREPROCESS AND TRAIN CLASSIFIER
    X_clf, y_clf, preprocessor_clf, num_cols_clf, cat_cols_clf = preprocess_data(df, task='classification')
    clf_model = train_model(X_clf, y_clf, task='classification')
    joblib.dump(clf_model, 'classifier_model.pkl')
    joblib.dump(preprocessor_clf, 'preprocessor_clf.pkl')

    # PREPROCESS AND TRAIN REGRESSOR
    X_reg, y_reg, preprocessor_reg, num_cols_reg, cat_cols_reg = preprocess_data(df, task='regression')
    reg_model = train_model(X_reg, y_reg, task='regression')
    joblib.dump(reg_model, 'regressor_model.pkl')
    joblib.dump(preprocessor_reg, 'preprocessor_reg.pkl')

    # USER INPUT PREDICTION
    print("\n" + "="*50)
    print("PREDICTION FOR USER-ENTERED INPUT")
    print("="*50)

    # Collect input from user
    input_data = {
        'Player_Age': float(input("Enter Player Age: ")),
        'Player_Weight': float(input("Enter Player Weight (kg): ")),
        'Player_Height': float(input("Enter Player Height (cm): ")),
        'Previous_Injuries': int(input("Enter Number of Previous Injuries: ")),
        'Training_Intensity': input("Enter Training Intensity (Low/Medium/High): ")
        # Add more input prompts here if you have more features
    }

    user_df = pd.DataFrame([input_data])

    # Load models and preprocessors
    clf_model = joblib.load('classifier_model.pkl')
    reg_model = joblib.load('regressor_model.pkl')
    preprocessor_clf = joblib.load('preprocessor_clf.pkl')
    preprocessor_reg = joblib.load('preprocessor_reg.pkl')

    # Transform user input
    user_X_clf = preprocessor_clf.transform(user_df)
    user_X_reg = preprocessor_reg.transform(user_df)

    # Predict
    user_clf_pred = clf_model.predict(user_X_clf)[0]
    user_reg_pred = reg_model.predict(user_X_reg)[0]

    # Show output
    print(f"\nPredicted Likelihood of Injury: {'Yes' if user_clf_pred == 1 else 'No'}")
    print(f"Predicted Recovery Time (in days): {user_reg_pred:.2f}")

if __name__ == '__main__':
    main()