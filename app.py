# # import pandas as pd
# # import joblib
# # from sklearn.compose import ColumnTransformer
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# # from sklearn.pipeline import Pipeline
# # from sklearn.model_selection import train_test_split

# # file_path = 'injury_data.csv'

# # def load_data(file_path):
# #     return pd.read_csv(file_path)

# # def preprocess_data(df, task='classification'):
# #     df = df.copy()

# #     # Drop rows with missing values
# #     df.dropna(inplace=True)

# #     # Define features and target
# #     if task == 'classification':
# #         X = df.drop(['Likelihood_of_Injury', 'Recovery_Time'], axis=1)
# #         y = df['Likelihood_of_Injury'] 
# #     else:
# #         X = df.drop(['Likelihood_of_Injury', 'Recovery_Time'], axis=1)
# #         y = df['Recovery_Time']

# #     # Identify numerical and categorical columns
# #     num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
# #     cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# #     # Create preprocessing pipeline
# #     num_pipeline = Pipeline([
# #         ('scaler', StandardScaler())
# #     ])

# #     cat_pipeline = Pipeline([
# #         ('onehot', OneHotEncoder(handle_unknown='ignore'))
# #     ])

# #     preprocessor = ColumnTransformer([
# #         ('num', num_pipeline, num_cols),
# #         ('cat', cat_pipeline, cat_cols)
# #     ])

# #     X_processed = preprocessor.fit_transform(X)

# #     return X_processed, y, preprocessor, num_cols, cat_cols

# # def train_model(X, y, task='classification'):
# #     if task == 'classification':
# #         model = RandomForestClassifier(random_state=42)
# #     else:
# #         model = RandomForestRegressor(random_state=42)

# #     model.fit(X, y)
# #     return model

# # def main():
# #     # LOAD DATA
# #     file_path = 'injury_data.csv'
# #     df = load_data(file_path)

# #     # PREPROCESS AND TRAIN CLASSIFIER
# #     X_clf, y_clf, preprocessor_clf, num_cols_clf, cat_cols_clf = preprocess_data(df, task='classification')
# #     clf_model = train_model(X_clf, y_clf, task='classification')
# #     joblib.dump(clf_model, 'classifier_model.pkl')
# #     joblib.dump(preprocessor_clf, 'preprocessor_clf.pkl')

# #     # PREPROCESS AND TRAIN REGRESSOR
# #     X_reg, y_reg, preprocessor_reg, num_cols_reg, cat_cols_reg = preprocess_data(df, task='regression')
# #     reg_model = train_model(X_reg, y_reg, task='regression')
# #     joblib.dump(reg_model, 'regressor_model.pkl')
# #     joblib.dump(preprocessor_reg, 'preprocessor_reg.pkl')

# #     # USER INPUT PREDICTION
# #     print("\n" + "="*50)
# #     print("PREDICTION FOR USER-ENTERED INPUT")
# #     print("="*50)

# #     # Collect input from user
# #     input_data = {
# #         'Player_Age': float(input("Enter Player Age: ")),
# #         'Player_Weight': float(input("Enter Player Weight (kg): ")),
# #         'Player_Height': float(input("Enter Player Height (cm): ")),
# #         'Previous_Injuries': int(input("Enter Number of Previous Injuries: ")),
# #         'Training_Intensity': input("Enter Training Intensity (Low/Medium/High): ")
# #         # Add more input prompts here if you have more features
# #     }

# #     user_df = pd.DataFrame([input_data])

# #     # Load models and preprocessors
# #     clf_model = joblib.load('classifier_model.pkl')
# #     reg_model = joblib.load('regressor_model.pkl')
# #     preprocessor_clf = joblib.load('preprocessor_clf.pkl')
# #     preprocessor_reg = joblib.load('preprocessor_reg.pkl')

# #     # Transform user input
# #     user_X_clf = preprocessor_clf.transform(user_df)
# #     user_X_reg = preprocessor_reg.transform(user_df)

# #     # Predict
# #     user_clf_pred = clf_model.predict(user_X_clf)[0]
# #     user_reg_pred = reg_model.predict(user_X_reg)[0]

# #     # Show output
# #     print(f"\nPredicted Likelihood of Injury: {'Yes' if user_clf_pred == 1 else 'No'}")
# #     print(f"Predicted Recovery Time (in days): {user_reg_pred:.2f}")

# # if __name__ == '__main__':
# #     main()


# import pandas as pd
# import joblib
# from flask import Flask, request, jsonify
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split

# app = Flask(__name__)

# # Load models and preprocessors
# try:
#     clf_model = joblib.load('classifier_model.pkl')
#     reg_model = joblib.load('regressor_model.pkl')
#     preprocessor_clf = joblib.load('preprocessor_clf.pkl')
#     preprocessor_reg = joblib.load('preprocessor_reg.pkl')
# except FileNotFoundError:
#     raise FileNotFoundError("Model or preprocessor files not found. Ensure 'classifier_model.pkl', 'regressor_model.pkl', 'preprocessor_clf.pkl', and 'preprocessor_reg.pkl' are in the project directory.")

# def load_data(file_path):
#     return pd.read_csv(file_path)

# def preprocess_data(df, task='classification'):
#     df = df.copy()
#     df.dropna(inplace=True)
#     if task == 'classification':
#         X = df.drop(['Likelihood_of_Injury', 'Recovery_Time'], axis=1)
#         y = df['Likelihood_of_Injury']
#     else:
#         X = df.drop(['Likelihood_of_Injury', 'Recovery_Time'], axis=1)
#         y = df['Recovery_Time']
#     num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     cat_cols = X.select_dtypes(include=['object']).columns.tolist()
#     num_pipeline = Pipeline([('scaler', StandardScaler())])
#     cat_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
#     preprocessor = ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])
#     X_processed = preprocessor.fit_transform(X)
#     return X_processed, y, preprocessor, num_cols, cat_cols

# def train_model(X, y, task='classification'):
#     if task == 'classification':
#         model = RandomForestClassifier(random_state=42)
#     else:
#         model = RandomForestRegressor(random_state=42)
#     model.fit(X, y)
#     return model

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get JSON data from request
#         data = request.get_json()
        
#         # Validate required fields
#         required_fields = ['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries', 'Training_Intensity']
#         if not all(field in data for field in required_fields):
#             return jsonify({'error': 'Missing required fields'}), 400

#         # Map Training_Intensity to numerical value (adjust based on your CSV data)
#         intensity_map = {'Low': 0.3, 'Medium': 0.6, 'High': 0.9}
#         try:
#             data['Training_Intensity'] = intensity_map[data['Training_Intensity']]
#         except KeyError:
#             return jsonify({'error': 'Invalid Training_Intensity. Use Low, Medium, or High'}), 400

#         # Create DataFrame from input
#         input_data = {
#             'Player_Age': float(data['Player_Age']),
#             'Player_Weight': float(data['Player_Weight']),
#             'Player_Height': float(data['Player_Height']),
#             'Previous_Injuries': int(data['Previous_Injuries']),
#             'Training_Intensity': data['Training_Intensity']
#         }
#         user_df = pd.DataFrame([input_data])

#         # Transform input using preprocessors
#         user_X_clf = preprocessor_clf.transform(user_df)
#         user_X_reg = preprocessor_reg.transform(user_df)

#         # Make predictions
#         clf_pred = clf_model.predict(user_X_clf)[0]
#         reg_pred = reg_model.predict(user_X_reg)[0]

#         # Prepare response
#         response = {
#             'likelihood_of_injury': 'Yes' if clf_pred == 1 else 'No',
#             'recovery_time_days': round(float(reg_pred), 2)
#         }
#         return jsonify(response), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/')
# def home():
#     return jsonify({'message': 'Welcome to the Injury Prediction API'})

# if __name__ == '__main__':
#     # Train models if not already trained
#     try:
#         clf_model
#     except NameError:
#         df = load_data('injury_data.csv')
#         X_clf, y_clf, preprocessor_clf, _, _ = preprocess_data(df, task='classification')
#         clf_model = train_model(X_clf, y_clf, task='classification')
#         joblib.dump(clf_model, 'classifier_model.pkl')
#         joblib.dump(preprocessor_clf, 'preprocessor_clf.pkl')
#         X_reg, y_reg, preprocessor_reg, _, _ = preprocess_data(df, task='regression')
#         reg_model = train_model(X_reg, y_reg, task='regression')
#         joblib.dump(reg_model, 'regressor_model.pkl')
#         joblib.dump(preprocessor_reg, 'preprocessor_reg.pkl')
#     app.run(debug=True, host='0.0.0.0', port=3000)

import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})  # Allow requests from Next.js
# Load models and preprocessors
try:
    clf_model = joblib.load('classifier_model.pkl')
    reg_model = joblib.load('regressor_model.pkl')
    preprocessor_clf = joblib.load('preprocessor_clf.pkl')
    preprocessor_reg = joblib.load('preprocessor_reg.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Model or preprocessor files not found. Ensure 'classifier_model.pkl', 'regressor_model.pkl', 'preprocessor_clf.pkl', and 'preprocessor_reg.pkl' are in the project directory.")
def load_data(file_path):
    return pd.read_csv(file_path)
def preprocess_data(df, task='classification'):
    df = df.copy()
    df.dropna(inplace=True)
    if task == 'classification':
        X = df.drop(['Likelihood_of_Injury', 'Recovery_Time'], axis=1)
        y = df['Likelihood_of_Injury']
    else:
        X = df.drop(['Likelihood_of_Injury', 'Recovery_Time'], axis=1)
        y = df['Recovery_Time']
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_pipeline = Pipeline([('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor, num_cols, cat_cols
def train_model(X, y, task='classification'):
    if task == 'classification':
        model = RandomForestClassifier(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    return model
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries', 'Training_Intensity']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        # Map Training_Intensity to numerical value (adjust based on your CSV data)
        intensity_map = {'Low': 0.3, 'Medium': 0.6, 'High': 0.9}
        try:
            data['Training_Intensity'] = intensity_map[data['Training_Intensity']]
        except KeyError:
            return jsonify({'error': 'Invalid Training_Intensity. Use Low, Medium, or High'}), 400
        # Create DataFrame from input
        input_data = {
            'Player_Age': float(data['Player_Age']),
            'Player_Weight': float(data['Player_Weight']),
            'Player_Height': float(data['Player_Height']),
            'Previous_Injuries': int(data['Previous_Injuries']),
            'Training_Intensity': data['Training_Intensity']
        }
        user_df = pd.DataFrame([input_data])
        # Transform input using preprocessors
        user_X_clf = preprocessor_clf.transform(user_df)
        user_X_reg = preprocessor_reg.transform(user_df)
        # Make predictions
        clf_pred = clf_model.predict(user_X_clf)[0]
        reg_pred = reg_model.predict(user_X_reg)[0]
        # Prepare response
        response = {
            'likelihood_of_injury': 'Yes' if clf_pred == 1 else 'No',
            'recovery_time_days': round(float(reg_pred), 2)
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Injury Prediction API'})
if __name__ == '__main__':
    # Train models if not already trained
    try:
        clf_model
    except NameError:
        df = load_data('injury_data.csv')
        X_clf, y_clf, preprocessor_clf, _, _ = preprocess_data(df, task='classification')
        clf_model = train_model(X_clf, y_clf, task='classification')
        joblib.dump(clf_model, 'classifier_model.pkl')
        joblib.dump(preprocessor_clf, 'preprocessor_clf.pkl')
        X_reg, y_reg, preprocessor_reg, _, _ = preprocess_data(df, task='regression')
        reg_model = train_model(X_reg, y_reg, task='regression')
        joblib.dump(reg_model, 'regressor_model.pkl')
        joblib.dump(preprocessor_reg, 'preprocessor_reg.pkl')
    app.run(debug=True, host='0.0.0.0', port=3000)