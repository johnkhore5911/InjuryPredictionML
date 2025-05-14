import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

# Load data and train models
df = load_data('injury_data.csv')
X_clf, y_clf, preprocessor_clf, _, _ = preprocess_data(df, task='classification')
clf_model = train_model(X_clf, y_clf, task='classification')
joblib.dump(clf_model, 'classifier_model.pkl')
joblib.dump(preprocessor_clf, 'preprocessor_clf.pkl')
X_reg, y_reg, preprocessor_reg, _, _ = preprocess_data(df, task='regression')
reg_model = train_model(X_reg, y_reg, task='regression')
joblib.dump(reg_model, 'regressor_model.pkl')
joblib.dump(preprocessor_reg, 'preprocessor_reg.pkl')
print("Models and preprocessors saved successfully.")