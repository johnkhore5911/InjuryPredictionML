import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

def train_regression_models(X_train, y_train):
    """Train multiple regression models and return the best one"""
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'SVR': SVR(),
        'Linear Regression': LinearRegression(),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge Regression': Ridge(random_state=42)
    }
    
    # Hyperparameter grids
    param_grids = {
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
        'SVR': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
        'Ridge Regression': {'alpha': [0.1, 1, 10]}
    }
    
    best_model = None
    best_score = float('-inf')
    model_results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Skip grid search for Linear Regression (no hyperparameters)
        if name == 'Linear Regression':
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
            model_results[name] = {
                'model': model,
                'best_params': {},
                'best_score': score
            }
        else:
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='r2')
            grid_search.fit(X_train, y_train)
            
            model_results[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
        
        # Update best model (higher R2 is better)
        if model_results[name]['best_score'] > best_score:
            best_score = model_results[name]['best_score']
            best_model = model_results[name]['model']
    
    return best_model, model_results

def evaluate_regression_model(model, X_test, y_test):
    """Evaluate the regression model and return metrics"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Regression Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2
    }

def save_regression_model(model, filepath):
    """Save the trained regression model"""
    joblib.dump(model, filepath)