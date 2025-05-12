import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier

def train_classification_models(X_train, y_train):
    """Train multiple classification models and return the best one"""
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Hyperparameter grids
    param_grids = {
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'Logistic Regression': {'C': [0.1, 1, 10]},
        'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
    }
    
    best_model = None
    best_score = 0
    model_results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='f1')
        grid_search.fit(X_train, y_train)
        
        # Store results
        model_results[name] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        # Update best model
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
    
    return best_model, model_results

def evaluate_classification_model(model, X_test, y_test):
    """Evaluate the classification model and return metrics"""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("Classification Report:")
    print(report)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'classification_report': report,
        'accuracy': accuracy,
        'f1_score': f1
    }

def save_classification_model(model, filepath):
    """Save the trained classification model"""
    joblib.dump(model, filepath)