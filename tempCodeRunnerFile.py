import pandas as pd
from src.data_preprocessing import load_data, preprocess_data
from src.classification import train_classification_models, evaluate_classification_model, save_classification_model
from src.regression import train_regression_models, evaluate_regression_model, save_regression_model
import joblib
import os

def main():
    # Create directories if they don't exist
    os.makedirs('models/classification', exist_ok=True)
    os.makedirs('models/regression', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load data
    df = load_data('data/injury_data.csv')
    
    # Classification task (Likelihood_of_Injury)
    print("\n" + "="*50)
    print("CLASSIFICATION TASK: Likelihood_of_Injury")
    print("="*50)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf, preprocessor_clf = preprocess_data(
        df, 'Likelihood_of_Injury', 'classification')
    
    # Train and evaluate classification models
    best_clf, clf_results = train_classification_models(X_train_clf, y_train_clf)
    clf_metrics = evaluate_classification_model(best_clf, X_test_clf, y_test_clf)
    
    # Save best classification model and metrics
    save_classification_model(best_clf, 'models/classification/best_classifier.pkl')
    with open('models/classification/classification_report.txt', 'w') as f:
        f.write(clf_metrics['classification_report'])
    
    # Regression task (Recovery_Time)
    print("\n" + "="*50)
    print("REGRESSION TASK: Recovery_Time")
    print("="*50)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, preprocessor_reg = preprocess_data(
        df, 'Recovery_Time', 'regression')
    
    # Train and evaluate regression models
    best_reg, reg_results = train_regression_models(X_train_reg, y_train_reg)
    reg_metrics = evaluate_regression_model(best_reg, X_test_reg, y_test_reg)
    
    # Save best regression model and metrics
    save_regression_model(best_reg, 'models/regression/best_regressor.pkl')
    with open('models/regression/regression_metrics.txt', 'w') as f:
        f.write(f"R2 Score: {reg_metrics['r2_score']:.4f}\n")
        f.write(f"RMSE: {reg_metrics['rmse']:.4f}\n")
        f.write(f"MAE: {reg_metrics['mae']:.4f}\n")
    
    # Generate predictions for new data (using test set as example)
    print("\n" + "="*50)
    print("GENERATING PREDICTIONS")
    print("="*50)
    
    # Load models
    clf_model = joblib.load('models/classification/best_classifier.pkl')
    reg_model = joblib.load('models/regression/best_regressor.pkl')
    
    # Make predictions
    clf_predictions = clf_model.predict(X_test_clf)
    reg_predictions = reg_model.predict(X_test_reg)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'Player_Age': X_test_reg[:, 0],  # Assuming age is first column
        'Player_Weight': X_test_reg[:, 1],
        'Player_Height': X_test_reg[:, 2],
        'Previous_Injuries': X_test_reg[:, 3],
        'Training_Intensity': X_test_reg[:, 4],
        'Actual_Recovery_Time': y_test_reg,
        'Predicted_Recovery_Time': reg_predictions,
        'Actual_Likelihood_of_Injury': y_test_clf,
        'Predicted_Likelihood_of_Injury': clf_predictions
    })
    
    # Save predictions to CSV
    predictions_df.to_csv('data/predictions.csv', index=False)
    print("Predictions saved to data/predictions.csv")

if __name__ == "__main__":
    main()