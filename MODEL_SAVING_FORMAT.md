# Model Saving Format - Updated

## Overview
The trained models are now saved in an improved `.pkl` format that includes comprehensive metrics alongside the model pipeline.

## File Location
Models are saved to: `Trained_Models/{person_name}.pkl`

Example: `Trained_Models/John_Doe.pkl`

## Pickle File Structure
Each `.pkl` file contains a Python dictionary with the following structure:

```python
{
    'model': Pipeline(steps=[
        ('scaler', StandardScaler()), 
        ('pca', PCA(n_components=0.85)),
        ('classifier', RandomForestClassifier(...))
    ]),
    
    # Training Metrics
    'train_accuracy': float,           # Training accuracy score
    'train_Recall': float,             # Training recall (sensitivity)
    'train_precision': float,          # Training precision
    'train_f1': float,                 # Training F1 score
    'train_FAR': float,                # Training False Acceptance Rate
    'train_FRR': float,                # Training False Rejection Rate
    'train_EER': float,                # Training Equal Error Rate
    
    # Testing Metrics
    'test_accuracy': float,            # Test accuracy score (PRIMARY METRIC FOR BEST MODEL SELECTION)
    'test_Recall': float,              # Test recall (sensitivity)
    'test_precision': float,           # Test precision
    'test_f1': float,                  # Test F1 score
    'test_FAR': float,                 # Test False Acceptance Rate
    'test_FRR': float,                 # Test False Rejection Rate
    'test_EER': float,                 # Test Equal Error Rate
    
    # Model Information
    'best_params': dict,               # Best hyperparameters from GridSearchCV
    'all_results': dict                # Comprehensive results from all models trained
}
```

## Model Selection Strategy
- The **best model** is selected based on **highest test accuracy**
- All metrics (train and test) are stored for reference
- The model pipeline can be directly used for inference

## Usage Example

```python
import pickle
from pathlib import Path

# Load a saved model
models_dir = Path("Trained_Models")
pkl_path = models_dir / "John_Doe.pkl"

with open(pkl_path, "rb") as f:
    model_data = pickle.load(f)

# Extract model and metrics
model_pipeline = model_data['model']
test_accuracy = model_data['test_accuracy']
test_recall = model_data['test_Recall']
train_accuracy = model_data['train_accuracy']

print(f"Model Test Accuracy: {test_accuracy:.4f}")
print(f"Model Test Recall: {test_recall:.4f}")

# Use model for predictions
prediction = model_pipeline.predict(features)
```

## Changes Made
Modified `src/Models/Person_model_saves.py`:
- Updated the best model saving logic (lines 555-579)
- Individual metrics are now stored at the dictionary top level
- All historical results are preserved in the 'all_results' key
- Format ensures backward compatibility with existing inference code

## Backward Compatibility
✅ The existing inference code in `src/model_output.py` is fully compatible
- The `load_person_model()` function correctly extracts the 'model' key
- The model can be directly used for predictions without modifications

