import os
import json
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    
    # Load model
    model = load_model('models/garbage_classifier.h5')
    
    # Load test data
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        params['data']['base_dir'],
        image_size=tuple(params['data']['image_size']),
        batch_size=params['data']['batch_size'],
        validation_split=params['evaluation']['test_split'],
        subset='validation',
        seed=params['data']['random_seed']
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_data)
    
    # Make predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_data:
        predictions = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Save metrics
    metrics = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save confusion matrix plot data
    os.makedirs('plots', exist_ok=True)
    cm_data = {
        'confusion_matrix': cm.tolist(),
        'classes': test_data.class_names
    }
    
    with open('plots/confusion_matrix.json', 'w') as f:
        json.dump(cm_data, f, indent=2)
    
    print("Model evaluation completed!")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()