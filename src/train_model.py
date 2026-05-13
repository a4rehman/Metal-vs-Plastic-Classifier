import os
import yaml
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_model(input_shape, num_classes):
    params = load_params()
    
    model = Sequential()
    
    # Add convolutional layers from params
    for i, layer_config in enumerate(params['model']['conv_layers']):
        if i == 0:
            # First layer needs input shape
            model.add(Conv2D(
                filters=layer_config['filters'],
                kernel_size=layer_config['kernel_size'],
                activation=params['model']['activation'],
                input_shape=input_shape
            ))
        else:
            model.add(Conv2D(
                filters=layer_config['filters'],
                kernel_size=layer_config['kernel_size'],
                activation=params['model']['activation']
            ))
        model.add(MaxPooling2D(2, 2))
    
    # Add fully connected layers
    model.add(Flatten())
    model.add(Dense(params['model']['dense_units'], activation=params['model']['activation']))
    model.add(Dropout(params['model']['dropout_rate']))
    model.add(Dense(num_classes, activation=params['model']['output_activation']))
    
    return model

def main():
    params = load_params()
    
    # Load training data
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        params['data']['base_dir'],
        image_size=tuple(params['data']['image_size']),
        batch_size=params['data']['batch_size'],
        validation_split=params['data']['validation_split'],
        subset='training',
        seed=params['data']['random_seed']
    )
    
    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        params['data']['base_dir'],
        image_size=tuple(params['data']['image_size']),
        batch_size=params['data']['batch_size'],
        validation_split=params['data']['validation_split'],
        subset='validation',
        seed=params['data']['random_seed']
    )
    
    # Create model
    input_shape = tuple(params['data']['image_size']) + (3,)
    num_classes = len(train_data.class_names)
    
    model = create_model(input_shape, num_classes)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=params['training']['learning_rate']),
        loss=params['training']['loss'],
        metrics=params['training']['metrics']
    )
    
    # Train model
    history = model.fit(
        train_data,
        epochs=params['training']['epochs'],
        validation_data=val_data,
        batch_size=params['training']['batch_size']
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/garbage_classifier.h5')
    
    # Save training history
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    print("Model training completed!")
    print(f"Model saved as: models/garbage_classifier.h5")

if __name__ == "__main__":
    main()