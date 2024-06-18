import tensorflow as tf

def convert_to_tflite(h5_model_path, tflite_model_path):
    """
    Convert a trained TensorFlow model (.h5 format) to TensorFlow Lite format (.tflite).

    Parameters:
    - h5_model_path (str): File path to the trained TensorFlow model in .h5 format.
    - tflite_model_path (str): File path to save the converted TensorFlow Lite model in .tflite format.
    """
    # Load the trained TensorFlow model from .h5 file
    model = tf.keras.models.load_model(h5_model_path)

    # Convert the TensorFlow model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the converted model to a .tflite file
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    # Print confirmation message
    print(f"Model converted to TFLite format and saved to: {tflite_model_path}")

# Example usage:
h5_model_path = 'final_model.h5'         # Path to the trained Keras model (.h5 format)
tflite_model_path = 'final_model.tflite' # Path to save the converted TensorFlow Lite model (.tflite format)
convert_to_tflite(h5_model_path, tflite_model_path)
