import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load your trained model
model = tf.keras.models.load_model("fine_tuned_flood_detection_model")

# Define function to preprocess image


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

# Define function to make prediction


def predict_image(img_path):
    preprocessed_image = preprocess_image(img_path)
    predictions = model.predict(preprocessed_image)
    result = np.argmax(predictions)
    return result

# Function to interpret prediction result


def interpret_result(prediction):
    if prediction == 0:
        return "The image contains flooding."
    elif prediction == 1:
        return "The image does not contain flooding."
    else:
        return "Invalid prediction."


# Test the function with an image
image_path = r"Samples\6.jpg"
prediction = predict_image(image_path)
result_text = interpret_result(prediction)
print(result_text)
