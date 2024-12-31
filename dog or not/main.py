import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os

# Function to load and preprocess the image
def load_image(img_path, target_size=(224, 224)):
    """Loads and preprocesses an image for the MobileNetV2 model."""
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

# Function to determine if the image is a dog
def is_dog(image_path, model):
    """Predicts if the given image contains a dog."""
    img_array = load_image(image_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=5)[0]  # Get top 5 predictions

    print("\nTop Predictions:")
    for pred in decoded_predictions:
        print(f"{pred[1]}: {pred[2]*100:.2f}%")

    # Check if any of the predictions correspond to a dog class
    dog_classes = [
        'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu',
        'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Rhodesian_ridgeback',
        'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick',
        'black-and-tan_coonhound', 'Walker_hound', 'English_foxhound',
        'redbone', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound',
        'whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound',
        'Saluki', 'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier',
        'American_Staffordshire_terrier', 'Bedlington_terrier', 'Border_terrier',
        'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier', 'Norwich_terrier',
        'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier',
        'Sealyham_terrier', 'Airedale', 'cairn', 'Australian_terrier',
        'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer', 'giant_schnauzer',
        'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier',
        'silky_terrier', 'soft-coated_wheaten_terrier', 'West_Highland_white_terrier',
        'Lhasa', 'flat-coated_retriever', 'curly-coated_retriever', 'golden_retriever',
        'Labrador_retriever', 'Chesapeake_Bay_retriever', 'German_short-haired_pointer',
        'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter',
        'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel',
        'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz',
        'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor',
        'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 'Border_collie',
        'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd', 'Doberman',
        'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog',
        'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff',
        'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog', 'malamute',
        'Siberian_husky', 'dalmatian', 'affenpinscher', 'basenji', 'pug', 'Leonberg',
        'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow',
        'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle',
        'miniature_poodle', 'standard_poodle', 'Mexican_hairless', 'dingo',
        'dhole', 'African_hunting_dog'
    ]

    for pred in decoded_predictions:
        if pred[1] in dog_classes:
            print("\n**This is a dog!**")
            return True

    print("\n**This is NOT a dog.**")
    return False

# Main function to test the project
def main():
    """Main function to run the dog detection script."""
    if len(sys.argv) < 2:
        print("Usage: python dog_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        sys.exit(1)

    # Load the pre-trained MobileNetV2 model
    print("Loading model...")
    model = MobileNetV2(weights="imagenet")

    # Check if the image is a dog
    print(f"Analyzing image: {image_path}")
    is_dog(image_path, model)

if __name__ == "__main__":
    main()