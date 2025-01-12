import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from keras.api.applications import vgg19
from keras.api.preprocessing.image import load_img, img_to_array

import os


# Load and preprocess images
def load_and_process_image(image_path, target_size=(512, 512)):
    """Loads an image, resizes it to a fixed size, and preprocesses it for VGG19."""
    img = load_img(image_path, target_size=target_size)  # Resize image to a fixed shape
    img = img_to_array(img)  # Convert to NumPy array

    # Convert to TensorFlow tensor
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    # Add batch dimension
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    # Preprocess for VGG19
    img = vgg19.preprocess_input(img)  # Preprocess (BGR format and mean subtraction)
    return img

def deprocess_image(processed_img):
    """Converts a processed image back to a format displayable by matplotlib."""
    x = processed_img.copy()
    # Reverse preprocessing steps
    x[:, :, 0] += 103.939  # Add back mean for R
    x[:, :, 1] += 116.779  # Add back mean for G
    x[:, :, 2] += 123.68   # Add back mean for B
    x = x[:, :, ::-1]  # Convert BGR to RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Define the model
def build_vgg_model(layer_names):
    """Creates a VGG19 model that outputs the specified layers."""
    vgg = vgg19.VGG19(weights='imagenet', include_top=False)
    vgg.trainable = False  # Freeze the model
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = Model([vgg.input], outputs)
    return model

# Loss functions
def compute_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(tensor):
    """Computes the Gram matrix, used for style representation."""
    channels = int(tensor.shape[-1])
    vectorized = tf.reshape(tensor, [-1, channels])
    gram = tf.matmul(tf.transpose(vectorized), vectorized)
    return gram

def compute_style_loss(base_style, gram_target):
    """Computes the style loss using the Gram matrix."""
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

# Neural Style Transfer class
class NeuralStyleTransfer:
    def __init__(self, content_path, style_path, content_layers, style_layers, target_size=(512, 512)):
        """
        Initialize the NeuralStyleTransfer model.
        
        Args:
            content_path (str): Path to the content image.
            style_path (str): Path to the style image.
            content_layers (list): List of content layer names.
            style_layers (list): List of style layer names.
            target_size (tuple): Target size to resize both images (height, width).
        """
        # Load and preprocess the content and style images
        self.content_image = load_and_process_image(content_path, target_size=target_size)
        self.style_image = load_and_process_image(style_path, target_size=target_size)

        # Store layer information
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)

        # Build the model to extract features
        self.model = build_vgg_model(style_layers + content_layers)

        # Extract target features from the content and style images
        self.style_targets = self.extract_features(self.style_image)['style']
        self.content_targets = self.extract_features(self.content_image)['content']

        # Initialize the generated image as a trainable variable
        self.generated_image = tf.Variable(tf.convert_to_tensor(self.content_image), dtype=tf.float32)

    def extract_features(self, image):
        """
        Extract style and content features from an image using the VGG19 model.
        
        Args:
            image (Tensor): Preprocessed image tensor.
        
        Returns:
            dict: Dictionary with 'style' and 'content' feature maps.
        """
        outputs = self.model(image)
        style_features = outputs[:len(self.style_layers)]
        content_features = outputs[len(self.style_layers):]
        style_dict = {name: value for name, value in zip(self.style_layers, style_features)}
        content_dict = {name: value for name, value in zip(self.content_layers, content_features)}
        return {'style': style_dict, 'content': content_dict}

    def compute_loss(self):
        """
        Compute the total loss for the style transfer process.
        
        Returns:
            Tensor: Total loss value.
        """
        outputs = self.model(self.generated_image)
        style_outputs = outputs[:len(self.style_layers)]
        content_outputs = outputs[len(self.style_layers):]

        # Compute style loss
        style_loss = 0
        for style_output, target_style in zip(style_outputs, self.style_targets.values()):
            style_loss += compute_style_loss(style_output, target_style)
        style_loss /= self.num_style_layers

        # Compute content loss
        content_loss = 0
        for content_output, target_content in zip(content_outputs, self.content_targets.values()):
            content_loss += compute_content_loss(content_output, target_content)
        content_loss /= self.num_content_layers

        # Combine the losses
        total_loss = 1e-4 * style_loss + 1e4 * content_loss
        return total_loss

    def train(self, epochs=10, steps_per_epoch=100, learning_rate=0.02):
        """
        Train the model to optimize the generated image to minimize the total loss.
        
        Args:
            epochs (int): Number of epochs to train.
            steps_per_epoch (int): Number of steps per epoch.
            learning_rate (float): Learning rate for the optimizer.
        """
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                loss = self.compute_loss()
            grads = tape.gradient(loss, self.generated_image)
            optimizer.apply_gradients([(grads, self.generated_image)])
            self.generated_image.assign(tf.clip_by_value(self.generated_image, -1.0, 1.0))

        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                train_step()
            print(f"Epoch {epoch + 1}/{epochs} completed")

    def get_result(self):
        """
        Get the generated image as a NumPy array.
        
        Returns:
            ndarray: Deprocessed image array for visualization.
        """
        return deprocess_image(self.generated_image.numpy()[0])
    
    
# Main
if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    # File paths for content and style images
    content_path = "neural style transfer/image1.png"
    style_path = "neural style transfer/image-2.jpg"

    # Define the layers for style and content extraction
    content_layers = ['block5_conv2']  # Content layer
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]  # Style layers

    # Instantiate and train the NST model
        # Instantiate and train the NST model
    nst = NeuralStyleTransfer(
        content_path,
        style_path,
        content_layers,
        style_layers,
        target_size=(512, 512)
    )
    nst.train(epochs=10, steps_per_epoch=100, learning_rate=0.02)

    # Display the result
    result = nst.get_result()
    plt.imshow(result)
    plt.axis('off')
    plt.show()