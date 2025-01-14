import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from keras.api.applications import vgg19
from keras.api.preprocessing.image import load_img, img_to_array
from keras.api.mixed_precision import set_global_policy
import os

set_global_policy('mixed_float16')


# Load and preprocess images
def load_and_process_image(image_path, target_size=(512, 512)):
    """Loads an image, resizes it to a fixed size, and preprocesses it for VGG19."""
    img = load_img(image_path, target_size=target_size)  # Resize image to fixed shape
    img = img_to_array(img)  # Convert image to NumPy array
    img = tf.convert_to_tensor(img, dtype=tf.float32)  # Convert to TensorFlow tensor
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    img = vgg19.preprocess_input(img)  # Preprocess for VGG19
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
    """Computes the Gram matrix for a given tensor."""
    # Ensure the tensor is in float32
    tensor = tf.cast(tensor, tf.float32)
    # Compute the Gram matrix
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_elements = tf.cast(input_shape[1] * input_shape[2], tf.float32)  # height * width
    return result / num_elements

def compute_style_loss(base_style, gram_target):
    """
    Computes the style loss between the style image and the generated image.
    
    Args:
        base_style (Tensor): Feature map of the generated image.
        gram_target (Tensor): Precomputed Gram matrix of the style image.
    
    Returns:
        Tensor: Style loss value.
    """
    gram_style = gram_matrix(base_style)  # Compute Gram matrix for the generated image
    return tf.reduce_mean(tf.square(gram_style - gram_target))  # Mean square error

# Neural Style Transfer class
class NeuralStyleTransfer:
    def __init__(self, content_path, style_path, content_layers, style_layers, target_size=(512, 512)):
        """Initialize the NeuralStyleTransfer model."""
        # Load and preprocess images
        self.content_image = load_and_process_image(content_path, target_size=target_size)
        self.style_image = load_and_process_image(style_path, target_size=target_size)

        # Store layer information
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)

        # Build the model to extract features
        self.model = build_vgg_model(style_layers + content_layers)

        # Extract features for style and content images
        style_features = self.extract_features(self.style_image)
        content_features = self.extract_features(self.content_image)

        # Precompute Gram matrices for style features
        self.style_targets = {
            layer: tf.cast(gram_matrix(style_features['style'][layer]), tf.float32)
            for layer in style_layers
        }
        self.content_targets = content_features['content']

        # Initialize the generated image as a trainable variable
        self.generated_image = tf.Variable(tf.convert_to_tensor(self.content_image), dtype=tf.float32)

    def extract_features(self, image):
        """Extract style and content features from an image using the VGG19 model."""
        outputs = self.model(image)
        style_features = outputs[:len(self.style_layers)]
        content_features = outputs[len(self.style_layers):]
        style_dict = {name: value for name, value in zip(self.style_layers, style_features)}
        content_dict = {name: value for name, value in zip(self.content_layers, content_features)}
        return {'style': style_dict, 'content': content_dict}

    def compute_loss(self):
        outputs = self.model(self.generated_image)
        style_outputs = outputs[:len(self.style_layers)]
        content_outputs = outputs[len(self.style_layers):]

        # Compute style loss
        style_loss = 0
        for style_output, target_style in zip(style_outputs, self.style_targets.values()):
            style_output = tf.cast(style_output, tf.float32)
            target_style = tf.cast(target_style, tf.float32)
            style_loss += compute_style_loss(style_output, target_style)
        style_loss /= self.num_style_layers

        # Compute content loss
        content_loss = 0
        for content_output, target_content in zip(content_outputs, self.content_targets.values()):
            content_output = tf.cast(content_output, tf.float32)
            target_content = tf.cast(target_content, tf.float32)
            content_loss += compute_content_loss(content_output, target_content)
        content_loss /= self.num_content_layers

        # Adjust the weights of style and content losses
        style_weight = 1e-1  # Increase the style weight
        content_weight = 1e3  # Decrease the content weight
        total_loss = style_weight * style_loss + content_weight * content_loss
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
            # Cast loss to float32 (if necessary)
            loss = tf.cast(loss, tf.float32)
            grads = tape.gradient(loss, self.generated_image)
            # Apply gradients
            optimizer.apply_gradients([(grads, self.generated_image)])
            # Ensure the generated image remains in the valid range
            self.generated_image.assign(tf.clip_by_value(self.generated_image, -1.0, 1.0))
            return loss

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
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    # File paths for content and style images
    content_path = "./neural style transfer/image1.png"
    style_path = "./neural style transfer/image-2.jpg"

    # Define the layers for style and content extraction
    content_layers = ['block4_conv2']  # Higher-level content layer
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'] # Fewer style layers

    # Instantiate and train the NST model
        # Instantiate and train the NST model
    nst = NeuralStyleTransfer(
        content_path,
        style_path,
        content_layers,
        style_layers,
        target_size=(512, 512)  # Smaller image size
)
    nst.train(epochs=20, steps_per_epoch=200, learning_rate=0.01)    

    # Display the result
    result = nst.get_result()
    plt.imshow(result)
    plt.axis('off')
    plt.show()