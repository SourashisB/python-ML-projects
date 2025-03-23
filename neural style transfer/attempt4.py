import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import time
import os
import argparse
import gradio as gr
from pathlib import Path

# Set TensorFlow to only use necessary GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class StyleTransfer:
    def __init__(self):
        # Load VGG19 model with pre-trained weights
        self.vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        self.vgg.trainable = False
        
        # Define layers for content and style extraction
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        
        # Get the model with outputs for all required layers
        self.model = self.get_model()

    def get_model(self):
        """Creates a model that outputs content and style features."""
        outputs = [self.vgg.get_layer(name).output for name in self.style_layers + self.content_layers]
        model = tf.keras.Model([self.vgg.input], outputs)
        return model

    def load_and_process_image(self, image_path, max_dim=512):
        """Loads and preprocesses images for the VGG19 model."""
        if isinstance(image_path, str):
            # Load from file path
            img = PIL.Image.open(image_path)
        else:
            # Already a PIL Image or similar
            img = image_path
            
        img = img.convert('RGB')  # Ensure 3 channels
        
        # Resize while preserving aspect ratio
        long_dim = max(img.size)
        scale = max_dim / long_dim
        new_width = int(img.size[0] * scale)
        new_height = int(img.size[1] * scale)
        img = img.resize((new_width, new_height), PIL.Image.LANCZOS)
        
        # Convert to numpy array and preprocess
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    def deprocess_image(self, img):
        """Converts processed image back to displayable format."""
        img = img.copy()  # Create a copy to avoid modifying the original
        if len(img.shape) == 4:
            img = np.squeeze(img, axis=0)
            
        # Reverse VGG preprocessing
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68
        
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def gram_matrix(self, tensor):
        """Calculates gram matrix for style representation."""
        # Reshape the tensor: [batch, height, width, channels] -> [batch * height * width, channels]
        batch_size, height, width, channels = tensor.shape
        reshaped = tf.reshape(tensor, (batch_size * height * width, channels))
        
        # Calculate gram matrix
        gram = tf.matmul(tf.transpose(reshaped), reshaped)
        
        # Normalize by number of locations
        num_locations = tf.cast(height * width, tf.float32)
        return gram / num_locations

    def compute_content_loss(self, generated_content, target_content):
        """Computes the content loss between generated and target content features."""
        return tf.reduce_mean(tf.square(generated_content - target_content))

    def compute_style_loss(self, generated_style, target_style):
        """Computes the style loss between generated and target style features."""
        gram_generated = self.gram_matrix(generated_style)
        return tf.reduce_mean(tf.square(gram_generated - target_style))

    def compute_total_loss(self, generated_outputs, content_targets, style_targets, 
                           content_weight=1e4, style_weight=1e-2, total_variation_weight=30):
        """Computes the total loss combining content, style, and total variation losses."""
        style_outputs = generated_outputs[:len(self.style_layers)]
        content_outputs = generated_outputs[len(self.style_layers):]
        
        # Calculate content loss
        content_loss = tf.add_n([self.compute_content_loss(content_outputs[i], content_targets[i]) 
                                for i in range(len(self.content_layers))])
        
        # Calculate style loss
        style_loss = tf.add_n([self.compute_style_loss(style_outputs[i], style_targets[i]) 
                              for i in range(len(self.style_layers))])
        
        # Add total variation loss for smoother results
        def total_variation_loss(image):
            x_var = tf.reduce_sum(tf.square(image[:, 1:, :, :] - image[:, :-1, :, :]))
            y_var = tf.reduce_sum(tf.square(image[:, :, 1:, :] - image[:, :, :-1, :]))
            return x_var + y_var
        
        tv_loss = total_variation_loss(generated_outputs[-1])
        
        # Weight the losses
        content_loss *= content_weight
        style_loss *= style_weight
        tv_loss *= total_variation_weight
        
        # Return total loss
        return content_loss + style_loss + tv_loss, content_loss, style_loss, tv_loss

    @tf.function
    def train_step(self, generated_image, content_targets, style_targets, optimizer, 
                   content_weight=1e4, style_weight=1e-2, total_variation_weight=30):
        """Performs one training step with gradient descent."""
        with tf.GradientTape() as tape:
            outputs = self.model(generated_image)
            total_loss, content_loss, style_loss, tv_loss = self.compute_total_loss(
                outputs, content_targets, style_targets, 
                content_weight, style_weight, total_variation_weight
            )
            
        # Compute and apply gradients
        gradients = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(gradients, generated_image)])
        
        # Clip to valid pixel range
        generated_image.assign(tf.clip_by_value(generated_image, -103.939, 255.0 - 103.939))
        
        return total_loss, content_loss, style_loss, tv_loss

    def style_transfer(self, content_image, style_image, epochs=1000, 
                       content_weight=1e4, style_weight=1e-2, total_variation_weight=30, 
                       learning_rate=0.02, callback=None):
        """Performs neural style transfer between content and style images."""
        # Process input images
        if isinstance(content_image, str):
            content_image = self.load_and_process_image(content_image)
        
        if isinstance(style_image, str):
            style_image = self.load_and_process_image(style_image)
            
        # Initialize the generated image (starting with content image)
        generated_image = tf.Variable(content_image, dtype=tf.float32)
        
        # Extract target feature representations
        content_target_features = self.model(content_image)[len(self.style_layers):]
        style_target_features = [self.gram_matrix(feature) for feature in self.model(style_image)[:len(self.style_layers)]]
        
        # Set up optimizer with learning rate decay
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=100,
            decay_rate=0.96
        )
        optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
        
        # Track best result
        best_loss = float('inf')
        best_img = None
        
        start_time = time.time()
        
        # Optimization loop
        for epoch in range(epochs):
            total_loss, content_loss, style_loss, tv_loss = self.train_step(
                generated_image, content_target_features, style_target_features, 
                optimizer, content_weight, style_weight, total_variation_weight
            )
            
            # Save best result
            if total_loss < best_loss:
                best_loss = total_loss
                best_img = generated_image.numpy()
            
            # Print progress
            if epoch % 50 == 0:
                end_time = time.time()
                print(f"Epoch {epoch}: Total loss {total_loss:.4f}, "
                      f"Content loss {content_loss:.4f}, Style loss {style_loss:.4f}, "
                      f"TV loss {tv_loss:.4f}, Time: {end_time - start_time:.2f}s")
                
                # Callback for UI update if provided
                if callback and epoch % 100 == 0:
                    current_img = self.deprocess_image(generated_image.numpy())
                    callback(epoch, epochs, current_img)
        
        # Return the best result
        final_image = self.deprocess_image(best_img)
        return final_image

def create_gradio_interface():
    """Creates a Gradio interface for the style transfer app."""
    # Initialize the style transfer model
    st = StyleTransfer()
    
    def process_images(content_img, style_img, content_weight, style_weight, tv_weight, iterations, progress=gr.Progress()):
        """Process function for Gradio interface."""
        # Preprocess images
        content_tensor = st.load_and_process_image(content_img)
        style_tensor = st.load_and_process_image(style_img)
        
        # Callback function to update progress bar
        def update_progress(epoch, total_epochs, current_img):
            progress((epoch + 1) / total_epochs, f"Iteration {epoch+1}/{total_epochs}")
            
        # Perform style transfer
        progress(0, "Starting style transfer...")
        result = st.style_transfer(
            content_tensor, style_tensor,
            epochs=iterations,
            content_weight=float(content_weight),
            style_weight=float(style_weight),
            total_variation_weight=float(tv_weight),
            callback=update_progress
        )
        
        # Return the result as PIL Image
        return PIL.Image.fromarray(result)
    
    # Create Gradio interface
    demo = gr.Interface(
        fn=process_images,
        inputs=[
            gr.Image(type="pil", label="Content Image"),
            gr.Image(type="pil", label="Style Image"),
            gr.Slider(minimum=1, maximum=1e5, value=1e4, label="Content Weight", step=100),
            gr.Slider(minimum=1e-4, maximum=1, value=1e-2, label="Style Weight", step=0.001),
            gr.Slider(minimum=0, maximum=100, value=30, label="Total Variation Weight", step=1),
            gr.Slider(minimum=100, maximum=2000, value=500, label="Iterations", step=100)
        ],
        outputs=gr.Image(type="pil", label="Stylized Result"),
        title="Neural Style Transfer",
        description="Upload a content image and a style image to generate a new artistic image.",
        examples=[
            ["examples/content1.jpg", "examples/style1.jpg"],
            ["examples/content2.jpg", "examples/style2.jpg"],
        ]
    )
    return demo

def main():
    """Main function to run the style transfer script or launch the Gradio interface."""
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--mode', type=str, default='app', choices=['app', 'script'], 
                        help='Run mode: "app" for Gradio interface, "script" for CLI execution')
    parser.add_argument('--content', type=str, help='Path to content image')
    parser.add_argument('--style', type=str, help='Path to style image')
    parser.add_argument('--output', type=str, default='output.png', help='Path to output image')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--content-weight', type=float, default=1e4, help='Content weight')
    parser.add_argument('--style-weight', type=float, default=1e-2, help='Style weight')
    parser.add_argument('--tv-weight', type=float, default=30, help='Total variation weight')
    
    args = parser.parse_args()
    
    # Create examples directory if it doesn't exist
    if not os.path.exists('examples'):
        os.makedirs('examples')
    
    if args.mode == 'app':
        # Launch Gradio interface
        demo = create_gradio_interface()
        demo.launch(share=True)
    else:
        # Run style transfer as script
        if not args.content or not args.style:
            parser.error("--content and --style arguments are required in script mode")
        
        st = StyleTransfer()
        output_image = st.style_transfer(
            args.content, args.style, 
            epochs=args.epochs,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            total_variation_weight=args.tv_weight
        )
        
        # Save output image
        PIL.Image.fromarray(output_image).save(args.output)
        print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()