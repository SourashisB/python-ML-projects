import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import time

# Load VGG19 model
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Function to load and process images
def load_and_process_image(image_path):
    max_dim = 512
    img = PIL.Image.open(image_path)
    img = img.convert('RGB')  # Ensure 3 channels
    img = img.resize((max_dim, max_dim), PIL.Image.LANCZOS)
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

# Function to deprocess image
def deprocess_image(img):
    img = img.reshape((img.shape[1], img.shape[2], 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# Layers for content and style extraction
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Function to get feature representations
def get_model():
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# Function to compute content loss
def compute_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# Function to compute style loss
def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    matrix = tf.reshape(tensor, [-1, channels])
    return tf.matmul(tf.transpose(matrix), matrix) / tf.cast(tf.shape(matrix)[0], tf.float32)

def compute_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

# Compute total loss
def compute_loss(model, generated_image, content_targets, style_targets, content_weight=1e4, style_weight=1e-2):
    outputs = model(generated_image)
    style_outputs, content_outputs = outputs[:len(style_layers)], outputs[len(style_layers):]
    
    content_loss = tf.add_n([compute_content_loss(content_outputs[i], content_targets[i]) for i in range(len(content_layers))])
    style_loss = tf.add_n([compute_style_loss(style_outputs[i], style_targets[i]) for i in range(len(style_layers))])
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    return total_loss

# Gradient descent function
@tf.function
def train_step(model, generated_image, content_targets, style_targets, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, generated_image, content_targets, style_targets)
    gradients = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(gradients, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, -103.939, 255.0 - 103.939))

# Style transfer function
def style_transfer(content_path, style_path, epochs=1000, learning_rate=5.0):
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)
    
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    
    model = get_model()
    content_targets = model(content_image)[len(style_layers):]
    style_targets = [gram_matrix(feature) for feature in model(style_image)[:len(style_layers)]]
    
    optimizer = tf.optimizers.Adam(learning_rate)
    
    for epoch in range(epochs):
        train_step(model, generated_image, content_targets, style_targets, optimizer)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}")
    
    final_image = deprocess_image(generated_image.numpy())
    return final_image

# Run the style transfer
content_image_path = "./image1.png"
style_image_path = "./image-2.jpg"

output_image = style_transfer(content_image_path, style_image_path)

# Display the final output
plt.imshow(output_image)
plt.axis('off')
plt.show()