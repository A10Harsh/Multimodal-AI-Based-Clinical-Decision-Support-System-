import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Access the base model (ResNet/VGG/Inception) inside your Sequential wrapper
    base_model = model.layers[0]
    
    # 2. Re-run the image through the model once to initialize the graph
    # This "calls" the model and defines the output tensors
    preds = model(img_array)
    
    # 3. Create a model that maps the BASE model's input to the LAST CONV LAYER output
    # We do this separately to avoid the "Sequential has no output" error
    inner_grad_model = tf.keras.models.Model(
        [base_model.input], 
        [base_model.get_layer(last_conv_layer_name).output]
    )

    # 4. Use GradientTape to track the flow
    with tf.GradientTape() as tape:
        # Get the convolutional output from the base model
        last_conv_layer_output = inner_grad_model(img_array)
        
        # Now get the final prediction from the full model
        # We watch the conv output to see how it affects the final prediction
        tape.watch(last_conv_layer_output)
        
        # We need to run the rest of the Sequential model (the top layers)
        # on the conv output. In your case, that's layers 1 to the end.
        x = last_conv_layer_output
        for i in range(1, len(model.layers)):
            x = model.layers[i](x)
        
        if pred_index is None:
            pred_index = tf.argmax(x[0])
        class_channel = x[:, pred_index]

    # 5. Compute gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # 6. Global Average Pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 7. Create the heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 8. Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Overlays the heatmap onto the original image and displays it.
    """
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Resize heatmap to match original image size
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(jet_heatmap * 255)

    # Create the superimposed image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.uint8(superimposed_img)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM: Tumor Localization")
    plt.axis("off")
    plt.show()

# --- Utility to get the right layer name ---
def get_last_conv_layer(model_name):
    layers = {
        "resnet": "conv5_block3_out",
        "vgg": "block5_conv3",
        "inception": "mixed10"
    }
    return layers.get(model_name.lower(), None)