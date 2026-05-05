import os
import numpy as np
import tensorflow as tf
from heatMap import get_gradcam_heatmap, display_gradcam, get_last_conv_layer
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras import layers, models

# Import specific preprocessing functions
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_prep
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_prep
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_prep

# --- 1. CONFIGURATION ---
MODEL_DIR = "./Models"  # Update this to your actual model directory
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Update these paths to your actual file names
MODEL_PATHS = {
    "resnet": os.path.join(MODEL_DIR, "ResNet50.keras"),
    "vgg": os.path.join(MODEL_DIR, "VGG16_.keras"),
    "inception": os.path.join(MODEL_DIR, "InceptionV3.keras")
}

# --- 2. ARCHITECTURE REBUILDERS ---
# We rebuild manually to bypass the "quantization_config" serialization error
def build_resnet():
    base = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    return models.Sequential([
        base, layers.GlobalAveragePooling2D(), layers.BatchNormalization(),
        layers.Dense(256, activation='relu'), layers.Dropout(0.4),
        layers.Dense(4, activation='softmax')
    ])

def build_vgg():
    base = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    return models.Sequential([
        base, layers.Flatten(), layers.Dense(512, activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])

def build_inception():
    base = InceptionV3(weights=None, include_top=False, input_shape=(299, 299, 3))
    return models.Sequential([
        base, layers.GlobalAveragePooling2D(), layers.Dense(512, activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Dense(4, activation='softmax')
    ])

# --- 3. SAFE LOADING ENGINE ---
def load_ensemble():
    ensemble = {}
    builders = {"resnet": build_resnet, "vgg": build_vgg, "inception": build_inception}
    
    for name, path in MODEL_PATHS.items():
        print(f"🔄 Loading {name} weights...")
        try:
            # Rebuild architecture and inject weights to avoid metadata errors
            model = builders[name]()
            model.load_weights(path)
            ensemble[name] = model
            print(f"✅ {name.upper()} ready.")
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
    return ensemble

# --- 4. PREPROCESSING & INFERENCE ---
def get_model_prediction(img_path, model, model_name):
    # Handle specific input sizes and scaling
    if model_name == "inception":
        size, prep = (299, 299), inception_prep
    else:
        size, prep = (224, 224), (resnet_prep if model_name == "resnet" else vgg_prep)
    
    img = image.load_img(img_path, target_size=size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = prep(x)
    
    return model.predict(x, verbose=0)

def run_ensemble_inference(img_path, models_dict):
    all_probs = []
    
    print(f"\nAnalyzing MRI: {os.path.basename(img_path)}")
    print("-" * 30)

    for name, model in models_dict.items():
        probs = get_model_prediction(img_path, model, name)
        all_probs.append(probs)
        pred_idx = np.argmax(probs[0])
        print(f"{name.upper():10} | Pred: {CLASSES[pred_idx]:12} | Conf: {probs[0][pred_idx]*100:.2f}%")

    # SOFT VOTING: Average the probabilities across all 3 models
    avg_probs = np.mean(all_probs, axis=0)[0]
    final_idx = np.argmax(avg_probs)
    
    print("-" * 30)
    print(f"FINAL ENSEMBLE DECISION: {CLASSES[final_idx]}")
    print(f"COMBINED CONFIDENCE:     {avg_probs[final_idx]*100:.2f}%")
    print("-" * 30)
    
    return CLASSES[final_idx], avg_probs

def preprocess_for_model(img_path, model_name):
    if model_name == "inception":
        size, prep = (299, 299), inception_prep
    else:
        size, prep = (224, 224), (resnet_prep if model_name == "resnet" else vgg_prep)
    
    img = image.load_img(img_path, target_size=size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return prep(x)

# --- 5. EXECUTION ---
# Load all models into memory once
brain_models = load_ensemble()

# Run a test
test_img = "pituitary.jpeg" # Replace with your image path
if os.path.exists(test_img):
    final_label, probabilities = run_ensemble_inference(test_img, brain_models)
else:
    print("\n⚠️ Please provide a valid path to an MRI image.")




model_name = "resnet"  # Change this to "resnet" or "vgg" to test other models
target_model = brain_models[model_name]

# 2. Preprocess image for that specific model
# (Use your existing get_model_prediction logic to get 'processed_x')
processed_x = preprocess_for_model(test_img, model_name)

# 3. Generate heatmap
layer_name = get_last_conv_layer(model_name)
heatmap = get_gradcam_heatmap(processed_x, target_model, layer_name)

# 4. Show the result
display_gradcam(test_img, heatmap)