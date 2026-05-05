import streamlit as st
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Import your custom logic from existing files
from main import load_ensemble, get_model_prediction, CLASSES
from heatMap import get_gradcam_heatmap, get_last_conv_layer, display_gradcam

# --- PAGE CONFIG ---
st.set_page_config(page_title="Brain Tumor Diagnostic System", layout="wide")

st.title("🧠 Brain Tumor MRI Analysis System")
st.markdown("""
This system uses an **Ensemble of Deep Learning Models** (ResNet50, VGG16, InceptionV3) 
to detect brain tumors and provides **Grad-CAM** heatmaps for clinical explainability.
""")

# --- LOAD MODELS ---
@st.cache_resource
def init_models():
    return load_ensemble()

with st.spinner("Loading Deep Learning Models... Please wait."):
    models_dict = init_models()

# --- SIDEBAR: IMAGE UPLOAD ---
st.sidebar.header("Upload MRI Image")
uploaded_file = st.sidebar.file_uploader("Choose a JPEG/PNG image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Create two columns for the UI
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Original MRI Scan")
        st.image(uploaded_file, width='stretch')

    # --- INFERENCE ---
    with st.spinner("Analyzing MRI..."):
        all_probs = []
        results = {}

        for name, model in models_dict.items():
            # Standardizing input and getting prediction
            if name == "inception":
                size = (299, 299)
                from tensorflow.keras.applications.inception_v3 import preprocess_input as prep
            elif name == "resnet":
                size = (224, 224)
                from tensorflow.keras.applications.resnet50 import preprocess_input as prep
            else:
                size = (224, 224)
                from tensorflow.keras.applications.vgg16 import preprocess_input as prep

            # Preprocess
            img = Image.open(temp_path).convert('RGB').resize(size)
            x = np.array(img)
            x = np.expand_dims(x, axis=0)
            x = prep(x.astype('float32'))
            
            # Predict
            probs = model.predict(x, verbose=0)
            all_probs.append(probs)
            results[name] = {"probs": probs[0], "processed_x": x}

        # Ensemble Soft Voting
        avg_probs = np.mean(all_probs, axis=0)[0]
        final_idx = np.argmax(avg_probs)
        final_label = CLASSES[final_idx]
        final_conf = avg_probs[final_idx]

    with col2:
        st.subheader("Diagnostic Results")
        
        # Display Final Result
        st.success(f"**Ensemble Prediction: {final_label}**")
        st.info(f"**Confidence Score: {final_conf*100:.2f}%**")

        # Display individual model breakdown in an expander
        with st.expander("View Individual Model Predictions"):
            for name in results:
                p = results[name]["probs"]
                idx = np.argmax(p)
                st.write(f"**{name.upper()}:** {CLASSES[idx]} ({p[idx]*100:.2f}%)")

    # --- GRAD-CAM VISUALIZATION ---
    st.divider()
    st.subheader("🔍 Grad-CAM Tumor Localization")
    st.write("The heatmap below highlights the regions of the brain the AI focused on to reach its conclusion.")

    # Generate Heatmap using InceptionV3 as the explainer
    explainer_model = "resnet"  # Change this to "resnet", "vgg", or "inception" to test different explainers
    layer_name = get_last_conv_layer(explainer_model)
    
    heatmap = get_gradcam_heatmap(
        results[explainer_model]["processed_x"], 
        models_dict[explainer_model], 
        layer_name,
        pred_index=final_idx
    )

    # Overlay heatmap
    orig_img = cv2.imread(temp_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Process heatmap for overlay
    heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap_colored = plt.get_cmap("jet")(heatmap_resized)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)
    
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)

    # Display Localization
    st.image(superimposed_img, caption=f"Tumor Focus Area showsn by grad-cam", width='content')

else:
    st.info("Please upload an MRI image from the sidebar to begin analysis.")

# Clean up
if os.path.exists("temp_upload.jpg"):
    pass # Keep for visualization, handle cleanup as needed