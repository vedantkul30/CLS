import streamlit as st
import numpy as np
import cv2
import pickle
from catboost import CatBoostClassifier
import joblib
from PIL import Image

# Function to preprocess the image
def preprocess_image(image, clip_limit, threshold, min_contour_area, kernel_size):
    
    if image is None:
        print(f"Error loading the image")
        return None, None

    height = image.shape[0]
    image_cropped = image[450:height - 300, :]

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image_cropped)

    _, thresholded = cv2.threshold(enhanced_image, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_result = np.zeros_like(image_cropped, dtype=np.uint8)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area:
            cv2.drawContours(filtered_result, [contour], -1, 255, thickness=cv2.FILLED)

    color_result = np.ones((enhanced_image.shape[0], enhanced_image.shape[1], 3), dtype=np.uint8) * 255
    color_result[filtered_result == 255] = [255, 0, 0]

    return image_cropped,cleaned, color_result

def extract_features(image):
    # Load the image with highlighted cracks
    
    if image is None:
        print(f"Error loading the image")
        return None

    # Convert to HSV and extract the red channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))  # Red pixel mask

    # Detect contours
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize feature values
    total_crack_area = 0
    crack_sizes = []
    crack_lengths = []
    aspect_ratios = []
    elongation_ratios = []
    circularities = []
    largest_crack_size = 0
    largest_crack_length = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10:  # Ignore small noise
            continue

        # Features per contour
        total_crack_area += area
        crack_sizes.append(area)
        length = cv2.arcLength(contour, closed=True)
        crack_lengths.append(length)
        largest_crack_size = max(largest_crack_size, area)
        largest_crack_length = max(largest_crack_length, length)

        # Fit a bounding box
        rect = cv2.minAreaRect(contour)
        (width, height), _, _ = rect
        width, height = sorted([width, height])  # Ensure width <= height
        aspect_ratios.append(height / width if width > 0 else 0)

        # Compute elongation ratio
        elongation_ratios.append(length / width if width > 0 else 0)

        # Compute circularity
        circularity = (4 * np.pi * area) / (length**2) if length > 0 else 0
        circularities.append(circularity)

    # Aggregate features
    crack_density = total_crack_area / (mask_red.shape[0] * mask_red.shape[1])
    median_crack_size = np.median(crack_sizes) if crack_sizes else 0
    median_crack_length = np.median(crack_lengths) if crack_lengths else 0
    top_3_crack_sizes = sorted(crack_sizes, reverse=True)[:3]
    largest_to_mean_area_ratio = (largest_crack_size / np.mean(crack_sizes)) if crack_sizes else 0
    proportion_dominant_crack = (largest_crack_size / total_crack_area) if total_crack_area > 0 else 0

    # Combine all features into a single dictionary
    features = {
        "Total Crack Area": total_crack_area,
        "Crack Density": crack_density,
        "Largest Crack Size": largest_crack_size,
        "Largest Crack Length": largest_crack_length,
        "Median Crack Size": median_crack_size,
        "Median Crack Length": median_crack_length,
        "First Largest Crack": top_3_crack_sizes[0] if len(top_3_crack_sizes) > 0 else 0,
        "Second Largest Crack": top_3_crack_sizes[1] if len(top_3_crack_sizes) > 1 else 0,
        "Third Largest Crack": top_3_crack_sizes[2] if len(top_3_crack_sizes) > 2 else 0,
        # "Bounding Box Aspect Ratios Mean": np.mean(aspect_ratios) if aspect_ratios else 0,
        "Elongation Ratios Mean": np.mean(elongation_ratios) if elongation_ratios else 0,
        # "Circularities Mean": np.mean(circularities) if circularities else 0,
        "Largest-to-Mean Area Ratio": largest_to_mean_area_ratio,
        # "Proportion of Total Area in Dominant Crack": proportion_dominant_crack,
    }

    return features

kernel_size = (3,3)
min_contour_area = 15
threshold = 75
clip_limit = 1.2
learning_rate = 0.1
l2_leaf_reg = 3
iterations = 300
depth = 8

model = CatBoostClassifier()
model.load_model("cat_clf.cbm")

# Load StandardScaler
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("CLS Classification App")
st.write("Upload an image here")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = np.array(image.convert('L')) 

    # Apply preprocessing
    image_cropped, cleaned_img,image_highlight_only = preprocess_image(image,clip_limit,threshold, min_contour_area, kernel_size)

    post_img = cv2.cvtColor(image_highlight_only, cv2.COLOR_RGB2BGR)

    st.write("### Preprocessed Image:")
    st.image(post_img, caption="Preprocessed Highlighted Image", use_container_width=True)

    # Extract features
    extracted_features = extract_features(post_img)

    if extracted_features is not None:
        feature_values = list(extracted_features.values())  # Extract numerical values
        scaled_features = scaler.transform([feature_values])  # Convert to 2D array

        # Predict using the trained model
        prediction = model.predict(scaled_features)
        labels = ["CLS-A LONGI", "CLS-B LONGI", "CLS-C LONGI", "CLS-D LONGI"]

        result = labels[int(prediction[0])]

        # Display prediction
        st.write("### Prediction:")
        st.write(f"Model Output Class: **{result}**")

        


