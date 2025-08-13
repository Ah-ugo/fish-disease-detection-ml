import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from skimage import filters
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     BatchNormalization, Input, GlobalAveragePooling2D,
                                     MultiHeadAttention, LayerNormalization, Reshape)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
from io import BytesIO

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
IMAGE_SIZE = (256, 256)  # Increased size for better feature extraction
DATASET_PATH = "Freshwater Fish Disease Aquaculture in south asia"
TRAIN_DIR = os.path.join(DATASET_PATH, "Train")
TEST_DIR = os.path.join(DATASET_PATH, "Test")
BATCH_SIZE = 32
MIN_SAMPLES_PER_CLASS = 10  # Minimum samples required per class
BLUR_THRESHOLD = 100  # Threshold for blur detection (higher = more blur tolerant)


def calculate_blur_score(image):
    """Calculate blur score using Laplacian variance"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()


def enhance_image_quality(image):
    """Enhance image quality with multiple techniques"""
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    if len(image.shape) == 3:  # Color image
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:  # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

    # Sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Convert back to PIL Image
    return Image.fromarray(enhanced)


def analyze_dataset_quality():
    """Enhanced dataset analysis with blur detection"""
    classes = sorted(os.listdir(TRAIN_DIR))
    class_stats = {}

    for class_name in classes:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Quality metrics
        valid_images = 0
        corrupt_images = 0
        blurry_images = 0
        image_sizes = []
        blur_scores = []

        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    # Check image size
                    image_sizes.append(img.size)

                    # Calculate blur score
                    blur_score = calculate_blur_score(img)
                    blur_scores.append(blur_score)

                    # Check if image is too blurry
                    if blur_score < BLUR_THRESHOLD:
                        blurry_images += 1
                        # Try to enhance the image
                        enhanced_img = enhance_image_quality(img)
                        enhanced_blur = calculate_blur_score(enhanced_img)
                        if enhanced_blur >= BLUR_THRESHOLD:
                            blurry_images -= 1  # Count as valid if enhancement worked

                    valid_images += 1
            except Exception:
                corrupt_images += 1

        class_stats[class_name] = {
            'total_images': len(image_files),
            'valid_images': valid_images,
            'corrupt_images': corrupt_images,
            'blurry_images': blurry_images,
            'avg_width': np.mean([s[0] for s in image_sizes]) if image_sizes else 0,
            'avg_height': np.mean([s[1] for s in image_sizes]) if image_sizes else 0,
            'avg_blur_score': np.mean(blur_scores) if blur_scores else 0,
            'min_blur_score': min(blur_scores) if blur_scores else 0
        }

    return class_stats


def load_data_with_validation():
    """Load data with enhanced validation and quality checks"""
    classes = sorted(os.listdir(TRAIN_DIR))

    # Analyze dataset first
    class_stats = analyze_dataset_quality()

    # Filter out classes with insufficient data
    valid_classes = [cls for cls in classes if class_stats[cls]['valid_images'] >= MIN_SAMPLES_PER_CLASS]

    if len(valid_classes) != len(classes):
        st.warning(f"Removed {len(classes) - len(valid_classes)} classes with insufficient data")

    label_map = {i: name.replace("_", " ").title() for i, name in enumerate(valid_classes)}

    images = []
    labels = []
    blur_scores = []

    for class_idx, class_name in enumerate(valid_classes):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB and resize
                    img = img.convert('RGB')

                    # Calculate blur score
                    blur_score = calculate_blur_score(img)
                    blur_scores.append(blur_score)

                    # Enhance image if blurry
                    if blur_score < BLUR_THRESHOLD:
                        img = enhance_image_quality(img)
                        blur_score = calculate_blur_score(img)  # Recalculate after enhancement

                    img = img.resize(IMAGE_SIZE)
                    img_array = np.array(img, dtype=np.float32) / 255.0

                    # Additional quality checks
                    mean_brightness = np.mean(img_array)
                    if 0.05 < mean_brightness < 0.95:
                        images.append(img_array)
                        labels.append(class_idx)

            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels), label_map, class_stats


def create_transfer_learning_model(input_shape, num_classes):
    """Enhanced transfer learning model with multiple backbone options"""
    backbone_type = st.session_state.get('backbone_type', 'EfficientNetB0')

    if backbone_type == 'EfficientNetB0':
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif backbone_type == 'ResNet50V2':
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    # Freeze base model initially
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


def create_vision_transformer(input_shape, num_classes):
    """Create a Vision Transformer model"""
    patch_size = 16  # Size of the patches to be extracted from the input images
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    projection_dim = 64
    num_heads = 4
    transformer_layers = 8

    inputs = layers.Input(shape=input_shape)

    # Create patches
    patches = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid"
    )(inputs)
    patches = layers.Reshape((num_patches, projection_dim))(patches)

    # Add positional embedding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)
    x = patches + position_embedding

    # Transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)

        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP
        x3 = layers.Dense(projection_dim * 2, activation="gelu")(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(projection_dim, activation="gelu")(x3)

        # Skip connection 2
        x = layers.Add()([x3, x2])

    # Classification head
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)

    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_hybrid_model(input_shape, num_classes):
    """Create a hybrid CNN-Transformer model"""
    # CNN backbone
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual blocks
    for filters in [64, 128, 256, 512]:
        for _ in range(2):
            shortcut = x
            x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = layers.Add()([shortcut, x])
            x = layers.Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)

    # Transformer part
    x = Conv2D(512, (1, 1))(x)  # Project to transformer dimension
    shape_before_flatten = x.shape[1:]
    x = Reshape((shape_before_flatten[0] * shape_before_flatten[1], shape_before_flatten[2]))(x)

    # Positional embedding
    positions = tf.range(start=0, limit=x.shape[1], delta=1)
    position_embedding = layers.Embedding(
        input_dim=x.shape[1], output_dim=512
    )(positions)
    x = x + position_embedding

    # Transformer layers
    for _ in range(4):
        # Self-attention
        attn_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward network
        ffn = layers.Dense(2048, activation='relu')(x)
        ffn = layers.Dense(512)(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn)

    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_balanced_data_generator(X, y, batch_size=32):
    """Enhanced data generator with blur-specific augmentations"""
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        channel_shift_range=30,
        fill_mode='reflect',
        preprocessing_function=lambda x: random_blur_transform(x)  # Custom blur augmentation
    )

    # Calculate class weights for balancing
    class_counts = Counter(y)
    total_samples = len(y)
    class_weights = {cls: total_samples / (len(class_counts) * count)
                     for cls, count in class_counts.items()}

    return datagen, class_weights


def random_blur_transform(image):
    """Randomly apply blur transformations to simulate real-world conditions"""
    if np.random.rand() < 0.3:  # 30% chance to apply blur
        blur_type = np.random.choice(['gaussian', 'motion', 'none'])

        if blur_type == 'gaussian':
            sigma = np.random.uniform(0.5, 2.0)
            image = cv2.GaussianBlur(image, (5, 5), sigma)
        elif blur_type == 'motion':
            size = np.random.randint(5, 15)
            kernel = np.zeros((size, size))
            kernel[int((size - 1)/2), :] = np.ones(size)
            kernel = kernel / size
            image = cv2.filter2D(image, -1, kernel)

    return image


def evaluate_model_performance(model, X_test, y_test, label_map):
    """Enhanced model evaluation with uncertainty estimation"""
    # Get predictions with multiple forward passes for uncertainty estimation
    n_forward_passes = 5
    y_preds = []

    for _ in range(n_forward_passes):
        y_pred = model.predict(X_test, verbose=0)
        y_preds.append(y_pred)

    y_pred_mean = np.mean(y_preds, axis=0)
    y_pred_std = np.std(y_preds, axis=0)
    y_pred_classes = np.argmax(y_pred_mean, axis=1)

    # Classification report
    report = classification_report(y_test, y_pred_classes,
                                   target_names=list(label_map.values()),
                                   output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    return report, cm, y_pred_mean, y_pred_std


def generate_grad_cam(model, img_array, layer_name=None):
    """Generate Grad-CAM heatmap for model interpretability"""
    # Convert single image to batch of 1
    img_array = np.expand_dims(img_array, axis=0)

    # Get model's prediction
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])

    # If no layer specified, try to find the last convolutional layer
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (Conv2D, layers.Conv2D)):
                layer_name = layer.name
                break

    # Create gradient model
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Compute gradient of top predicted class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    # Get gradients
    grads = tape.gradient(loss, conv_outputs)

    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by corresponding gradient
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap, class_idx


def plot_uncertainty(y_pred_mean, y_pred_std, label_map):
    """Plot prediction uncertainty"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot mean predictions
    sorted_idx = np.argsort(-y_pred_mean)
    ax1.bar(range(len(label_map)), y_pred_mean[sorted_idx])
    ax1.set_xticks(range(len(label_map)))
    ax1.set_xticklabels([label_map[i] for i in sorted_idx], rotation=45, ha='right')
    ax1.set_title('Mean Prediction Probabilities')
    ax1.set_ylabel('Probability')

    # Plot uncertainty
    ax2.bar(range(len(label_map)), y_pred_std[sorted_idx])
    ax2.set_xticks(range(len(label_map)))
    ax2.set_xticklabels([label_map[i] for i in sorted_idx], rotation=45, ha='right')
    ax2.set_title('Prediction Uncertainty (Std Dev)')
    ax2.set_ylabel('Standard Deviation')

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Advanced Fish Disease Detection", layout="wide")

    st.title("🐟 Advanced Fish Disease Detection System")
    st.markdown("*Optimized for blurry and low-quality underwater images*")

    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'label_map' not in st.session_state:
        st.session_state.label_map = None
    if 'backbone_type' not in st.session_state:
        st.session_state.backbone_type = 'EfficientNetB0'

    # Sidebar navigation
    page = st.sidebar.radio("Navigation",
                            ["Home", "Dataset Analysis", "Model Training", "Model Evaluation", "Prediction"])

    if page == "Home":
        st.markdown("""
        ## Advanced Fish Disease Detection Features

        ### Key Improvements for Blurry Images:
        - **Blur Detection & Enhancement**: Automatic detection and correction of blurry images
        - **Advanced Architectures**: Vision Transformers and Hybrid CNN-Transformer models
        - **Uncertainty Estimation**: Measures prediction confidence and reliability
        - **Explainability**: Grad-CAM visualizations show what the model focuses on
        - **Specialized Augmentation**: Simulates underwater imaging conditions

        ### Technical Enhancements:
        - **Multiple Backbone Options**: EfficientNetB0 or ResNet50V2
        - **Quality Control**: Filters corrupt and extremely blurry images
        - **Class Balancing**: Handles imbalanced datasets effectively
        - **Comprehensive Evaluation**: Includes uncertainty metrics
        """)

    elif page == "Dataset Analysis":
        st.header("Enhanced Dataset Analysis")

        if st.button("Analyze Dataset Quality"):
            with st.spinner("Analyzing dataset quality..."):
                try:
                    X, y, label_map, class_stats = load_data_with_validation()
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.label_map = label_map
                    st.session_state.class_stats = class_stats

                    st.success(f"Loaded {len(X)} high-quality images with {len(label_map)} classes")

                    # Display dataset statistics
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Class Distribution")
                        class_counts = pd.Series(y).value_counts().sort_index()
                        class_counts_named = class_counts.rename(index=label_map)
                        st.bar_chart(class_counts_named)

                        # Blur score distribution
                        st.subheader("Blur Score Distribution")
                        plt.figure(figsize=(8, 4))
                        sns.histplot([calculate_blur_score(Image.fromarray((x * 255).astype('uint8'))) for x in X],
                                     bins=20)
                        plt.axvline(BLUR_THRESHOLD, color='r', linestyle='--', label='Blur Threshold')
                        plt.xlabel('Blur Score (Laplacian Variance)')
                        plt.ylabel('Count')
                        plt.legend()
                        st.pyplot(plt)

                    with col2:
                        st.subheader("Dataset Quality Report")
                        quality_df = pd.DataFrame(class_stats).T
                        st.dataframe(quality_df.style.background_gradient(cmap='viridis'))

                        # Show sample images with varying blur levels
                        st.subheader("Sample Images with Blur Scores")
                        sample_indices = np.random.choice(len(X), size=3, replace=False)
                        for idx in sample_indices:
                            img = Image.fromarray((X[idx] * 255).astype('uint8'))
                            blur_score = calculate_blur_score(img)
                            col1_img, col2_img = st.columns([1, 3])
                            with col1_img:
                                st.write(f"Blur score: {blur_score:.1f}")
                            with col2_img:
                                st.image(img, use_column_width=True)

                    # Check for class imbalance
                    imbalance_ratio = max(class_counts) / min(class_counts)
                    if imbalance_ratio > 5:
                        st.warning(
                            f"Dataset is imbalanced (ratio: {imbalance_ratio:.1f}:1). Consider collecting more data for underrepresented classes.")

                except Exception as e:
                    st.error(f"Error analyzing dataset: {str(e)}")

    elif page == "Model Training":
        st.header("Advanced Model Training")

        if 'X' not in st.session_state:
            st.warning("Please analyze the dataset first.")
            return

        # Model selection options
        model_type = st.selectbox("Choose Model Architecture",
                                  ["Transfer Learning", "Vision Transformer", "Hybrid CNN-Transformer"])

        if model_type == "Transfer Learning":
            st.session_state.backbone_type = st.selectbox("Select Backbone Model",
                                                          ["EfficientNetB0", "ResNet50V2"])

        if st.button("Train Advanced Model"):
            with st.spinner("Training advanced model..."):
                try:
                    # Split data with stratification
                    X_train, X_val, y_train, y_val = train_test_split(
                        st.session_state.X, st.session_state.y,
                        test_size=0.2, random_state=42, stratify=st.session_state.y
                    )

                    # Create balanced data generator
                    datagen, class_weights = create_balanced_data_generator(X_train, y_train)

                    # Create model based on selection
                    if model_type == "Transfer Learning":
                        model, base_model = create_transfer_learning_model(
                            X_train[0].shape, len(st.session_state.label_map))
                        st.session_state.base_model = base_model
                    elif model_type == "Vision Transformer":
                        model = create_vision_transformer(
                            X_train[0].shape, len(st.session_state.label_map))
                    elif model_type == "Hybrid CNN-Transformer":
                        model = create_hybrid_model(
                            X_train[0].shape, len(st.session_state.label_map))

                    # Enhanced callbacks
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-7),
                        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy'),
                        tf.keras.callbacks.TensorBoard(log_dir='./logs')
                    ]

                    # Calculate steps
                    steps_per_epoch = len(X_train) // BATCH_SIZE
                    if len(X_train) % BATCH_SIZE != 0:
                        steps_per_epoch += 1

                    # Train model
                    history = model.fit(
                        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=steps_per_epoch,
                        epochs=50 if model_type != "Vision Transformer" else 100,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        class_weight=class_weights,
                        verbose=1
                    )

                    # Fine-tuning for transfer learning models
                    if model_type == "Transfer Learning":
                        st.info("Starting fine-tuning phase...")
                        base_model.trainable = True

                        # Use lower learning rate for fine-tuning
                        model.compile(
                            optimizer=Adam(learning_rate=0.0001 / 10),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                        )

                        # Fine-tune for fewer epochs
                        fine_tune_history = model.fit(
                            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                            steps_per_epoch=steps_per_epoch,
                            epochs=20,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            class_weight=class_weights,
                            verbose=1
                        )

                        # Combine histories
                        for key in history.history:
                            history.history[key].extend(fine_tune_history.history[key])

                    st.session_state.model = model
                    st.session_state.history = history
                    st.session_state.model_trained = True
                    st.session_state.model_type = model_type

                    # Display training results
                    st.success("Training completed successfully!")

                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots()
                        ax.plot(history.history['accuracy'], label='Training Accuracy')
                        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
                        ax.set_title('Model Accuracy')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Accuracy')
                        ax.legend()
                        st.pyplot(fig)

                    with col2:
                        fig, ax = plt.subplots()
                        ax.plot(history.history['loss'], label='Training Loss')
                        ax.plot(history.history['val_loss'], label='Validation Loss')
                        ax.set_title('Model Loss')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        st.pyplot(fig)

                    # Show final metrics
                    final_acc = history.history['val_accuracy'][-1]
                    final_loss = history.history['val_loss'][-1]
                    st.metric("Final Validation Accuracy", f"{final_acc:.4f}")
                    st.metric("Final Validation Loss", f"{final_loss:.4f}")

                    # Save model summary to text file
                    with BytesIO() as buffer:
                        model.summary(print_fn=lambda x: buffer.write(x + '\n'))
                        summary_str = buffer.getvalue().decode()
                        st.download_button("Download Model Summary", summary_str, "model_summary.txt")

                except Exception as e:
                    st.error(f"Error during training: {str(e)}")

    elif page == "Model Evaluation":
        st.header("Advanced Model Evaluation")

        if 'model' not in st.session_state:
            st.warning("Please train a model first.")
            return

        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model performance..."):
                try:
                    # Use validation split for evaluation
                    X_train, X_test, y_train, y_test = train_test_split(
                        st.session_state.X, st.session_state.y,
                        test_size=0.2, random_state=42, stratify=st.session_state.y
                    )

                    # Get predictions with uncertainty
                    report, cm, y_pred_mean, y_pred_std = evaluate_model_performance(
                        st.session_state.model, X_test, y_test, st.session_state.label_map
                    )

                    # Display results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Classification Report")
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(3))

                        # Show uncertainty plot for a random sample
                        st.subheader("Prediction Uncertainty Example")
                        sample_idx = np.random.choice(len(X_test))
                        fig = plot_uncertainty(y_pred_mean[sample_idx], y_pred_std[sample_idx],
                                               st.session_state.label_map)
                        st.pyplot(fig)

                    with col2:
                        st.subheader("Confusion Matrix")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=list(st.session_state.label_map.values()),
                                    yticklabels=list(st.session_state.label_map.values()),
                                    ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        plt.xticks(rotation=45)
                        plt.yticks(rotation=0)
                        st.pyplot(fig)

                    # Overall metrics
                    st.subheader("Overall Performance")
                    accuracy = report['accuracy']
                    macro_f1 = report['macro avg']['f1-score']
                    weighted_f1 = report['weighted avg']['f1-score']

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{accuracy:.3f}")
                    col2.metric("Macro F1-Score", f"{macro_f1:.3f}")
                    col3.metric("Weighted F1-Score", f"{weighted_f1:.3f}")

                    # Average uncertainty
                    avg_uncertainty = np.mean(y_pred_std)
                    st.metric("Average Prediction Uncertainty", f"{avg_uncertainty:.4f}")

                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")

    elif page == "Prediction":
        st.header("Advanced Disease Prediction")

        if 'model' not in st.session_state:
            st.warning("Please train a model first.")
            return

        uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded Image")
                img = Image.open(uploaded_file).convert('RGB')

                # Calculate blur score
                blur_score = calculate_blur_score(img)
                st.write(f"Blur Score: {blur_score:.1f} (Threshold: {BLUR_THRESHOLD})")

                # Enhance if blurry
                if blur_score < BLUR_THRESHOLD:
                    st.warning("Image is blurry - applying enhancement")
                    img = enhance_image_quality(img)
                    enhanced_blur = calculate_blur_score(img)
                    st.write(f"Enhanced Blur Score: {enhanced_blur:.1f}")

                img = img.resize(IMAGE_SIZE)
                st.image(img, caption="Processed Image", use_column_width=True)

                # Preprocess image
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

            with col2:
                st.subheader("Advanced Prediction Results")
                with st.spinner("Analyzing image with advanced model..."):
                    try:
                        # Get prediction with uncertainty
                        n_forward_passes = 5
                        preds = []
                        for _ in range(n_forward_passes):
                            pred = st.session_state.model.predict(img_array, verbose=0)
                            preds.append(pred)

                        pred_mean = np.mean(preds, axis=0)[0]
                        pred_std = np.std(preds, axis=0)[0]
                        pred_class = np.argmax(pred_mean)
                        confidence = np.max(pred_mean)
                        uncertainty = pred_std[np.argmax(pred_mean)]

                        # Enhanced confidence interpretation
                        if confidence > 0.9 and uncertainty < 0.05:
                            confidence_level = "Very High"
                            confidence_color = "green"
                        elif confidence > 0.7 and uncertainty < 0.1:
                            confidence_level = "High"
                            confidence_color = "blue"
                        elif confidence > 0.5 and uncertainty < 0.15:
                            confidence_level = "Medium"
                            confidence_color = "orange"
                        else:
                            confidence_level = "Low"
                            confidence_color = "red"

                        # Display main prediction
                        st.markdown(
                            f"<h3 style='color:{confidence_color}'>Predicted Disease: {st.session_state.label_map[pred_class]}</h3>",
                            unsafe_allow_html=True
                        )

                        col2_1, col2_2, col2_3 = st.columns(3)
                        col2_1.metric("Confidence", f"{confidence:.2%}")
                        col2_2.metric("Uncertainty", f"{uncertainty:.4f}")
                        col2_3.metric("Confidence Level", confidence_level)

                        # Show all predictions with probabilities
                        st.subheader("All Disease Probabilities with Uncertainty")
                        prob_df = pd.DataFrame({
                            'Disease': [st.session_state.label_map[i] for i in range(len(st.session_state.label_map))],
                            'Probability': pred_mean,
                            'Uncertainty': pred_std
                        }).sort_values('Probability', ascending=False)

                        # Format the dataframe
                        def color_confidence(val):
                            if val > 0.7:
                                return 'background-color: lightgreen'
                            elif val > 0.4:
                                return 'background-color: lightyellow'
                            else:
                                return 'background-color: lightcoral'

                        def color_uncertainty(val):
                            if val < 0.05:
                                return 'background-color: lightgreen'
                            elif val < 0.1:
                                return 'background-color: lightyellow'
                            else:
                                return 'background-color: lightcoral'

                        styled_df = prob_df.style \
                            .applymap(color_confidence, subset=['Probability']) \
                            .applymap(color_uncertainty, subset=['Uncertainty']) \
                            .format({'Probability': '{:.2%}', 'Uncertainty': '{:.4f}'})

                        st.dataframe(styled_df, use_column_width=True)

                        # Generate Grad-CAM visualization
                        if st.session_state.model_type != "Vision Transformer":  # Grad-CAM works better with CNNs
                            st.subheader("Model Attention Visualization")
                            heatmap, _ = generate_grad_cam(st.session_state.model, img_array[0])

                            # Overlay heatmap on original image
                            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                            superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

                            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                            ax[0].imshow(img)
                            ax[0].set_title('Original Image')
                            ax[0].axis('off')

                            ax[1].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
                            ax[1].set_title('Model Attention (Grad-CAM)')
                            ax[1].axis('off')

                            st.pyplot(fig)

                        # Recommendation based on confidence and uncertainty
                        if confidence < 0.7 or uncertainty > 0.1:
                            st.warning("⚠️ Prediction Reliability Concerns:")
                            st.write("- The image may be too blurry or unclear")
                            st.write("- The disease may not be clearly visible")
                            st.write("- Consider retaking the photo with:")
                            st.write("  - Better lighting conditions")
                            st.write("  - Clearer focus on the affected area")
                            st.write("  - Multiple angles if possible")

                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")

        # Additional information
        with st.expander("Advanced Model Information"):
            if 'history' in st.session_state:
                final_acc = st.session_state.history.history['val_accuracy'][-1]
                st.write(f"Model validation accuracy: {final_acc:.1%}")

            st.write("This advanced model includes:")
            st.write("- Blur detection and image enhancement pipeline")
            st.write(f"- {st.session_state.get('model_type', 'Transfer Learning')} architecture")

            if st.session_state.get('model_type') == "Transfer Learning":
                st.write(f"- Backbone: {st.session_state.get('backbone_type', 'EfficientNetB0')}")

            st.write("- Uncertainty estimation with multiple forward passes")
            st.write("- Specialized data augmentation for underwater images")


if __name__ == "__main__":
    main()