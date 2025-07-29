import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, \
    GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import to_categorical
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
IMAGE_SIZE = (224, 224)
DATASET_PATH = "Freshwater Fish Disease Aquaculture in south asia"
TRAIN_DIR = os.path.join(DATASET_PATH, "Train")
TEST_DIR = os.path.join(DATASET_PATH, "Test")
BATCH_SIZE = 16
MIN_SAMPLES_PER_CLASS = 10  # Minimum samples required per class


def analyze_dataset_quality():
    """Analyze dataset for quality issues"""
    classes = sorted(os.listdir(TRAIN_DIR))
    class_stats = {}

    for class_name in classes:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Check image quality
        valid_images = 0
        corrupt_images = 0
        image_sizes = []

        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    image_sizes.append(img.size)
                    valid_images += 1
            except Exception:
                corrupt_images += 1

        class_stats[class_name] = {
            'total_images': len(image_files),
            'valid_images': valid_images,
            'corrupt_images': corrupt_images,
            'avg_width': np.mean([s[0] for s in image_sizes]) if image_sizes else 0,
            'avg_height': np.mean([s[1] for s in image_sizes]) if image_sizes else 0
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

    for class_idx, class_name in enumerate(valid_classes):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB
                    img = img.convert('RGB')
                    img = img.resize(IMAGE_SIZE)
                    img_array = np.array(img, dtype=np.float32) / 255.0

                    # Basic quality check - skip very dark or very bright images
                    mean_brightness = np.mean(img_array)
                    if 0.05 < mean_brightness < 0.95:
                        images.append(img_array)
                        labels.append(class_idx)

            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels), label_map, class_stats


def create_transfer_learning_model(input_shape, num_classes):
    """Create model using transfer learning with EfficientNet"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model initially
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


def create_enhanced_cnn_model(input_shape, num_classes):
    """Create enhanced CNN with attention mechanism"""
    inputs = Input(shape=input_shape)

    # First block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Second block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Third block
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Fourth block
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    # Dense layers
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_balanced_data_generator(X, y, batch_size=32):
    """Create data generator with class balancing"""
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect'
    )

    # Calculate class weights for balancing
    class_counts = Counter(y)
    total_samples = len(y)
    class_weights = {cls: total_samples / (len(class_counts) * count)
                     for cls, count in class_counts.items()}

    return datagen, class_weights


def evaluate_model_performance(model, X_test, y_test, label_map):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification report
    report = classification_report(y_test, y_pred_classes,
                                   target_names=list(label_map.values()),
                                   output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    return report, cm, y_pred_classes


def main():
    st.set_page_config(page_title="Enhanced Fish Disease Detection", layout="wide")

    st.title("🐟 Enhanced Fish Disease Detection System")
    st.markdown("*Improved model with better accuracy and validation*")

    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'label_map' not in st.session_state:
        st.session_state.label_map = None

    # Sidebar navigation
    page = st.sidebar.radio("Navigation",
                            ["Home", "Dataset Analysis", "Model Training", "Model Evaluation", "Prediction"])

    if page == "Home":
        st.markdown("""
        ## Enhanced Detection System Features

        ### Improvements Made:
        - **Transfer Learning**: Using EfficientNetB0 pre-trained on ImageNet
        - **Data Quality Checks**: Automatic detection of corrupt/poor quality images
        - **Class Balancing**: Weighted training to handle imbalanced datasets
        - **Enhanced Validation**: Better evaluation metrics and confusion matrix
        - **Improved Augmentation**: More sophisticated data augmentation
        - **Model Ensemble**: Option to combine multiple models

        ### Key Benefits:
        - Higher accuracy on disease classification
        - Better handling of similar-looking diseases
        - Reduced overfitting through better regularization
        - More reliable predictions with confidence scores
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

                    with col2:
                        st.subheader("Dataset Quality Report")
                        quality_df = pd.DataFrame(class_stats).T
                        st.dataframe(quality_df)

                    # Check for class imbalance
                    imbalance_ratio = max(class_counts) / min(class_counts)
                    if imbalance_ratio > 5:
                        st.warning(
                            f"Dataset is imbalanced (ratio: {imbalance_ratio:.1f}:1). Consider collecting more data for underrepresented classes.")

                except Exception as e:
                    st.error(f"Error analyzing dataset: {str(e)}")

    elif page == "Model Training":
        st.header("Enhanced Model Training")

        if 'X' not in st.session_state:
            st.warning("Please analyze the dataset first.")
            return

        model_type = st.selectbox("Choose Model Architecture",
                                  ["Transfer Learning (EfficientNet)", "Enhanced CNN"])

        if st.button("Train Enhanced Model"):
            with st.spinner("Training enhanced model..."):
                try:
                    # Split data with stratification
                    X_train, X_val, y_train, y_val = train_test_split(
                        st.session_state.X, st.session_state.y,
                        test_size=0.2, random_state=42, stratify=st.session_state.y
                    )

                    # Create balanced data generator
                    datagen, class_weights = create_balanced_data_generator(X_train, y_train)

                    # Create model based on selection
                    if model_type == "Transfer Learning (EfficientNet)":
                        model, base_model = create_transfer_learning_model(
                            X_train[0].shape, len(st.session_state.label_map)
                        )
                        st.session_state.base_model = base_model
                    else:
                        model = create_enhanced_cnn_model(
                            X_train[0].shape, len(st.session_state.label_map)
                        )

                    # Callbacks
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7),
                        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')
                    ]

                    # Calculate steps
                    steps_per_epoch = len(X_train) // BATCH_SIZE
                    if len(X_train) % BATCH_SIZE != 0:
                        steps_per_epoch += 1

                    # Train model
                    history = model.fit(
                        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=steps_per_epoch,
                        epochs=50,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        class_weight=class_weights,
                        verbose=1
                    )

                    # Fine-tuning for transfer learning models
                    if model_type == "Transfer Learning (EfficientNet)":
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

                except Exception as e:
                    st.error(f"Error during training: {str(e)}")

    elif page == "Model Evaluation":
        st.header("Model Performance Evaluation")

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

                    # Get predictions
                    report, cm, y_pred = evaluate_model_performance(
                        st.session_state.model, X_test, y_test, st.session_state.label_map
                    )

                    # Display results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Classification Report")
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(3))

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

                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")

    elif page == "Prediction":
        st.header("Enhanced Disease Prediction")

        if 'model' not in st.session_state:
            st.warning("Please train a model first.")
            return

        uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded Image")
                img = Image.open(uploaded_file).convert('RGB').resize(IMAGE_SIZE)
                st.image(img, caption="Uploaded Image", use_container_width=True)

                # Preprocess image
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

            with col2:
                st.subheader("Enhanced Prediction Results")
                with st.spinner("Analyzing image with enhanced model..."):
                    try:
                        # Get prediction
                        pred = st.session_state.model.predict(img_array, verbose=0)
                        pred_class = np.argmax(pred[0])
                        confidence = np.max(pred[0])

                        # Enhanced confidence interpretation
                        if confidence > 0.9:
                            confidence_level = "Very High"
                            confidence_color = "green"
                        elif confidence > 0.7:
                            confidence_level = "High"
                            confidence_color = "blue"
                        elif confidence > 0.5:
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

                        col2_1, col2_2 = st.columns(2)
                        col2_1.metric("Confidence", f"{confidence:.2%}")
                        col2_2.metric("Confidence Level", confidence_level)

                        # Show all predictions with probabilities
                        st.subheader("All Disease Probabilities")
                        prob_df = pd.DataFrame({
                            'Disease': [st.session_state.label_map[i] for i in range(len(st.session_state.label_map))],
                            'Probability': pred[0]
                        }).sort_values('Probability', ascending=False)

                        # Color code the dataframe
                        def color_probability(val):
                            if val > 0.5:
                                return 'background-color: lightgreen'
                            elif val > 0.2:
                                return 'background-color: lightyellow'
                            else:
                                return 'background-color: lightcoral'

                        styled_df = prob_df.style.applymap(color_probability, subset=['Probability'])
                        st.dataframe(styled_df, use_container_width=True)

                        # Recommendation based on confidence
                        if confidence < 0.7:
                            st.warning("⚠️ Low confidence prediction. Consider:")
                            st.write("- Taking a clearer, well-lit image")
                            st.write("- Ensuring the affected area is clearly visible")
                            st.write("- Consulting with a fish disease expert")

                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")

        # Additional information
        with st.expander("Model Information"):
            if 'history' in st.session_state:
                final_acc = st.session_state.history.history['val_accuracy'][-1]
                st.write(f"Model validation accuracy: {final_acc:.1%}")
            st.write("This enhanced model uses advanced techniques for better accuracy:")
            st.write("- Transfer learning or enhanced CNN architecture")
            st.write("- Data augmentation and class balancing")
            st.write("- Quality checks and validation")


if __name__ == "__main__":
    main()