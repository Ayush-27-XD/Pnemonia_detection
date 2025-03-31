import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image


# Define paths
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Normalization for validation and testing
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=32, class_mode='binary', shuffle=False
)

# Build the InceptionV3 model
def build_inceptionv3_model(input_shape=(150, 150, 3)):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze first 100 layers
    for layer in base_model.layers[:100]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model = build_inceptionv3_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Checkpoint callback (Fixed the filename)
checkpoint_path = 'pneumonia_detection_model.keras'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=5,
    callbacks=[checkpoint, early_stopping]
)

# Load best model weights
model.load_weights(checkpoint_path)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Predict on test set
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.round(predictions).astype(int)

# True labels
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Function to predict Pneumonia vs Normal from an image
def predict_pneumonia(img_path, model):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))  # Resize to match model input
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"

    # Load the original image with OpenCV
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (300, 300))  # Resize for display

    # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0) if predicted_class == "NORMAL" else (0, 0, 255)  # Green for normal, Red for pneumonia
    position = (10, 50)  # Position to display text
    font_scale = 1
    thickness = 2

    cv2.putText(img_cv, f"Predicted: {predicted_class}", position, font, font_scale, color, thickness)

    # Convert image from BGR (OpenCV format) to RGB (for displaying with matplotlib)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # Display the image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(img_cv)
    plt.axis("off")
    plt.show()

    print(f"Predicted Class: {predicted_class}")

# Example usage:
predict_pneumonia("C:/Users/Ayush/Desktop/STENDEC/dataset/test_image/person9_bacteria_40.jpeg", model)