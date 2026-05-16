# Biometrics-ML
This is private repository for Machine Learning 22AIE213 project work

# Problem Statement:
This project is about handwriting classification where given a page of handwriting, and a name
The ML model tests if the handwriting matches the person or not

---

# Overall Pipeline
This is the overall pipeline including everything from training to application
## Image preprocessing
`src/preproc_pipeline.py`
This is the image preprocessing pipeline. When this file is run then for all the people it creates Processed_Data folder where for each person there is a folder having all the processed images stored as individual segments


## Image Embedding (Feature Extraction)
`src/mk_vectors_v6.py`
Here the images need to be converted to vectors. This is where in the Processed_Data folder each Person's folder is traversed where each image is converted to vector using the AlexNet model. The final vectors are stored with the names of the person in a large parquet file which will be used for training

## Model Training
`src/Models/Person_model_ModelOutput.py`
This script is responsible for model training and storing the best models in the `Trained_Models` folder. Here from the parquet file generated earlier it takes individual person and creates a bi-class classifier model which says if the image segment is of a person or not

## Model Output
`src/model_output.py`

Tkinter app for handwriting verification on a single image. Run from the repo root (with the virtual environment active):

```bash
python src/model_output.py
```

Pipeline:
1. Accept image and the person name (dropdown of trained models in `Trained_Models/`)
2. Run the image preprocessing pipeline
3. For each segment run the image embedding and generate vectors from the segments
4. Use these vectors and run the appropriate pretrained model from `Trained_Models/`
5. Take the majority vote of per-segment predictions and return the final decision with confidence: $$\text{confidence} = \frac{\text{number of majority votes}}{\text{total number of votes}} \times 100$$

---
# Image Preprocessing Pipeline
## 🏞️ Overall Pipeline Flow
The pipeline follows a sequential structure, transforming the raw image through several stages:
1. Grayscale Conversion: Reduces the image data to intensity values.
2. Noise Reduction: Filters out random pixel variations.
3. Contrast Normalization/Binarization: Enhances the contrast and converts the image to a black-and-white binary mask, making text stand out sharply.
4. Segmentation: Divides the processed image into manageable, logical parts (segments) using a sophisticated sliding window approach.


## ✨ Detailed Step Analysis of the Image Processing Pipeline

This analysis provides a deep dive into the five core steps used to prepare and segment document images for sophisticated text analysis.

### 1. Grayscale Conversion
**Working:**
This step converts the input image from its multichannel format (e.g., BGR for color) into a single-channel grayscale image. Each pixel's value now 
represents its intensity (brightness), discarding color information.

**Complexity:**
Low. It is a standard transformation in image processing.

**Importance:**
High. Color information is often irrelevant for text recognition, and converting to grayscale reduces computational complexity while preserving structural 
information vital for text analysis.

### 2. Noise Reduction
**Working:**
Utilizes a fast non-local means (NLM) filtering algorithm. Instead of averaging pixel values (which blurs edges), this method models the image in patches, 
preserving edges and textures while effectively removing random "salt-and-pepper" noise or sensor noise.

**Complexity:**
Medium. It is a dedicated filtering algorithm that analyzes local neighborhoods rather than just the direct neighbors.

**Importance:**
High. Noise can severely degrade the quality of text characters, leading OCR or analysis algorithms to misinterpret pixels. Removing noise ensures the 
underlying structure of the text remains intact.


### 3. Contrast Normalization/Binarization (`normalise_contrast`)
**Working:**
This function performs local thresholding using Gaussian weights. Instead of applying a single threshold across the entire image (Global Thresholding), it 
calculates a threshold dynamically for small regions (blocks) of the image. By using a binary inverse threshold (`THRESH_BINARY_INV`), the technique ensures 
that the text (which is typically dark) converts into black pixels, and the background (which is typically light) converts into white pixels. This results 
in a precise binary mask.

**Complexity:**
High. The use of Adaptive Thresholding makes it robust to uneven lighting or varying background backgrounds across the document.

**Importance:**
Critical. This is perhaps the most crucial preparatory step. It standardizes the image appearance, ensuring that the subsequent segmentation steps deal with 
clean, unambiguous black-and-white data, maximizing the signal-to-noise ratio for text features.


### 4. Image Segmentation
**Working:**
This is the most complex step, responsible for taking the clean, binary image and dividing it into meaningful, localized patches (segments).

**Detailed Process:**
1. **Initial Scaling:** First, the image is resized to a standard minimum dimension to ensure consistent processing size.
2. **Connected Component Analysis (CCA):** The function identifies all distinct, connected groups of pixels (e.g., a single word, a small cluster of 
letters). Noise or extremely small components are filtered out based on a minimum area threshold.
3. **Dynamic Dimension Calculation:** It analyzes the median height of the detected components to estimate a characteristic handwriting or line height.
4. **Sliding Window Segmentation:** Using this estimated height, the function implements a grid-based "sliding window." It moves across the image in 
predetermined steps (e.g., stepping by `seg_h` vertical and `seg_w` horizontal).
5. **Patching:** Each window captures a potential segment patch. Empty patches are skipped, and the resulting valid patches, along with their coordinates, 
are collected.

**Complexity:**
Very High. It combines multiple CV techniques (CCA, statistical analysis, controlled iteration/windowing).

**Importance:**
Critical. Segmentation allows the pipeline to work on localized pieces of content rather than the entire image at once. This compartmentalization is 
essential for passing patches to specialized downstream models (e.g., a handwriting recognition model that expects only a single word or line).


### 5. Image Resizing for Display
**Working:**
This is a utility function, not part of the core processing pipeline. It scales the image down proportionally so that it fits within specified maximum 
dimensions (e.g., 800x800 pixels) without distortion, primarily for visualization and user display purposes.

**Complexity:**
Low. It uses simple scaling ratios.

**Importance:**
Low (Pipeline Context). It is important only for debugging or visualization, having no effect on the functional processing quality.

---

# Understanding Image Embedding and Feature Vectors

Image embedding is a fundamental concept in modern computer vision and machine learning, enabling computers to "understand" the content of an image in a 
numerical format that can be processed by algorithms.

## 💡 1. The Importance of Image Embedding

At its core, a raw image is just a grid of pixel intensities (RGB values). While this is data, it is not mathematically meaningful for complex tasks like 
classification, detection, or similarity comparison.

**Image embedding** is the process of converting this high-dimensional pixel data into a compact, dense numerical representation (a vector) in a 
mathematical space. This vector is the image's "fingerprint" or semantic embedding.

### 🌟 Why is this important?

1.  **Computational Efficiency:** Instead of processing millions of raw pixel values, algorithms can process a few thousand numbers in the compact vector. 
This drastically reduces computational load.
2.  **Meaningful Similarity:** In the embedding space, the distance between two vectors directly correlates with the perceived similarity between the 
original images. Images that look or are related (e.g., two pictures of cats, even different breeds) will have vectors that are close together.
3.  **Robust Feature Representation:** Embeddings capture high-level, semantic features—the *meaning* of the image (e.g., "it contains a face," or "it is a 
tropical scene")—rather than just low-level details like individual pixel colors.

The layers collected are
1. Penultimate-2 Layer Output (The Early Feature Details)<br>
Has 4096 dimensions, source=`model.classifier[4]`
2. Penultimate-1 Layer Output (The Raw Score Confidence)<br>
Has 1000 dimensions, Source=logits
3. Softmax Output (The Normalized Probability)<br>
Has 1000 dimensions, Source `torch.softmax`

---

## Training Implementation Details
Implements an advanced machine learning pipeline designed for solving a binary classification problem: identifying a specific "person" (label) from a larger dataset containing multiple individuals.

### 📚 Overview and Purpose

The primary goal of the script is to train and save multiple state-of-the-art classifiers (e.g., Random Forest, XGBoost, SVM, etc.) to distinguish between the target person and all other individuals in the dataset.

The pipeline is designed to be robust by:
1.  **Addressing Class Imbalance:** Using a custom function to create a balanced dataset.
2.  **Feature Engineering:** Applying feature scaling and dimensionality reduction (PCA).
3.  **Model Comparison:** Training and comparing a wide variety of machine learning models using Hyperparameter Tuning (GridSearchCV).
4.  **Reproducibility:** Ensuring that all steps are reproducible by setting random seeds and using cross-validation.

### 🧠 Reasoning Behind Key Design Choices

#### 1. Data Balancing (`create_balanced_dataset`)
*   **Problem:** In real-world datasets, the target class (the specific person's data) might be rare compared to the negative class (everyone else). This class imbalance can cause models to be biased towards the majority class, performing poorly on the minority class.
*   **Solution:** The `create_balanced_dataset` function manually ensures that the number of negative samples (others) for any given target label is at least equal to the number of positive samples (the target person). It achieves this by sampling other individuals until a desired balance is reached, preventing the model from ignoring the rare positive class.

#### 2. Feature Preparation (`build_pipeline` and `split_features_targets`)
*   **Scaling:** All machine learning models, especially distance-based ones like KNN or kernel-based ones like SVM, require features to be on a similar scale. `StandardScaler` normalizes the data to have a mean of 0 and a standard deviation of 1.
*   **Dimensionality Reduction (PCA):** High-dimensional data can suffer from the curse of dimensionality, making models overfit or slowing down training. Principal Component Analysis (PCA) is used to project the data onto a lower-dimensional subspace while retaining a high percentage (e.g., 85%) of the original variance. This simplifies the feature set without significant loss of information.
*   **Data Splitting:** The data is split into training (80%) and testing (20%) sets, ensuring that the proportion of positive and negative examples is maintained in both sets using `stratify=y`.

#### 3. Model Selection and Hyperparameter Tuning (`get_models` and `train_and_evaluate_models`)
*   **Model Diversity:** The code includes a wide array of model types (linear, tree-based, probabilistic, ensemble) to maximize the chance of finding the best classifier for the specific dataset.
*   **Grid Search:** Instead of relying on default parameters, `GridSearchCV` is used. It systematically tests multiple combinations of hyperparameters (e.g., different learning rates for XGBoost, different regularization strengths for Logistic Regression) to find the optimal configuration for each model.
*   **Robustness:** Using `StratifiedKFold` cross-validation ensures that the performance metrics (like accuracy) are not dependent on a single train-test split but are averaged across multiple folds.

### 💻 Implementation Flow Details

The execution follows a structured workflow defined by the `run()` function:

1.  **Initialization:** The pipeline iterates through all unique individuals found in the dataset (`unique_names`).
2.  **Data Preparation (Per Label):** For each `label`, the `create_balanced_dataset` function executes to generate a balanced dataset for training.
3.  **Data Split:** The balanced data is split into training and testing sets.
4.  **Model Training Loop:** The `train_and_evaluate_models` function takes the split data and iterates through all defined models:
    *   **Pipelining:** For each model, a pipeline is constructed: `StandardScaler` $\rightarrow$ `PCA` $\rightarrow$ `Classifier`.
    *   **Fitting:** `GridSearchCV` is fitted on the training data (`X_train`, `y_train`) using cross-validation.
    *   **Evaluation:** The best model found by the grid search is used to predict outcomes on both the training and the held-out test sets.
5.  **Result Tracking:** The model with the highest **Test Accuracy** is selected as the best performer for that specific person (`label`).
6.  **Persistence:** The best model's pipeline, along with its associated train and test accuracies, is serialized and saved using `pickle` for later deployment or use.

---

## Web App Runner Using Flask

This application is structured as a classic full-stack web service, separating the user interface (HTML/JavaScript) from the core business logic (Python/Flask).

Here is an in-depth breakdown of the implementation details.

***

### 💻 Architecture Overview

The project uses **Flask** for the web backend, handling routing and API calls. The frontend uses Jinja templating combined with vanilla **JavaScript** for dynamic interactions, especially the camera capture feature. The central logic relies on external functions like `list_trained_persons()` and `run_inference()` imported from `model_output.py`.

***

### 🖼️ Frontend Implementation (HTML/JavaScript)

The frontend is contained within the large `INDEX_HTML` string and is responsible for user interaction, especially image input.

#### 1. User Interface (HTML/Jinja2)

*   **Form Structure:** A main `<form id="analyze-form">` manages all inputs.
*   **Person Selection:** A `<select id="person">` dropdown is dynamically populated using Jinja templating (`{% for name in persons %}`) with pre-existing trained persons.
*   **Input Source Tabs:** The interface uses a tab system (`source-tabs`) to switch between two primary input modes: `Upload file` and `Use camera`.
*   **Camera Setup:**
    *   `<video id="camera-video">`: Displays the live video feed.
    *   `<canvas id="camera-canvas" hidden>`: Used behind the scenes to draw the captured frame for processing, ensuring the image data is available for the browser API.
    *   `<img id="capture-preview">`: Displays the final captured photo to the user.
*   **Error/Result Display:** Jinja blocks (`{% if error %}...{% endif %}`, `{% if result %}...{% endif %}`) are used to render feedback to the user (e.g., validation errors or the final analysis results).

#### 2. Client-Side Logic (JavaScript)

The JavaScript IIFE (Immediately Invoked Function Expression) handles the complex front-end interactions:

*   **Camera API Handling (`startCamera`, `stopCamera`):**
    *   It utilizes `navigator.mediaDevices.getUserMedia()` to request access to the camera stream.
    *   It implements state management to manage the `stream` object, allowing the user to start, stop, and restart the camera stream gracefully.
    *   It includes logic for error handling if the camera access is denied or unavailable.
*   **Capture Logic (`capturePhoto`):**
    *   It draws the current video frame onto a hidden `<canvas>` element using `canvas.getContext("2d").drawImage()`.
    *   It then converts the canvas content into a `Blob` using `canvas.toBlob()`.
    *   Finally, it simulates file input by setting the captured `Blob` data into the hidden file input element (`fileInput.files`) and updating the preview image.
*   **Tab Switching:** An event listener is attached to the source tabs, which toggles the visibility of the `panel-upload` and `panel-camera` sections, ensuring the UI matches the active source.
*   **Submission Validation:** The form submission handler prevents default submission if no file is attached, providing user-friendly alerts based on the active input source.

***

### 🐍 Backend Implementation (Python/Flask)

The Flask routes and helper functions manage the requests, secure the data, and execute the ML inference.

#### 1. Initialization and Configuration

*   **Path Setup:** `_SRC_DIR = Path(__file__).resolve().parent` and `sys.path.insert(0, str(_SRC_DIR))` ensure that local modules (like `model_output`) are correctly importable regardless of where the application is run.
*   **App Setup:** `app = Flask(__name__)` initializes the Flask application.
*   **Security:** `app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024` sets a limit on uploaded file size (16 MB).
*   **Constants:** `ALLOWED_EXTENSIONS` defines acceptable file types for robust validation.

#### 2. Routing and Request Handling (`index` function)

The `@app.route("/", methods=["GET", "POST"])` function is the core entry point:

*   **GET Request:** If the request is a `GET`, it simply loads the `INDEX_HTML` template, populating the available person list (`persons = list_trained_persons()`).
*   **POST Request (Processing):**
    1.  **Data Retrieval:** It retrieves the `person` selection and the uploaded `file` (`request.files.get("image")`).
    2.  **Validation:** It performs sequential validation:
        *   Is a person selected?
        *   Is a file present?
        *   Is the file type supported? (Checked by `_allowed_file(filename)`)
    3.  **Temporary File Management (Crucial Detail):**
        *   The application **must** save the uploaded file to a temporary, physical location on the server before passing it to the inference function.
        *   It uses `tempfile.NamedTemporaryFile` within a `try...finally` block.
        *   The `finally` block guarantees that the temporary file (`Path(tmp_path).unlink()`) is deleted from the filesystem, preventing resource leakage, regardless of whether inference succeeded or failed.
    4.  **Inference Execution:**
        *   `result = run_inference(tmp_path, person)`: This is the point where the system hands off the processed path and the target person ID to the model's core logic.
    5.  **Error Handling:** Comprehensive `try...except` blocks catch various failures (`FileNotFoundError`, `ValueError`, general `Exception`) and format them into an `error` variable for display on the frontend.

#### 3. File Utility

*   **`_allowed_file(filename: str) -> bool`:** This helper function abstracts the file type validation, ensuring that only supported extensions can proceed to the ML model.