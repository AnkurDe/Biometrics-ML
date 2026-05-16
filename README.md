# Biometrics-ML
This is private repository for Machine Learning 22AIE213 project work

# Problem Statement:
This project is about handwriting classification where given a page of handwriting, and a name
The ML model tests if the handwriting matches the person or not

# Pipeline

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