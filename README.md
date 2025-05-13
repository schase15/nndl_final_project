# Inference Pipeline README

## Overview: Data Pipeline Flow

1. **Input**: A directory of images to classify.
2. **Superclass Prediction**: Images are classified into one of the known superclasses (or marked as novel) using a CLIP-based model embeddings with cosine similarity and/or a VLM for ambiguous cases.
3. **Subclass Prediction**: For each superclass, images are further classified into subclasses using the same methods as in superclass.
4. **Result Combination**: Superclass and subclass predictions are merged into a single results file.
5. **Metrics (Optional)**: If true labels are provided, the pipeline computes accuracy, precision, recall, F1, and categorical cross-entropy for both superclass and subclass.

---

## Repository Structure (Structure and files required for inference pipeline)

```
inference_pipeline/
│
├── predict_full_pipeline.py         # Main entry point for running the inference pipeline (runs predict_superclass.py and predict_subclass.py)
├── predict_superclass.py            # Predicts superclasses for all images (runs cosine and llm superclass scripts)
├── cosine_superclass_predict.py     # Cosine similarity-based superclass prediction
├── llm_superclass_classifier.py     # LLM-based fallback for superclass prediction
├── predict_subclass.py              # Runs subclass prediction for each superclass (runs cosine and llm subclass scripts)
├── cosine_subclass_predict.py       # Cosine similarity-based subclass prediction
├── llm_subclass_classifier.py       # LLM-based fallback for subclass prediction
├── combine_final_results.py         # Combines superclass and subclass predictions into final_results.csv
├── Copy of cas-on-clab.ipynb        # Notebook for running CLIP server in Colab
│
├── val_run/                         # Directory containing custom validation image set including novel images from Imagenet64
│
├── results/                         # Output directory for predictions and metrics
│   ├── final_results.csv            # Contains superclass and subclass predictions per image
│   ├── superclass_predictions.csv   # Result file from the superclas predictions step only
│   ├── subclass_predictions/        # Result files for the subclass predictions of each superclass
│   └── metrics.txt                  # Calculated metrics if true label file is provided
│
├── results/superclass_subsets/      # HDF5 files with embeddings grouped by each superclass (needed for subclass prediction)
│
data/
└── Released_Data_NNDL_2025/
    ├── test_images/                 # Directory of test images to classify
    ├── train_images/
    ├── subclass_mapping.csv         # Mapping of subclass indices to names
    ├── superclass_mapping.csv       # Mapping of superclass indices to names
    └── train_data.csv

phase1/                             # Calculated during training 
└── data/class_centroids.npy        # Superclass centroids

phase2/                             # Calculated during training 
├── dog/                            # Subclass centroids for dog
├── bird/                           # Subclass centroids for bird
└── reptile/                        # Subclass centroids for reptile
```

---

## How to Run the Inference Pipeline

### **1. Prepare Data and Centroids**
- Place your images to classify in a directory (e.g., `data/Released_Data_NNDL_2025/test_images/`).
- Ensure subclass centroids exist in `phase2/dog/`, `phase2/bird/`, and `phase2/reptile/` as `{subclass}_centroid.npy`. (Where subclass is the subclass index)

### **2. Start the CLIP server**
- Use the `inference_pipeline/Copy of cas-on-clab.ipynb` to start the CLIP server. Instructions are in the notebook. It is designed to run in Colab with a T4 GPU.
- Requires an ngrok auth token
- Test the server is active and you are connected by running `inference_pipeline/test_clip.py`

### **3. Run the Full Pipeline**
From the `inference_pipeline/` directory, run:
```bash
python predict_full_pipeline.py --image_dir <path_to_images> --output_dir <output_dir>
```
- Example:
  ```bash
  python predict_full_pipeline.py --image_dir ../data/Released_Data_NNDL_2025/test_images --output_dir results/
  ```

#### **Optional: Evaluate with True Labels**
If you have a CSV of true labels (with columns: `image`, `superclass_index`, `subclass_index`), add:
```bash
  --true_labels <path_to_true_labels.csv>
```
This will compute and save metrics in `results/metrics/metrics.txt`.

### **4. Output**
- The main results will be in `<output_dir>/final_results.csv`.
- Intermediate predictions and metrics will be in the `results/` subfolders.

---

## Notes

- Only the scripts and folders listed above are required for inference. Training, analysis, and experimental scripts are not needed. Some are mentioned below
- Make sure all dependencies are installed (see `requirements.txt`).

## Other files

- `phase1/embedding_storage.py` — Utilities for saving/loading image embeddings (used in both training and inference)
- `phase2/create_train_embeddings_per_superclass.py` — Script to generate CLIP embeddings for each superclass from the training set
- `phase2/create_subclass_centroids.py` — Script to compute subclass centroids from training embeddings
- `phase2/analyze_centroid_classification.py` — Script to analyze centroid-based classification results
- `phase2/tune_subclass_thresholds.py` — Script to tune thresholds for subclass classification
- `phase1/create_imagenet_mapping.py` — Script to create a mapping between ImageNet classes and your label space
- `phase1/create_imagenet_embeddings.py` — Script to generate CLIP embeddings for ImageNet images
- `phase1/tune_cosine_superclass_classifier.py` — Script to tune thresholds and parameters for the cosine similarity-based superclass classifier
- `phase1/calculate_centroids.py` — Script to calculate centroids for superclasses from embeddings
