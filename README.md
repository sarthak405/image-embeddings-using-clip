# image-embeddings-using-clip
Simple demo to create image embeddings using Open AI's CLIP model and push the vector to CSV 


## Prerequisite
1. Download [Bean leaf dataset](https://www.kaggle.com/datasets/prakharrastogi534/bean-leaf-dataset) and put it under `data/beanleaf_dataset`
2. Install files from `requirements.txt`

## Usage
1. Execute `beanleaf_classification_local_dataset.ipynb` for generate the tf model for classifying bean leaf

2. Execute `create_image_embedding_csv.ipynb` and create a csv file with the following headers
    ```
    ['model_name', 'model_version', 'name', 'url', 'actual_label', 'predicted_label', 'score', 'prediction_ts', 'vector']
    ```

    ```
    model_name:     name of the model
    model_version:  model version
    name:           name of the image file
    url:            path of the image file
    actual_label:   what is the actual label
    predicted_label:what did the model predict
    score:          model's confidence score
    prediction_ts:  prediction time stamp
    vector:         image embedding
    ```

3. Execute `beanleaf_classification_local_dataset_observability.ipynb` to push it to arize

4. Go to Arize dashboard to see a new model `beanleaf-disease-classifier` created in the homepage

## Dataset
[Bean leaf dataset](https://www.kaggle.com/datasets/prakharrastogi534/bean-leaf-dataset)

[Know Your Data](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=beans)