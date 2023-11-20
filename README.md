# Easy Image Classification Model Maker
Create a custom image classification model with a few lines of code.

## Installation
```bash
pip install -r requirements.txt
```

## Train Model
Import the module
```python
import modelmaker
```
Define the model and dataset parameters:
- **keyword** list of strings will be the labels of the model
- **num_images** number of images in the training dataset
- **key** HuggingFace write access token can be created [here](https://huggingface.co/settings/tokens).
- **dataset_name** name of dataset that will uploaded to HuggingFace
- **model_name** name of model that will be uploaded to HuggingFace
- **train_epochs** number of training epochs the model will go through
```python
model = modelmaker.ModelMaker(keywords = ['cubism', 'impressionism', 'abstract expressionism'],
                              num_images = 100,
                              key = YOUR_TOKEN,
                              dataset_name = 'art_dataset',
                              model_name = 'art_classifier',
                              train_epochs = 10)
```
Download images from Bing into the './images' folder. It is suggested to manually go through the image folders to make sure there isn't any incorrect images in their respective folders. 
```python
model.download_images()
```
Upload dataset to HuggingFace
```python
model.upload_dataset()
```
Train the model and upload it to Huggingface
```python
model.train_model()
```
