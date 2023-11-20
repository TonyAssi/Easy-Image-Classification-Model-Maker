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
Define the model and dataset parameters
- **keyword** list of strings will be the labels of the model
- **num_images** number of images in the training dataset
- **key** HuggingFace write access token can be created [here](https://huggingface.co/settings/tokens).
```python
model = modelmaker.ModelMaker(keywords,
                              num_images,
                              key,
                              dataset_name,
                              model_name,
                              train_epochs)
```
