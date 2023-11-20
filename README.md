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
                              key = 'YOUR_TOKEN',
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
Train the model and upload it to HuggingFace
```python
model.train_model()
```

## Model Usage
### Inference API Widget
Go to the model page, which can be found on your HuggingFace page. Drag and drag images onto the Inference API section to test it.
![](https://cdn.discordapp.com/attachments/1120417968032063538/1176305346671804426/hf_art_classifier_example.png?ex=656e62b9&is=655bedb9&hm=afa10c53e851d9400f600ceb27ef3fb871d3eb39d3966c4c7df44a25550bc860&)

### Python
```python
from transformers import pipeline

pipe = pipeline("image-classification", model="tonyassi/art_classifier")
result = pipe('image.png')

print(result)
```

### JavaScript API
```js
async function query(filename) {
	const data = fs.readFileSync(filename);
	const response = await fetch(
		"https://api-inference.huggingface.co/models/tonyassi/art_classifier",
		{
			headers: { Authorization: "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" },
			method: "POST",
			body: data,
		}
	);
	const result = await response.json();
	return result;
}

query("art.jpg").then((response) => {
	console.log(JSON.stringify(response));
});
```
