# Generate descriptive text based on the image's content

## 1. Environment
This project runs on ```Google Colab```, and the execution results can be found in ```Image2Text.ipynb```.   


## 2. Dataset Preprocessing (SynCap-Flickr8k [[Hugging Face]](https://huggingface.co/datasets/kargwalaryan/SynCap-Flickr8k)   
This project uses the ```SynCap-Flickr8k``` dataset from Hugging Face for training.  
> Using the dataset requires Hugging Face's user access tokens. You can refer to: [[Hugging Face]](https://huggingface.co/docs/hub/security-tokens)


## 3. Training
```python train.py```  
You can change the following parameters in the code.
```
emb_dim: Embedding Dimension
lr: Learning Rate
epochs: Number of Epochs
batch_size: Batch Size
``` 


## 4. Testing  
```python test.py```  
The results will be saved in the ```results``` folder.  


## 5. Results
