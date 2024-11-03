import torchvision.transforms as T

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer


class Flickr8kDataset(Dataset):
    def __init__(self, train=True):
        dataset = load_dataset("kargwalaryan/SynCap-Flickr8k")
        split_dataset = dataset["train"].train_test_split(test_size=0.15, seed=666)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if train:
            self.dataset = split_dataset['train']
        else:
            self.dataset = split_dataset['test']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]['image']
        img = img.convert("RGB")
        img = self.transform(img)
        caption = self.dataset[idx]['caption']
        encoded_caption = self.tokenizer(caption, return_tensors='pt', padding='max_length',
                                         truncation=True, max_length=16) # max_length=32
        input_ids = encoded_caption['input_ids'].squeeze(0)
        attention_mask = encoded_caption['attention_mask'].squeeze(0)
        return img, input_ids, attention_mask
