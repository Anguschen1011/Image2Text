import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from transformers import BertModel


class TextEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-large-uncased')

        for name, param in self.bert.named_parameters():
            if 'encoder.layer.22' in name or 'encoder.layer.23' in name or 'encoder.layer.24' in name or 'pooler' in name:
                # print(f'layer : {name} requires_grad = True')
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.projection = nn.Linear(self.bert.config.hidden_size, emb_dim)

    def forward(self, text):
        outputs = self.bert(**text)
        x = outputs.last_hidden_state[:, 0, :]

        x = self.projection(x)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        return x


class ImageEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.resnet = models.resnet101(weights='IMAGENET1K_V1')

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        for name, param in self.resnet.named_parameters():
            if '7.0' in name or '7.1' in name or '7.2' in name:
                # print(f'layer : {name} requires_grad = True')
                param.requires_grad = True
            else:
                param.requires_grad = False


        self.output_dim = 2048
        self.projection = nn.Linear(self.output_dim, emb_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)

        x = self.projection(x)
        x = F.normalize(x, p=2, dim=-1)

        return x


class MODEL(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.image_encoder = ImageEncoder(emb_dim)
        self.text_encoder = TextEncoder(emb_dim)

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, image, text):
        # 編碼圖像和文本
        I_e = self.image_encoder(image)
        T_e = self.text_encoder(text)

        # 計算 logits（相似度分數）
        logits = (I_e @ T_e.transpose(-2, -1)) * torch.exp(self.temperature)

        # 創建對比損失的標籤
        labels = torch.arange(logits.shape[0]).to(self.device)

        # 圖像和文本的交叉熵損失
        loss_i = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)
        loss_t = nn.functional.cross_entropy(logits, labels)
        loss = (loss_i + loss_t) / 2

        # 對比損失
        # 正規化嵌入
        I_e_norm = nn.functional.normalize(I_e, dim=-1)
        T_e_norm = nn.functional.normalize(T_e, dim=-1)

        # 計算成對相似度
        similarity_matrix = I_e_norm @ T_e_norm.transpose(-2, -1)

        # 創建正樣本和負樣本的遮罩
        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        negative_mask = ~positive_mask

        # 計算正樣本和負樣本的相似度
        positive_similarities = similarity_matrix[positive_mask].view(logits.size(0), -1)
        negative_similarities = similarity_matrix[negative_mask].view(logits.size(0), -1)

        # 對比損失（使用 InfoNCE 損失公式）
        temperature = 0.1  # 可以根據需要調整此超參數
        positive_loss = -torch.log(torch.exp(positive_similarities / temperature) /
                                    (torch.exp(positive_similarities / temperature).sum(dim=1, keepdim=True) +
                                     torch.exp(negative_similarities / temperature).sum(dim=1, keepdim=True)))

        contrastive_loss = positive_loss.mean()

        # 最終損失是之前損失的組合
        total_loss = loss + contrastive_loss

        return total_loss