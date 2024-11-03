import os
import clip
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from Codes.Model import MODEL
from Codes.Dataset import Flickr8kDataset


def compute_average_precision(relevance, scores):
    return average_precision_score(relevance, scores.cpu().numpy())


def display_images_with_captions(val_dataset, model, text_embeddings, num_examples=20, K=3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 設定隨機數種子
    seed = 666
    np.random.seed(seed)

    # 確保 results 資料夾存在
    os.makedirs('results', exist_ok=True)

    # 開啟文字檔案以寫入每張圖像的前 K 個描述
    with open("results/results.txt", "w") as file:
        
        # 隨機選取圖像索引
        indices = np.random.choice(len(val_dataset), num_examples, replace=False)
        img_idx = 1  # 用來追蹤圖像的編號

        # 迴圈處理每一個隨機選取的圖像索引
        for idx in indices:
            # 從 val_dataset 中取得圖像及其相關資料 (input_ids, attention_mask)
            img, input_ids, attention_mask = val_dataset[idx.item()]

            # 將圖像移至裝置 (例如 GPU) 上並添加一個維度，符合模型輸入格式
            img_tensor = img.to(device).unsqueeze(0)

            # 停用梯度計算，以節省記憶體及提升速度
            with torch.no_grad():
                # 使用模型的 image_encoder 編碼圖像，取得圖像的特徵向量
                image_feature = model.image_encoder(img_tensor)

                # 計算圖像特徵與所有文字嵌入之間的相似度
                sims = image_feature @ text_embeddings.T
                # 壓縮維度，使相似度向量符合需求
                sims = sims.squeeze(0)

            # 根據相似度從大到小排序，取出前 K 個最高相似度的索引
            topk_indices = torch.argsort(sims, descending=True)[:K]
            # 將索引轉換為整數，方便後續使用
            topk_indices = [int(i.item()) for i in topk_indices]

            # 從 val_dataset 中取出與這些索引對應的描述文字
            topk_captions = [val_dataset.dataset[i]['caption'] for i in topk_indices]

            # 使用 PIL 儲存圖像
            img_np = img.permute(1, 2, 0).cpu().numpy()  # 轉換為 numpy 格式

            # 正規化到 [0, 1] 的範圍
            img_min, img_max = img_np.min(), img_np.max()
            img_np = (img_np - img_min) / (img_max - img_min)  # 確保數值在 [0, 1] 範圍內
            img_np = (img_np * 255).astype(np.uint8)  # 將數值縮放到 [0, 255]

            img_pil = Image.fromarray(img_np).resize((256, 256))  # 轉為 PIL 影像
            img_pil.save(f"results/{img_idx}.png")  # 儲存影像

            # 寫入檔案標題和描述
            file.write(f"Top {K} Captions for Image {img_idx} Index {idx}:\n")
            for j, caption in enumerate(topk_captions):
                file.write(f"{j+1}: {caption}\n")
            file.write("\n")  # 分隔每張圖像的描述

            img_idx += 1  # 更新圖像編號


def test(val_loader, val_dataset, model, clip_model, clip_preprocess, num_examples=20, K=3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_embeddings = []
    text_embeddings = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Extracting embeddings"):
            images, input_ids, attention_mask = batch
            images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)

            # 使用模型的影像和文本編碼器
            image_features = model.image_encoder(images)
            text_features = model.text_encoder({"input_ids": input_ids, "attention_mask": attention_mask})

            image_embeddings.append(image_features)
            text_embeddings.append(text_features)

    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    # 計算相似度矩陣
    similarity_matrix = image_embeddings @ text_embeddings.T
    num_images = len(image_embeddings)

    average_top_probs = {1: [], 3: [], 5: []}
    max_top_probs = {1: [], 3: [], 5: []}
    ranks = []
    ap_list = []

    # 計算相似度和 AP
    for i in tqdm(range(num_images), desc="Calculating similarities"):
        sims = similarity_matrix[i]
        sorted_indices = torch.argsort(sims, descending=True)

        # 獲取相關文本的二元標籤
        relevance = (sorted_indices == i).cpu().numpy().astype(int)
        ap = compute_average_precision(relevance, sims)
        ap_list.append(ap)

        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    recall_at_1 = np.mean(ranks <= 1)
    recall_at_3 = np.mean(ranks <= 3)
    recall_at_5 = np.mean(ranks <= 5)

    # 計算前 K (1, 3, 5) 的準確度
    for i in tqdm(range(num_images), desc="Calculating accuracies"):
        sims = similarity_matrix[i]
        for K in [1, 3, 5]:
            topk_indices = torch.argsort(sims, descending=True)[:K]
            topk_texts = [val_dataset.dataset[int(idx)]['caption'] for idx in topk_indices.cpu().numpy()]

            image = val_dataset[i][0]
            image = clip_preprocess(Image.fromarray(image.permute(1, 2, 0).cpu().numpy().astype("uint8"))).unsqueeze(0).to(device)
            text_tokens = clip.tokenize(topk_texts).to(device)

            with torch.no_grad():
                logits_per_image, _ = clip_model(image, text_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            average_top_probs[K].append(np.mean(probs))
            max_top_probs[K].append(np.max(probs))

    display_images_with_captions(val_dataset, model, text_embeddings, num_examples=20, K=3)

    return recall_at_1, recall_at_3, recall_at_5, average_top_probs, max_top_probs


if __name__ == "__main__":
    batch_size = 32
    emb_dim = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODEL(emb_dim).to(device)
    model.load_state_dict(torch.load("checkpoints/best_score.pt", weights_only=True))
    
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    model.eval()
    clip_model.eval()

    val_dataset = Flickr8kDataset(train=False)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=2)

    os.makedirs('results', exist_ok=True)
    print('\n\n========== Testing ==========')
    recall_at_1, recall_at_3, recall_at_5, average_top_probs, max_top_probs = test(val_loader, val_dataset, model, clip_model, clip_preprocess, num_examples=20, K=3)
    
    print(f"\nImage-to-Text Retrieval:")
    print(f"Recall@1: {recall_at_1 * 100:.4f}%")
    print(f"Recall@3: {recall_at_3 * 100:.4f}%")
    print(f"Recall@5: {recall_at_5 * 100:.4f}%")

    print("\nProbabilities Calculate by CLIP \"ViT-B/32\" Model:")
    for K in [1, 3, 5]:
        print(f"\nFor Top-{K} results:")
        print(f"Average Probability: {np.mean(average_top_probs[K]):.4f}")
        print(f"Max Probability: {np.mean(max_top_probs[K]):.4f}")