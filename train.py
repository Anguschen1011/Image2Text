import os
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader

from Codes.Dataset import Flickr8kDataset
from Codes.Model import MODEL

# Train
def train(model, train_loader, optimizer, epochs, device, lr):
    losses = []
    best_loss = np.inf

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:02}/{epochs:02}", leave=True)

        for batch_idx, batch in enumerate(pbar):
            images = batch[0].to(device)
            input_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

            # 前向傳播並計算損失
            loss = model(images, text_inputs)
            running_loss += loss.item()

            # 反向傳播和參數更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

        # 在每個epoch結束後計算總的平均損失
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        pbar.set_description(f"Epoch {epoch + 1:02}/{epochs:02}, Loss: {avg_loss:.4f}")

        # 儲存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            # model_name = f"checkpoints/best_score.pt"
            model_name = f"/content/checkpoints/best_score.pt"
            torch.save(model.state_dict(), model_name)
            print("Checkpoint Saved")
    return losses



if __name__ == "__main__":
    # Training Parameters
    emb_dim = 512
    lr = 1e-4
    epochs = 50
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    train_dataset = Flickr8kDataset(train=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=2)

    model = MODEL(emb_dim).to(device)

    optimizer = optim.Adam([
        {'params': model.image_encoder.resnet.parameters(), 'lr': 1e-5},
        {'params': model.text_encoder.bert.parameters(), 'lr': 1e-5},
        {'params': model.image_encoder.projection.parameters(), 'lr': lr},
        {'params': model.text_encoder.projection.parameters(), 'lr': lr},
        {'params': [model.temperature], 'lr': lr}
    ])
    
    # Training
    # os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('/content/checkpoints', exist_ok=True)
    print('\n\n========== Start Training ==========')
    loss = train(model, train_loader, optimizer, epochs, device, lr)

    plt.figure(figsize=(10, 5))
    plt.plot(loss, marker='o')

    plt.title('Loss Visualization')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('loss.png')
    plt.close()