import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import pandas as pd


# Define a transformation for your images (e.g., resizing and converting to tensors)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Define the directories for the original and augmented spectrogram images
original_data_dir = 'img'
augmented_data_dir = 'aug'

# Custom dataset class for individual spectrogram images
class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image).unsqueeze(0)

        return image

# Create custom datasets for original and augmented images
original_dataset = SpectrogramDataset(original_data_dir, transform=transform)
augmented_dataset = SpectrogramDataset(augmented_data_dir, transform=transform)

# Create separate DataLoaders for original and augmented images
batch_size = 32
original_dataloader = DataLoader(original_dataset, batch_size=batch_size, shuffle=True)
augmented_dataloader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)

class SiameseNetwork(nn.Module):
    def __init__(self, base_encoder, num_classes):
        super(SiameseNetwork, self).__init__()
        self.base_encoder = base_encoder
        self.fc1 = nn.Linear(512, 512)  
        self.fc2 = nn.Linear(512, num_classes)

    def forward_one(self, x):
        x = self.base_encoder(x)
        x = x.view(x.size()[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


# Example: Use a pre-trained ResNet as a base encoder
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity() 
num_classes = 3 
siamese_net = SiameseNetwork(resnet,num_classes)

# Define the contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, output1, output2, labels):
        # Use cross-entropy loss with logits for classification
        loss_fn = nn.CrossEntropyLoss()
        loss1 = loss_fn(output1, labels)
        loss2 = loss_fn(output2, labels)
        return loss1 + loss2

def save_printout_to_txt(printout, file_path):
    with open(file_path, 'w') as txt_file:
        txt_file.write(printout)

# Define optimizer and learning rate scheduler
optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop on the labeled dataset (label.csv)
label_data = pd.read_csv('label.csv')  # Load your labeled dataset here
label_labels = label_data['labels'].values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
siamese_net.to(device)

# Train the Siamese network with the classification head
num_epochs = 20
siamese_net.train()

# Define the loss function
loss_fn = ContrastiveLoss()

for epoch in range(num_epochs):
    total_loss = 0.0
    for i, (original_batch, augmented_batch) in enumerate(zip(original_dataloader, augmented_dataloader)):
        anchor = original_batch[0].to(device)  # Take the first item as the anchor
        positive = augmented_batch[0].to(device)  # Take the first item as the positive sample
        negative = augmented_batch[1].to(device)  # Take the second item as the negative sample

        # Assuming the data in the labeled dataset is ordered consistently
        label = torch.tensor(label_labels[i], dtype=torch.long).unsqueeze(0).to(device) 

        # Forward pass
        anchor_output, positive_output = siamese_net(anchor, positive)
        negative_output = siamese_net.forward_one(negative)

        # Compute the loss for each anchor-positive pair separately
        loss1 = loss_fn(anchor_output, negative_output, label)
        loss2 = loss_fn(positive_output, negative_output, label)
        
        # Combine the losses (you can adjust how you want to combine them)
        loss = (loss1 + loss2) / 2.0

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(original_dataloader)}')
    scheduler.step()



# Save the encoder's weights
torch.save(siamese_net.base_encoder.state_dict(), 'encoder_weights_finetuned.pth')


# Define the Siamese network architecture with the same encoder structure
new_resnet = models.resnet18(pretrained=False)  # Ensure not to load pre-trained weights
new_resnet.fc = nn.Identity()  # Remove the final classification layer
new_siamese_net = SiameseNetwork(new_resnet, num_classes)

new_siamese_net.base_encoder.load_state_dict(torch.load('encoder_weights_finetuned.pth'))
new_siamese_net.eval()
