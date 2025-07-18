import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

import cv2  # OpenCV for video processing

# ---------------------------
# Custom Transform for Frames
# ---------------------------
class FrameTransform:
    """
    Applies a given transform to each frame in a video.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, video):
        # video: T x C x H x W
        transformed = torch.stack([self.transform(frame) for frame in video])
        return transformed

# ---------------------------
# Custom Dataset for Videos using OpenCV
# ---------------------------
class VideoDataset(Dataset):
    """
    Custom Dataset for loading videos organized in class-specific folders using OpenCV.
    """
    def __init__(self, video_paths, labels, transform=None, num_frames=30, frame_size=(224, 224)):
        """
        Args:
            video_paths (list): List of video file paths.
            labels (list): List of corresponding labels.
            transform (callable, optional): Transform to be applied on a sample.
            num_frames (int): Number of frames to sample from each video.
            frame_size (tuple): Desired frame size (width, height).
        """
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames
        self.frame_size = frame_size  # (width, height)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"No frames found in video: {video_path}")

            # Calculate frame indices to sample
            if total_frames >= self.num_frames:
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            else:
                indices = list(range(total_frames))
                # Pad by repeating the last frame
                pad_indices = [total_frames - 1] * (self.num_frames - total_frames)
                indices.extend(pad_indices)

            frame_dict = {i: idx for idx, i in enumerate(indices)}

            current_frame = 0
            sampled_frame = 0
            while sampled_frame < self.num_frames:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                if current_frame in frame_dict:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize frame
                    frame = cv2.resize(frame, self.frame_size)
                    # Convert to tensor
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # C x H x W
                    frames.append(frame)
                    sampled_frame += 1

                current_frame += 1

            cap.release()

            # Handle cases where video has fewer frames
            if len(frames) < self.num_frames:
                last_frame = frames[-1]
                while len(frames) < self.num_frames:
                    frames.append(last_frame)

            video = torch.stack(frames)  # T x C x H x W

            if self.transform:
                video = self.transform(video)

        except Exception as e:
            print(f"Error reading {video_path}: {e}")
            # Return a tensor of zeros if video reading fails
            video = torch.zeros((self.num_frames, 3, self.frame_size[1], self.frame_size[0]))

        return video, label

# ---------------------------
# Video Classification Model
# ---------------------------
class VideoClassificationModel(nn.Module):
    """
    Video Classification Model using ResNet, LSTM, and Transformer-based Attention.
    """
    def __init__(self, num_classes, hidden_size=256, num_lstm_layers=1, 
                 bidirectional=True, transformer_layers=2, nhead=8, dropout=0.5):
        """
        Args:
            num_classes (int): Number of output classes.
            hidden_size (int): Hidden size for LSTM.
            num_lstm_layers (int): Number of LSTM layers.
            bidirectional (bool): If True, use bidirectional LSTM.
            transformer_layers (int): Number of Transformer encoder layers.
            nhead (int): Number of attention heads in Transformer.
            dropout (float): Dropout probability.
        """
        super(VideoClassificationModel, self).__init__()

        # Load pretrained ResNet and remove the final classification layer
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze ResNet parameters
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Output: 512-dim features
        self.feature_size = 512

        # LSTM for temporal feature aggregation
        self.lstm = nn.LSTM(input_size=self.feature_size, 
                            hidden_size=hidden_size,
                            num_layers=num_lstm_layers, 
                            batch_first=True,
                            bidirectional=bidirectional)
        self.lstm_dropout = nn.Dropout(dropout)

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        # Multi-head Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * (2 if bidirectional else 1),
                                               num_heads=nhead, 
                                               dropout=dropout,
                                               batch_first=True)

        # Transformer Encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size * (2 if bidirectional else 1),
                                                   nhead=nhead, 
                                                   dim_feedforward=2048,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Final Classification Layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, C, H, W)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        batch_size, num_frames, C, H, W = x.size()

        # Reshape to process all frames at once through ResNet
        x = x.view(batch_size * num_frames, C, H, W)
        features = self.resnet(x)  # (batch_size*num_frames, 512, 1, 1)
        features = features.view(batch_size, num_frames, self.feature_size)  # (batch_size, num_frames, 512)

        # LSTM
        lstm_out, _ = self.lstm(features)  # (batch_size, num_frames, hidden_size * num_directions)
        lstm_out = self.lstm_dropout(lstm_out)

        # Attention
        attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)  # (batch_size, num_frames, embed_dim)

        # Transformer
        transformer_out = self.transformer(attn_output)  # (batch_size, num_frames, embed_dim)

        # Global Average Pooling over time
        pooled = transformer_out.mean(dim=1)  # (batch_size, embed_dim)

        # Classification
        logits = self.classifier(pooled)  # (batch_size, num_classes)

        return logits

# ---------------------------
# Training and Validation
# ---------------------------
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=50, patience=10):
    """
    Trains the model and validates it. Implements early stopping.

    Args:
        model (nn.Module): The neural network to train.
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to run the training on.
        num_epochs (int): Maximum number of epochs.
        patience (int): Number of epochs with no improvement after which training stops.

    Returns:
        nn.Module: The trained model with the best validation loss.
    """
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_path = 'best_model.pth'

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()}'):
                inputs = inputs.to(device)  # (batch_size, num_frames, C, H, W)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # (batch_size, num_classes)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # Gradient clipping
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Step the scheduler if in validation phase
            if phase == 'val':
                scheduler.step(epoch_loss)

                # Deep copy the model if it has better validation loss or accuracy
                if epoch_loss < best_val_loss or epoch_acc > best_val_acc:
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                    if epoch_acc > best_val_acc:
                        best_val_acc = epoch_acc
                    epochs_no_improve = 0
                    # Format accuracy to four decimal places
                    acc_percent = epoch_acc.item()
                    acc_formatted = f"{acc_percent:.4f}"
                    # Create filename with loss and accuracy
                    model_filename = f'best_model_loss_{epoch_loss:.4f}_acc_{acc_formatted}.pth'
                    torch.save(model.state_dict(), model_filename)
                    print(f'Validation loss or accuracy improved. Saving model to {model_filename}')
                else:
                    epochs_no_improve += 1
                    print(f'No improvement in validation loss or accuracy for {epochs_no_improve} epoch(s).')

        # Check early stopping condition
        if epochs_no_improve >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs.')
            break

    # Load the best model weights based on validation loss and accuracy
    best_models = [f for f in os.listdir('.') if f.startswith('best_model_loss_') and f.endswith('.pth')]
    if best_models:
        # Sort models by loss ascending and accuracy descending
        best_models_sorted = sorted(best_models, key=lambda x: (float(x.split('_loss_')[1].split('_acc_')[0]),
                                                               -float(x.split('_acc_')[1].split('.pth')[0])))
        best_model_path = best_models_sorted[0]
        model.load_state_dict(torch.load(best_model_path))
        print(f'Loaded the best model from {best_model_path}')
    else:
        print('No best model found. Loading the current model state.')

    return model

# ---------------------------
# Main Function
# ---------------------------
def main():
    # Configuration
    root_dir = 'dataset'       # Root directory of the dataset
    num_epochs = 100           # Maximum number of epochs
    batch_size = 2             # Batch size (reduced to prevent memory issues)
    learning_rate = 1e-4       # Learning rate
    num_frames = 30            # Number of frames per video
    num_classes = 4            # Number of classes: EL, MD, NM, SV
    patience = 15              # Early stopping patience
    frame_size = (224, 224)    # Frame size (width, height)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Define transforms with data augmentation
    frame_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform = FrameTransform(frame_transform)

    # Initialize the dataset
    # First, gather all video paths and labels
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    video_paths = []
    labels = []

    # Supported video extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.MOV')

    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        for video_file in os.listdir(cls_dir):
            if video_file.lower().endswith(video_extensions):
                video_paths.append(os.path.join(cls_dir, video_file))
                labels.append(clascs_to_idx[cls])

    print(f"Total videos found: {len(video_paths)}")

    if len(video_paths) == 0:
        print("No videos found in the dataset. Please check the dataset directory.")
        return

    # Convert labels to numpy array for StratifiedShuffleSplit
    labels_np = np.array(labels)

    # Stratified split into train and validation sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, val_indices = next(sss.split(video_paths, labels_np))

    train_video_paths = [video_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]

    val_video_paths = [video_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    print(f'Training samples: {len(train_video_paths)}')
    print(f'Validation samples: {len(val_video_paths)}')

    # Compute class weights to handle class imbalance
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)  # Normalize weights
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f'Class counts (Training): {class_counts}')
    print(f'Class weights: {class_weights}')

    # Initialize datasets
    train_dataset = VideoDataset(train_video_paths, train_labels, transform=transform, num_frames=num_frames, frame_size=frame_size)
    val_dataset = VideoDataset(val_video_paths, val_labels, transform=FrameTransform(transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])), num_frames=num_frames, frame_size=frame_size)

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    }

    # Initialize the model
    model = VideoClassificationModel(num_classes=num_classes, hidden_size=256, num_lstm_layers=1, 
                                     bidirectional=True, transformer_layers=2, nhead=8, dropout=0.5)
    model = model.to(device)
    print(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Train the model
    best_model = train_model(model, dataloaders, criterion, optimizer, scheduler, device, 
                             num_epochs=num_epochs, patience=patience)

    # Save the final model
    final_model_path = 'final_model.pth'
    torch.save(best_model.state_dict(), final_model_path)
    print(f'\nTraining complete. Best model saved to {final_model_path}')

    # ---------------------------
    # Evaluation on Validation Set
    # ---------------------------
    def evaluate_model(model, dataloader, device, class_names):
        """
        Evaluates the model on the validation set and prints classification metrics.

        Args:
            model (nn.Module): Trained model.
            dataloader (DataLoader): Validation DataLoader.
            device (torch.device): Device to perform evaluation on.
            class_names (list): List of class names.

        Returns:
            None
        """
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc='Evaluating'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

    # Define class names
    class_names = classes  # ['EL', 'MD', 'NM', 'SV']

    # Evaluate the model
    evaluate_model(best_model, dataloaders['val'], device, class_names)

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == '__main__':
    main()
