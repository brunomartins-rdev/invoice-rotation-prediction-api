import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from dataset import InvoiceDataset
from model import get_model
import constants
from typing import List, Optional, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on {device}")

def train_on_batch(image_paths: List[str],
                   num_epochs: int = constants.NUMBER_EPOCHS,
                   model: Optional[nn.Module] = None,
                   batch_size: int = constants.BATCH_SIZE,
                   image_size: int = constants.IMAGE_SIZE) -> Tuple[nn.Module, bool]:
    """
    Trains the model on a list of image paths for prefined epochs.
    Prints the average loss after each epoch.
    Returns the trained model and a flag that always indicates success.

    Args:
        image_paths: List of file paths to images.
        num_epochs: How many times to go through the dataset.
        model: An existing model to continue training. If not given for continuous training, a new one is used.
        batch_size: How many images to load at once during training.
        image_size: The size to resize each image to before training. Images will be square.
    """
    # TODO: Check if rectangle instead of square help with accuracy simce images are rectangular
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = InvoiceDataset(constants.CSV_PATH, image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if model is None:
        model = get_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_idx, (images, angles) in enumerate(dataloader):
            images, angles = images.to(device), angles.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            preds = model(images)
            loss = F.mse_loss(preds, angles)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    return model, True

