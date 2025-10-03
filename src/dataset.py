import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Tuple, Any, Callable

class InvoiceDataset(Dataset):
    def __init__(self, 
                 csv_path: str, 
                 image_paths: List[str], 
                 transform: Optional[Callable] = None) -> None:
        """
        Reads the CSV file and keeps only the rows with filenames that match the image paths.
        """
        self.df = pd.read_csv(csv_path, sep=";")
        self.image_paths = image_paths
        self.transform = transform
        self.df = self.df[self.df['file'].isin([os.path.basename(p) for p in image_paths])]

    def __len__(self) -> int:
        """
        Returns the number of rows in the filtered DataFrame.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        """
        Returns the image and its angle value as a tensor for the given index.
        """
        row = self.df.iloc[idx]
        img_path = next(p for p in self.image_paths if os.path.basename(p) == row["file"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        angle = torch.tensor(row["angle"], dtype=torch.float32)
        return image, angle

