# vit_data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class Flickr8kDataset(Dataset):
    """Custom dataset for ViT+GPT training"""
    
    def __init__(self, captions_df, images_dir, tokenizer, feature_extractor, max_length=50):
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.data = captions_df.copy()
        
        # Create image-caption pairs
        self.samples = []
        for _, row in self.data.iterrows():
            caption = self.preprocess_caption(row['caption'])
            self.samples.append((row['image'], caption))
    
    def preprocess_caption(self, caption):
        """Clean and preprocess caption text for GPT-2"""
        # Remove startseq and endseq tokens, add GPT-2 tokens
        caption = caption.replace('startseq', '').replace('endseq', '').strip()
        return f"<|startoftext|>{caption}<|endoftext|>"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        
        # Load and process image
        img_path = os.path.join(self.images_dir, img_name)
        if not os.path.exists(img_path):
            # Create a dummy image if file doesn't exist
            image = Image.new('RGB', (224, 224), color='black')
        else:
            image = Image.open(img_path).convert('RGB')
        
        # Extract image features using ViT
        image_inputs = self.feature_extractor(image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].squeeze(0)
        
        # Tokenize caption
        caption_tokens = self.tokenizer.encode(caption, max_length=self.max_length, 
                                             truncation=True, padding='max_length')
        
        return {
            'pixel_values': pixel_values,
            'caption_tokens': torch.tensor(caption_tokens, dtype=torch.long),
            'caption_text': caption
        }

def create_vit_loaders(train_data, val_data, IMAGE_PATH, tokenizer_vit, feature_extractor, batch_size=8):
    """Create and return train and validation loaders"""
    vit_train_dataset = Flickr8kDataset(train_data, IMAGE_PATH, tokenizer_vit, feature_extractor)
    vit_val_dataset = Flickr8kDataset(val_data, IMAGE_PATH, tokenizer_vit, feature_extractor)
    
    # Create data loaders with num_workers=0 for Jupyter compatibility
    vit_train_loader = DataLoader(vit_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    vit_val_loader = DataLoader(vit_val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return vit_train_loader, vit_val_loader