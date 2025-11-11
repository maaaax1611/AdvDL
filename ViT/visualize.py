import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse # Import argparse

# Import model definitions
from models import ViT, CrossViT

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize ViT/CrossViT Attention')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'cvit'], 
                        help='Model to visualize (default: vit)')
    parser.add_argument('--image_index', type=int, default=0, 
                        help='Image index from test set to visualize (default: 0)')
    return parser.parse_args()

# --- 1. Define Un-normalization Function ---
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m) # t = t * s + m
        return tensor

def visualize():
    args = parse_args()
    
    # --- 2. Load Data ---
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    testset = datasets.CIFAR10('./data', download=True, train=False, transform=transform_test)
    img, label = testset[args.image_index]
    img_batch = img.unsqueeze(0) 

    # --- 3. Load Model ---
    model = None
    last_attn_layer = None
    PATCH_SIZE = 8 # Default patch size
    IMAGE_SIZE = 32

    if args.model == 'vit':
        print("Visualizing: ViT")
        MODEL_PATH = 'models/checkpoint_vit.pt'
        PATCH_SIZE = 8 # As defined in main.py
        model = ViT(image_size = IMAGE_SIZE, patch_size = PATCH_SIZE, num_classes = 10, dim = 64,
                    depth = 2, heads = 8, mlp_dim = 128, dropout = 0.1,
                    emb_dropout = 0.1) 
        
        try:
            checkpoint = torch.load(MODEL_PATH, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded 'vit' model from epoch {checkpoint.get('epoch', 'N/A')} with accuracy {checkpoint.get('accuracy', 'N/A')}%")
        except Exception as e:
            print(f"Error loading model '{MODEL_PATH}': {e}")
            exit()
        
        # Define path to attention layer
        try:
            last_attn_layer = model.transformer.layers[-1][0].fn
        except AttributeError:
            print("Could not find 'self.attn' in ViT. Did you modify 'models.py' correctly?")
            exit()

    elif args.model == 'cvit':
        print("Visualizing: CrossViT (Small Branch Self-Attention)")
        MODEL_PATH = 'models/checkpoint_cvit.pt'
        # We visualize the small branch, which uses sm_patch_size = 8
        PATCH_SIZE = 8 # sm_patch_size from main.py
        
        # Load CrossViT with all params from main.py
        model = CrossViT(image_size = IMAGE_SIZE, num_classes = 10, sm_dim = 64, 
                         lg_dim = 128, sm_patch_size = PATCH_SIZE, sm_enc_depth = 2,
                         sm_enc_heads = 8, sm_enc_mlp_dim = 128, 
                         sm_enc_dim_head = 64, lg_patch_size = 16, 
                         lg_enc_depth = 2, lg_enc_heads = 8, 
                         lg_enc_mlp_dim = 128, lg_enc_dim_head = 64,
                         cross_attn_depth = 2, cross_attn_heads = 8,
                         cross_attn_dim_head = 64, depth = 3, dropout = 0.1,
                         emb_dropout = 0.1)

        try:
            checkpoint = torch.load(MODEL_PATH, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded 'cvit' model from epoch {checkpoint.get('epoch', 'N/A')} with accuracy {checkpoint.get('accuracy', 'N/A')}%")
        except Exception as e:
            print(f"Error loading model '{MODEL_PATH}': {e}")
            exit()

        # Define path to attention layer (small branch)
        try:
            # model -> multi_scale_encoder -> last layer -> small_transformer (idx 0)
            sm_transformer = model.multi_scale_encoder.layers[-1][0]
            # small_transformer -> last layer -> PreNorm(Attn) (idx 0) -> .fn
            last_attn_layer = sm_transformer.layers[-1][0].fn
        except AttributeError:
            print("Could not find 'self.attn' in CrossViT. Did you modify 'models.py' correctly?")
            exit()

    model.eval()

    # --- 4. Run Forward Pass & Get Attention ---
    with torch.no_grad():
        output = model(img_batch)
    
    attn_map = last_attn_layer.attn # Get the stored attention map
    print(f"Successfully extracted attention map of shape: {attn_map.shape}")

    # --- 5. Process Attention Map ---
    # Shape of attn_map: [b, h, n, m] -> [1, 8, 17, 17] (since 32/8=4 -> 4x4=16 patches + 1 CLS)
    
    # Get attention from CLS token (idx 0) to all patch tokens (idx 1:)
    attn_cls = attn_map[0, :, 0, 1:] # Shape: [8, 16]
    avg_attn = torch.mean(attn_cls, dim=0) # Shape: [16]
    
    max_patch_idx = avg_attn.argmax().item()
    print(f"Patch with the biggest impact: index {max_patch_idx}")

    # Reshape the 1D patch attention vector [16] into a 2D grid
    patch_grid_size = int(np.sqrt(avg_attn.shape[0])) # sqrt(16) = 4
    attn_grid = avg_attn.reshape(patch_grid_size, patch_grid_size) # Shape: [4, 4]

    # --- 6. Visualize ---
    heatmap = F.interpolate(
        attn_grid.unsqueeze(0).unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode='bilinear',
        align_corners=False
    ).squeeze() # Shape: [32, 32]

    img_to_plot = unorm(img).permute(1, 2, 0) # (C, H, W) -> (H, W, C)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

    ax1.imshow(img_to_plot)
    ax1.set_title(f"Original Image (Label: {testset.classes[label]})")
    ax1.axis('off')

    ax2.imshow(img_to_plot)
    ax2.imshow(heatmap, alpha=0.5, cmap='jet')
    ax2.set_title(f"Attention Heatmap (Model: {args.model.upper()})")
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(f'attention_map_{args.model}_img{args.image_index}.png')
    print(f"Saved visualization to 'attention_map_{args.model}_img{args.image_index}.png'")
    plt.show()

if __name__ == "__main__":
    visualize()