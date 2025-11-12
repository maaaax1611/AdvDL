import os
import argparse
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm # tqdm is useful here too

# Import your model definitions
from models import ViT, CrossViT

# Take the 'test' function 1:1 from main.py
def test(model, device, test_loader, criterion, set="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    
    # (Code is identical to main.py)
    pbar = tqdm(test_loader, total=len(test_loader), desc=f'{set} Set', leave=False)
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return accuracy

def run_evaluation():
    parser = argparse.ArgumentParser(description='Evaluate a trained ViT/CrossViT model')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'cvit'], 
                        help='Model to evaluate (default: vit)')
    args = parser.parse_args()

    # --- 1. Load only the Test-Data ---
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    testset = datasets.CIFAR10('./data', download=True, train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    # --- 2. Build the Model Skeleton ---
    # (This block is the same as in main.py)
    model = None
    if args.model == "vit":
        model = ViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64,
                    depth = 2, heads = 8, mlp_dim = 128, dropout = 0.1,
                    emb_dropout = 0.1) 
    elif args.model == "cvit":
        model = CrossViT(image_size = 32, num_classes = 10, sm_dim = 64, 
                         lg_dim = 128, sm_patch_size = 8, sm_enc_depth = 2,
                         sm_enc_heads = 8, sm_enc_mlp_dim = 128, 
                         sm_enc_dim_head = 64, lg_patch_size = 16, 
                         lg_enc_depth = 2, lg_enc_heads = 8, 
                         lg_enc_mlp_dim = 128, lg_enc_dim_head = 64,
                         cross_attn_depth = 2, cross_attn_heads = 8,
                         cross_attn_dim_head = 64, depth = 3, dropout = 0.1,
                         emb_dropout = 0.1)

    # --- 3. Load the Weights ---
    criterion = nn.CrossEntropyLoss(reduction="sum")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_path = os.path.join('models', f'checkpoint_{args.model}.pt')

    if os.path.exists(model_path):
        # Load the weights
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded: {model_path}")
        print(f"(Using model from epoch {checkpoint.get('epoch', 'N/A')} with {checkpoint.get('accuracy', 'N/A')}% val-acc)")
        
        # --- 4. Run the Test ---
        print("\nStarting evaluation on the test set...")
        test(model, device, testloader, criterion)
    else:
        print(f"Error: Model file not found at: {model_path}")
        print(f"Please train the '{args.model}' model first using 'main.py --save-model'.")

if __name__ == '__main__':
    run_evaluation()