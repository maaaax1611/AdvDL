import os
import argparse
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from models import ViT, CrossViT


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify CIFAR10')
    parser.add_argument('--model', type=str, default='r18', help='model to train (default: r18)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status (used by tqdm postfix)')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()

def train(model, trainloader, optimizer, criterion, device, epoch, args):
    model.train()
    
    pbar = tqdm(trainloader, total=len(trainloader), desc=f'Train Epoch {epoch}', leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)/len(output)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            pbar.set_postfix({'loss': loss.item()})
            
        if args.dry_run:
            break

def test(model, device, test_loader, criterion, set="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    
    pbar = tqdm(test_loader, total=len(test_loader), desc=f'{set} Set', leave=False)
    
    with torch.no_grad():
        for data, target in pbar:  # Loop over the progress bar
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


def run(args):
    # Download and load the training data
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    # ImageNet mean/std values should also fit okayish for CIFAR
									transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ])
    # create a separate transforms pipelien for testing/validation purposes
    transform_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                        ])
    # adjust folder
    dataset = datasets.CIFAR10('./data', download=True, train=True, transform=transform_train)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])

    # override valset transforms
    valset.dataset.transform = transform_test

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('./data', download=True, train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # Build a feed-forward network
    print(f"Using {args.model}")
    if args.model == "r18":
        model = models.resnet18(pretrained=False)
    elif args.model == "vit":
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

    # Define the loss
    criterion = nn.CrossEntropyLoss(reduction="sum")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # folder to save models
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    # Track best model
    best_accuracy = 0.0
    best_model_path = os.path.join(model_dir, f'checkpoint_{args.model}.pt')

    # --- Epoch Loop ---
    for epoch in range(1, args.epochs + 1):
        train(model, trainloader, optimizer, criterion, device, epoch, args)
        val_accuracy = test(model, device, valloader, criterion, set="Validation")
        
        # Save model if validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(f'New best model! Validation accuracy: {val_accuracy:.2f}%')
            if args.save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                }, best_model_path)

    # Load best model for final test
    if args.save_model:
        if os.path.exists(best_model_path):
            # Add weights_only=True to address the security warning
            checkpoint = torch.load(best_model_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'\nLoaded best model from epoch {checkpoint["epoch"]} with validation accuracy: {checkpoint["accuracy"]:.2f}%')
        else:
            print('\nNo model was loaded. Running tests with model from last epoch!')

    test(model, device, testloader, criterion)

if __name__ == '__main__':
    args = parse_args()
    run(args)