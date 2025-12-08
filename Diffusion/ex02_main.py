import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from torchvision.utils import save_image, make_grid

import argparse

# CIFAR 10 label list for plotting
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()


# def save_diffusion_animation(diffusor, model, device, store_path, n_images=4):
#     """
#     Generates a GIF animation of the process. (Task 2.2 f - Optional)
#     """
#     model.eval()
#     os.makedirs(os.path.dirname(store_path), exist_ok=True)
    
#     # Needs return_all_timesteps implemented in Diffusion.sample
#     with torch.no_grad():
#         # Check if sample supports return_all_timesteps (it should if you implemented it)
#         try:
#             all_images = diffusor.sample(model, image_size=diffusor.img_size, batch_size=n_images, channels=3, return_all_timesteps=True)
#         except TypeError:
#             print("Warning: return_all_timesteps not implemented in Diffusion.sample yet. Skipping animation.")
#             return

#     # Post-processing
#     all_images = (all_images + 1) * 0.5
#     all_images = all_images.clamp(0, 1)
    
#     frames = []
#     step_size = 10 if all_images.shape[0] > 200 else 1 # Downsample frames
    
#     for t in range(0, all_images.shape[0], step_size):
#         batch_t = all_images[t]
#         grid = make_grid(batch_t, nrow=int(np.sqrt(n_images)), padding=2)
#         ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
#         frames.append(Image.fromarray(ndarr))
        
#     if frames:
#         frames[0].save(store_path, save_all=True, append_images=frames[1:], duration=20, loop=0)



# def sample_and_save_images(n_images, diffusor, model, device, store_path):
#     # TODO: Implement - adapt code and method signature as needed
#     model.eval()

#     os.makedirs(os.path.dirname(store_path), exist_ok=True)

#     print(f"Sampling {n_images} images...")
#     with torch.no_grad():
#         sampled_images = diffusor.sample(
#             model=model,
#             image_size=diffusor.img_size,
#             batch_size=n_images,
#             channels=3
#         )
#         # we need to do some postprocessing
#         # images should be in [-1, 1] so map them back to [0, 1] for saving
#         sampled_images = (sampled_images + 1) * 0.5
#         sampled_images = sampled_images.clamp(0, 1)

#         # save them as a grid
#         save_image(sampled_images, store_path, nrow=int(np.sqrt(n_images)))

#     print(f"Images saved to {store_path}")


# def comparison_plot(diffusor, model, device, real_batch, real_labels, store_path, epoch=None):
#     """
#     Saves a plot comparing Real Images (Top) vs. Generated Images (Bottom).
#     Includes Class Labels for Real Images.
#     """
#     model.eval()
#     n_images = 8  # Wir zeigen 8 echte und 8 generierte
    
#     # 1. Generieren
#     with torch.no_grad():
#         fake_images = diffusor.sample(model, image_size=diffusor.img_size, batch_size=n_images, channels=3)
    
#     # Post-processing ([-1, 1] -> [0, 1])
#     fake_images = (fake_images + 1) * 0.5
#     fake_images = fake_images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy() # (B, H, W, C)
    
#     # Real Images vorbereiten (wir nehmen die ersten n_images vom Batch)
#     real_images = real_batch[:n_images].to(device)
#     real_images = (real_images + 1) * 0.5
#     real_images = real_images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    
#     real_labels = real_labels[:n_images].cpu().numpy()

#     # 2. Plotting mit Matplotlib
#     fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 5))
    
#     # Titel für das ganze Bild
#     title = f"Epoch {epoch}" if epoch is not None else "Final Samples"
#     fig.suptitle(f'Top: Real Data | Bottom: Generated (Unconditional) - {title}', fontsize=16)

#     for i in range(n_images):
#         # --- Top Row: Real Images ---
#         axes[0, i].imshow(real_images[i])
#         axes[0, i].set_title(CLASSES[real_labels[i]])
#         axes[0, i].axis('off')

#         # --- Bottom Row: Generated Images ---
#         axes[1, i].imshow(fake_images[i])
#         axes[1, i].set_title("Generated")
#         axes[1, i].axis('off')

#     plt.tight_layout()
#     plt.savefig(store_path)
#     plt.close()
#     print(f"Comparison plot saved to {store_path}")

def save_diffusion_animation(diffusor, model, device, store_path, n_images=4):
    """
    Generates a GIF animation of the process.
    """
    model.eval()
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    
    with torch.no_grad():
        try:
            all_images = diffusor.sample(model, image_size=diffusor.img_size, batch_size=n_images, channels=3, return_all_timesteps=True)
        except TypeError:
            print("Warning: return_all_timesteps not implemented. Skipping animation.")
            return

    # Post-processing for animation (keeping simple manual clamp here for speed on tensor batches)
    all_images = (all_images + 1) * 0.5
    all_images = all_images.clamp(0, 1)
    
    frames = []
    step_size = 10 if all_images.shape[0] > 200 else 1
    
    for t in range(0, all_images.shape[0], step_size):
        batch_t = all_images[t]
        grid = make_grid(batch_t, nrow=int(np.sqrt(n_images)), padding=2)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        frames.append(Image.fromarray(ndarr))
        
    if frames:
        frames[0].save(store_path, save_all=True, append_images=frames[1:], duration=20, loop=0)


def sample_and_save_images(n_images, diffusor, model, device, store_path, transform=None):
    """
    Samples images and uses the provided 'transform' to convert them to PIL images,
    then stitches them into a grid.
    """
    model.eval()
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    
    print(f"Sampling {n_images} images...")
    with torch.no_grad():
        sampled_images = diffusor.sample(
            model=model,
            image_size=diffusor.img_size,
            batch_size=n_images,
            channels=3
        )
        
        # Use the specific reverse_transform provided
        if transform is not None:
            # transform expects a single image (C,H,W), so we iterate
            pil_images = [transform(img.cpu()) for img in sampled_images]
            
            # Manually stitch PIL images into a grid
            rows = int(np.sqrt(n_images))
            cols = n_images // rows
            w, h = pil_images[0].size
            grid = Image.new('RGB', size=(cols*w, rows*h))
            
            for i, img in enumerate(pil_images):
                grid.paste(img, box=(i % cols * w, i // cols * h))
            
            grid.save(store_path)
        else:
            # Fallback if no transform provided
            sampled_images = (sampled_images + 1) * 0.5
            save_image(sampled_images, store_path, nrow=int(np.sqrt(n_images)))

    print(f"Images saved to {store_path}")


def comparison_plot(diffusor, model, device, real_batch, real_labels, store_path, epoch=None, transform=None):
    """
    Saves a plot comparing Real Images (Top) vs. Generated Images (Bottom).
    Uses 'transform' to convert tensors to PIL images for plotting.
    """
    model.eval()
    n_images = 8
    
    # 1. Generate
    with torch.no_grad():
        fake_images = diffusor.sample(model, image_size=diffusor.img_size, batch_size=n_images, channels=3)
    
    # 2. Plotting
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 5))
    
    title = f"Epoch {epoch}" if epoch is not None else "Final Samples"
    fig.suptitle(f'Top: Real Data | Bottom: Generated (Cosine) - {title}', fontsize=16)

    for i in range(n_images):
        # --- Top Row: Real Images ---
        # Use the reverse_transform
        if transform:
            real_img_pil = transform(real_batch[i].cpu())
            axes[0, i].imshow(real_img_pil)
        else:
            # Fallback
            real_img = (real_batch[i] + 1) * 0.5
            axes[0, i].imshow(real_img.cpu().permute(1, 2, 0).numpy())
            
        axes[0, i].set_title(CLASSES[real_labels[i]])
        axes[0, i].axis('off')

        # --- Bottom Row: Generated Images ---
        if transform:
            fake_img_pil = transform(fake_images[i].cpu())
            axes[1, i].imshow(fake_img_pil)
        else:
            # Fallback
            fake_img = (fake_images[i] + 1) * 0.5
            axes[1, i].imshow(fake_img.cpu().permute(1, 2, 0).numpy())
            
        axes[1, i].set_title("Generated")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(store_path)
    plt.close()
    print(f"Comparison plot saved to {store_path}")


def loss_plot(train_losses, val_losses, store_path):
    """
    Plots the training and validation loss curves.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(val_losses, label="Validation Loss", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(store_path)
    plt.close()


def evaluate(model, testloader, diffusor, device, args):
    # TODO: Implement - adapt code and method signature as needed
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for step, (images, _) in enumerate(testloader):
            images = images.to(device)

            # sample random timesteps t
            t = torch.randint(0, args.timesteps, (len(images),), device=device).long()

            # calc loss
            # use p_losses method (copy from training)
            loss = diffusor.p_losses(model, images, t, loss_type ="l2")
            total_loss += loss.item()

    avg_loss = total_loss / len(testloader)
    print(f"\nTest set: Average loss: {avg_loss:.4f}\n")
    return avg_loss


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps
    total_loss = 0

    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, loss_type="l2")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        if args.dry_run:
            break

    avg_loss = total_loss / len(trainloader)
    return avg_loss


def test(args):
    """
    Standalone test function (Task 2.2 e)
    """
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"
    image_size = 32
    channels = 3
    print(f"--- Running Test on {device} ---")

    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
    # Ensure these match training!
    my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    diffusor = Diffusion(args.timesteps, my_scheduler, image_size, device)

    ckpt_path = os.path.join(f"./models/{args.run_name}", "ckpt.pt")
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"WARNING: No checkpoint found at {ckpt_path}. Testing with random weights.")

    transform = Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    # Local path ./data
    testset = datasets.CIFAR10('./data', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    avg_loss = evaluate(model, testloader, diffusor, device, args)
    print(f"Test Set Average Loss: {avg_loss:.6f}")

    save_path = f"./results/{args.run_name}/test_samples.png"
    sample_and_save_images(n_images=16, diffusor=diffusor, model=model, device=device, store_path=save_path)
    print(f"Test samples saved to {save_path}")


def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    print(f"Starting Run: {args.run_name} on {device}")

    # Model and Optim setup
    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Diffusion setup
    #my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    my_scheduler = cosine_beta_schedule
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10('./data', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('./data', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    # get labels and images for comparison plot
    fix_real_images, fix_real_labels = next(iter(valloader))

    ckpt_dir = f"./models/{args.run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(f"./results/{args.run_name}", exist_ok=True)

    # loss lists
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        #training
        avg_train_loss = train(model, trainloader, optimizer, diffusor, epoch, device, args)
        train_losses.append(avg_train_loss)

        # validation
        avg_val_loss = evaluate(model, valloader, diffusor, device, args)
        val_losses.append(avg_val_loss)

        # plot loss and save
        loss_plot(train_losses, val_losses, f"./results/{args.run_name}/losses.png")

        # save imgs
        # sample random images and save them as a plot
        grid_path = f"./results/{args.run_name}/epoch_{epoch}_grid.png"
        sample_and_save_images(n_images=9, diffusor=diffusor, model=model, device=device, store_path=grid_path, transform=reverse_transform)

        # create a coparison plot original images vs. generated images
        comp_path = f"./results/{args.run_name}/epoch_{epoch}_comparison.png"
        comparison_plot(diffusor, model, device, fix_real_images, fix_real_labels, comp_path, epoch=epoch, transform=reverse_transform)

        # Checkpoint strategy:
        # always safe the current state of the model
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"ckpt_last.pt"))

        # save another model (only when val loss decreases)
        if avg_val_loss < best_val_loss:
            print(f"New Best Model! Loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "ckpt_best.pt"))
    
    print("\n--- Final Evaluation on Test Set ---")
    evaluate(model, testloader, diffusor, device, args)

    anim_path = f"./results/{args.run_name}/training_process.gif"
    save_diffusion_animation(diffusor, model, device, anim_path, n_images=9)

    print("Training finished.")
    


if __name__ == '__main__':
    args = parse_args()
    # TODO (2.2): Add visualization capabilities
    run(args)
