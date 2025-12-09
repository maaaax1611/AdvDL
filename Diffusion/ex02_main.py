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
    parser.add_argument('--test_only', action='store_true', default=False, help='run only inference/testing using a saved model')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['linear', 'cosine', 'sigmoid'], help='type of noise schedule to use (linear, cosine, sigmoid)')
    parser.add_argument('--guidance_scale', type=float, default=2.0, help='scale for classifier-free guidance during sampling')
    parser.add_argument('--unconditional', action='store_true', default=False, help='if set, trains a standard unconditional model without classes')
    return parser.parse_args()


def get_scheduler(name):
    """
    Returns the scheduler function based on the string name.
    """
    if name == "linear":
        return lambda x: linear_beta_schedule(0.0001, 0.02, x)
    elif name == "cosine":
        return cosine_beta_schedule
    elif name == "sigmoid":
        return lambda x: sigmoid_beta_schedule(0.0001, 0.02, x)
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def plot_labeled_grid(diffusor, model, device, store_path, n_images=16, transform=None, guidance_scale=2.0):
    """
    Generates a grid containing generated images and labels
    """
    model.eval()
    
    rows = int(np.sqrt(n_images))
    cols = int(np.ceil(n_images / rows))
    
    labels = torch.randint(0, 10, (n_images,), device=device).long()
    
    with torch.no_grad():
        images = diffusor.sample(
            model=model,
            image_size=diffusor.img_size,
            batch_size=n_images,
            channels=3,
            classes=labels,
            guidance_scale=guidance_scale
        )

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    axes = axes.flatten()
    
    fig.suptitle(f'Generated Samples (Scale: {guidance_scale})', fontsize=16)

    for i in range(n_images):
        ax = axes[i]
        if transform:
            img_pil = transform(images[i].cpu())
            ax.imshow(img_pil)
        else:
            img = (images[i].cpu().permute(1, 2, 0) + 1) * 0.5
            img = img.clamp(0, 1).numpy()
            ax.imshow(img)
            
        class_name = CLASSES[labels[i].item()]
        ax.set_title(class_name, fontsize=10)
        ax.axis('off')

    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(store_path)
    plt.close()
    print(f"Labeled grid saved to {store_path}")


def save_diffusion_animation(diffusor, model, device, store_path, reverse_transform, n_images=4, guidance_scale=2.0):
    """
    Generate a GIF to visualize sampling
    """
    model.eval()
    demo_classes = torch.randint(0, 10, (n_images,), device=device).long()

    with torch.no_grad():
        x_seq = diffusor.sample(
            model, 
            image_size=diffusor.img_size, 
            batch_size=n_images, 
            channels=3, 
            return_all_timesteps=True,
            classes=demo_classes,
            guidance_scale=guidance_scale
        )
    
    frames = []
    step_size = max(1, len(x_seq) // 50)
    
    for i in range(0, len(x_seq), step_size):
        batch_t = x_seq[i]
        pil_images = [reverse_transform(img.cpu()) for img in batch_t]
        
        # Grid construction logic simplified for animation
        w, h = pil_images[0].size
        rows = int(np.sqrt(n_images))
        cols = n_images // rows
        grid = Image.new('RGB', size=(cols*w, rows*h))
        for j, img in enumerate(pil_images):
            grid.paste(img, box=(j % cols * w, j // cols * h))
        frames.append(grid)

    if frames:
        frames.extend([frames[-1]] * 10)
        frames[0].save(store_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

def loss_plot(train_losses, val_losses, store_path):
    """
    Generate loss plot
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
    """
    Calculate validation loss
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for step, (images, labels) in enumerate(testloader):
            images = images.to(device)
            labels = labels.to(device)

            # sample random timesteps t
            t = torch.randint(0, args.timesteps, (len(images),), device=device).long()

            # calc loss
            # use p_losses method (copy from training)
            loss = diffusor.p_losses(
                denoise_model=model,
                x_zero=images,
                t=t,
                noise=None, # we want to generate random noise in eval (see p_losses)
                loss_type='l2',
                classes=labels,
                cond_drop_prob=0.0 # no label dropout during eval
            )
            total_loss += loss.item()

    avg_loss = total_loss / len(testloader)
    print(f"\nTest set: Average loss: {avg_loss:.4f}\n")
    return avg_loss


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    """
    Training loop
    """
    model.train()
    total_loss = 0
    pbar = tqdm(trainloader)
    
    # Check if we want to train strictly unconditionally
    use_classes = not args.unconditional
    
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device) if use_classes else None
        optimizer.zero_grad()

        t = torch.randint(0, args.timesteps, (len(images),), device=device).long()
        
        # If unconditional: classes=None, prob=0.0
        # If conditional: classes=labels, prob=0.1
        drop_prob = 0.1 if use_classes else 0.0
        
        loss = diffusor.p_losses(
            denoise_model=model, 
            x_zero=images, 
            t=t, 
            loss_type="l2",
            classes=labels,
            cond_drop_prob=drop_prob 
        )

        loss.backward()
        # Apparently cosine schedule produced exploding gradients (or smth. like this)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        if args.dry_run:
            break

    return total_loss / len(trainloader)


def test(args):
    """
    Standalone test function
    """
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"
    image_size = 32
    channels = 3
    print(f"--- Running Test on {device} ---")
    print(f"Using Scheduler: {args.scheduler}")
    print(f"Guidance Scale: {args.guidance_scale}")

    # Determine num_classes based on args
    num_classes = None if args.unconditional else 10
    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,), num_classes=num_classes).to(device)
    
    my_scheduler = get_scheduler(args.scheduler)
    diffusor = Diffusion(args.timesteps, my_scheduler, image_size, device)

    ckpt_dir = f"./Diffusion/models/{args.run_name}"
    ckpt_path = os.path.join(ckpt_dir, "ckpt_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "ckpt_last.pt")

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"WARNING: No checkpoint found at {ckpt_dir}. Testing with random weights.")
        return

    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    save_path = f"./Diffusion/results/{args.run_name}/test_samples_labeled_scale{args.guidance_scale}.png"
    plot_labeled_grid(
        diffusor=diffusor, 
        model=model, 
        device=device, 
        store_path=save_path, 
        n_images=25, 
        transform=reverse_transform, 
        guidance_scale=args.guidance_scale
    )
    
    anim_path = f"./Diffusion/results/{args.run_name}/test_animation.gif"
    save_diffusion_animation(
        diffusor, model, device, anim_path, reverse_transform, n_images=4, guidance_scale=args.guidance_scale
    )
    print(f"Done! Images saved to ./Diffusion/results/{args.run_name}/")


def run(args):
    """
    Train a model and safe visualizations
    """
    timesteps = args.timesteps
    image_size = 32
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    print(f"Starting Run: {args.run_name} on {device}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Guidance scale (used for sampling): {args.guidance_scale}")
    print(f"Training mode: {'UNCONDITIONAL' if args.unconditional else 'CONDITIONAL'}")

    # Model and Optim setup
    num_classes = None if args.unconditional else 10
    model = Unet(dim=image_size, 
                 channels=channels, 
                 dim_mults=(1, 2, 4,), 
                 num_classes=num_classes
            ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Diffusion setup
    #my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    my_scheduler = get_scheduler(args.scheduler)
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

    # set up directories
    ckpt_dir = f"./Diffusion/models/{args.run_name}"
    results_dir = f"./Diffusion/results/{args.run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

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
        loss_plot(train_losses, val_losses, os.path.join(results_dir, "losses.png"))

        # save imgs
        # sample random images and save them as a plot
        grid_path = os.path.join(results_dir, f"epoch_{epoch}_grid.png")
        plot_labeled_grid(
            diffusor=diffusor, 
            model=model, 
            device=device, 
            store_path=grid_path, 
            n_images=16, 
            transform=reverse_transform, 
            guidance_scale=args.guidance_scale
        )

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

    anim_path = os.path.join(results_dir, "diffusion_process.gif")
    save_diffusion_animation(
        diffusor=diffusor,
        model=model,
        device=device,
        store_path=anim_path,
        reverse_transform=reverse_transform,
        n_images=4,
        guidance_scale=args.guidance_scale
    )

    print("Training finished.")
    

if __name__ == '__main__':
    args = parse_args()
    if args.test_only:
        test(args)
    else:
        run(args)