import torch
import matplotlib.pyplot as plt
import numpy as np
from ex02_diffusion import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule

def test_schedulers():
    timesteps = 1000
    
    betas_linear = linear_beta_schedule(0.0001, 0.02, timesteps)
    betas_cosine = cosine_beta_schedule(timesteps)
    betas_sigmoid = sigmoid_beta_schedule(0.0001, 0.02, timesteps) # Falls implementiert

    
    def get_alphas_cumprod(betas):
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod

    alphas_bar_lin = get_alphas_cumprod(betas_linear)
    alphas_bar_cos = get_alphas_cumprod(betas_cosine)
    alphas_bar_sig = get_alphas_cumprod(betas_sigmoid)

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(betas_linear.numpy(), label='Linear', linestyle='--')
    plt.plot(betas_cosine.numpy(), label='Cosine')
    plt.plot(betas_sigmoid.numpy(), label='Sigmoid')
    plt.title("Betas (Variance per Step)")
    plt.xlabel("Timestep t")
    plt.ylabel("Beta")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(alphas_bar_lin.numpy(), label='Linear', linestyle='--')
    plt.plot(alphas_bar_cos.numpy(), label='Cosine')
    plt.plot(alphas_bar_sig.numpy(), label='Sigmoid')
    plt.title("Alphas Cumprod (Signal remaining)")
    plt.xlabel("Timestep t")
    plt.ylabel("Alpha Bar")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("scheduler_comparison.png")
    plt.show()

if __name__ == "__main__":
    test_schedulers()