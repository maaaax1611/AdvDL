import torch
import matplotlib.pyplot as plt
import numpy as np
from ex02_diffusion import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule

def test_schedulers():
    timesteps = 1000  # Dein Setup
    
    # 1. Berechne Betas für alle Schedules
    betas_linear = linear_beta_schedule(0.0001, 0.02, timesteps)
    betas_cosine = cosine_beta_schedule(timesteps)
    betas_sigmoid = sigmoid_beta_schedule(0.0001, 0.02, timesteps) # Falls implementiert

    # 2. Berechne Alphas Cumprod (Das ist wichtig für das Signal-Rausch-Verhältnis)
    # alpha = 1 - beta
    # alpha_bar = cumprod(alpha)
    
    def get_alphas_cumprod(betas):
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod

    alphas_bar_lin = get_alphas_cumprod(betas_linear)
    alphas_bar_cos = get_alphas_cumprod(betas_cosine)
    alphas_bar_sig = get_alphas_cumprod(betas_sigmoid)

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))

    # Plot 1: Betas (Wie viel Rauschen kommt pro Schritt dazu?)
    plt.subplot(1, 2, 1)
    plt.plot(betas_linear.numpy(), label='Linear', linestyle='--')
    plt.plot(betas_cosine.numpy(), label='Cosine')
    plt.plot(betas_sigmoid.numpy(), label='Sigmoid')
    plt.title("Betas (Variance per Step)")
    plt.xlabel("Timestep t")
    plt.ylabel("Beta")
    plt.legend()
    plt.grid(True)

    # Plot 2: Alphas Cumprod (Wieviel "Originalbild" ist noch übrig?)
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

    # --- KRITISCHE WERTE PRÜFEN ---
    print(f"--- Check numerische Stabilität (T={timesteps}) ---")
    
    print(f"\nLINEAR:")
    print(f"  Min Beta: {betas_linear.min().item():.6f}")
    print(f"  Max Beta: {betas_linear.max().item():.6f}")
    print(f"  Final Alpha_Bar (sollte > 0 sein): {alphas_bar_lin[-1].item():.10f}")

    print(f"\nCOSINE:")
    print(f"  Min Beta: {betas_cosine.min().item():.6f}")
    print(f"  Max Beta: {betas_cosine.max().item():.6f} (WICHTIG: Sollte < 0.999 sein!)")
    print(f"  Final Alpha_Bar: {alphas_bar_cos[-1].item():.10f}")
    
    
if __name__ == "__main__":
    test_schedulers()