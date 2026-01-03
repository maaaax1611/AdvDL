import os

experiments = [
    {
        "run_name": "DDPM_Cosine_Guidance2_epochs200_bs128",
        "scheduler": "cosine",
        "epochs": 200,
        "lr": 0.0002,
        "extra_args": "--guidance_scale 2.0 " " --batch_size 128"
    },
    {
        "run_name": "DDPM_Sigmoid_Guidance2_epochs200_bs128",
        "scheduler": "sigmoid",
        "epochs": 200,
        "lr": 0.0002,
        "extra_args": "--guidance_scale 2.0 " " --batch_size 128"
    }
]

for i, exp in enumerate(experiments):
    print(f"\n{'='*40}")
    print(f"STARTING EXPERIMENT {i+1}/{len(experiments)}: {exp['run_name']}")
    print(f"{'='*40}\n")
    
    command = (
        f"python .\\Diffusion\\ex02_main.py "
        f"--run_name {exp['run_name']} "
        f"--scheduler {exp['scheduler']} "
        f"--epochs {exp['epochs']} "
        f"--lr {exp['lr']} "
        f"--timesteps 1000 "
        f"{exp['extra_args']}"
    )
    
    print(f"Executing: {command}")
    
    exit_code = os.system(command)
    
    if exit_code != 0:
        print(f"WARNING: Experiment {exp['run_name']} failed!")
    else:
        print(f"Experiment {exp['run_name']} finished.")

print("\nALL EXPERIMENTS DONE!")