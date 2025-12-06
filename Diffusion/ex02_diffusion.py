import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
    # So we choose a limit like 6 to cover the active range of the sigmoid function (approx -6 to +6)
    s_limit = 6 
    t = torch.linspace(0, timesteps, timesteps)
    
    # Equation from the exercise sheet (Eq. 10)
    sigmoid_input = -s_limit + (2 * t / timesteps) * s_limit
    sigmoid_values = torch.sigmoid(sigmoid_input)
    
    betas = beta_start + sigmoid_values * (beta_end - beta_start)
    return betas


class Diffusion:

    # TODO (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps)

        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help

        # define alphas
        # alpha_cumprod: product of all alphas at time step t
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # equation (4)
        self.sqrt_alphas_cumprd = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0): reverse process
        # equation (8)
        # 1 / sqrt(alpha_t)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # beta_t / sqrt(1 - alpha_cumprod_t)
        self.posterior_eps_coef = self.betas / self.sqrt_one_minus_alphas_cumprod

        # Variance term (sigma_t * z): we use sigma_t^2 = beta_t, so we need sqrt(beta_t)
        self.sqrt_betas = torch.sqrt(self.betas)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t. Note that x and t both have a batch dimension

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean

        # TODO (2.2): The method should return the image at timestep t-1.
        
        # get precomputed values
        beta_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # predict noise and subtract it from image --> model mean
        # equation (8)
        predicted_noise = model(x, t)
        model_mean = sqrt_recip_alphas_t * (x - beta_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise)

        if t_index == 0:
            return model_mean
        else:
            # add noise term
            # z ~ N(0, I) 
            posterior_variance_t = extract(self.sqrt_betas, t, x.shape)
            noise = torch.randn_like(x)
            # Full equation: mean + sigma * z
            return model_mean + posterior_variance_t * noise
        
    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing'' the noise in the generated image.
        # TODO (2.2): Return the generated images
        # we start with x_t ~ N(0, I): random normal noise
        x = torch.randn((batch_size, channels, image_size, image_size), device=self.device)

        # iterate backwards over all timesteps
        for timestep in reversed(range(self.timesteps)):
            # create tensor containing the current timestep for all samples in the batch
            t = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, timestep)

        return x

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        
        # this is important: 
        #   - during training we provide defined noise (to be able to compute a loss)
        #   - during testing we want to sample random noise
        if noise is None:
            noise = torch.randn_like(x_zero)
        
        # forward diffusion
        # equation (4)
        # use helper function 'extract' to get values at ts t
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprd, t, x_zero.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_zero.shape)

        x_t = sqrt_alphas_cumprod_t * x_zero + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t

    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1"):
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        # generate noise if not provided
        if noise is None:
            noise = torch.randn_like(x_zero)

        # compute noise image at timestep t x_t
        x_t = self.q_sample(x_zero=x_zero, t=t, noise=noise)

        # predict the noise using the model
        predicted_noise = denoise_model(x_t, t)

        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = F.l1_loss(predicted_noise, noise)
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = F.mse_loss(predicted_noise, noise)
        else:
            raise NotImplementedError()

        return loss
