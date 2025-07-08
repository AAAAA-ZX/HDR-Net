import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(sr, hr):
    sr = sr.clamp(0,1).cpu().detach().numpy()
    hr = hr.clamp(0,1).cpu().detach().numpy()
    return peak_signal_noise_ratio(sr, hr)

def calculate_ssim(sr, hr):
    sr = sr.squeeze().cpu().detach().numpy().transpose(1,2,0)
    hr = hr.squeeze().cpu().detach().numpy().transpose(1,2,0)
    return structural_similarity(sr, hr, multichannel=True)