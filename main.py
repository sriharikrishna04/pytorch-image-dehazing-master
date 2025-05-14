import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import wandb
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from data import MyDataset
from model import Generator, Discriminator
from vgg16 import Vgg16
from utils import ImagePool, SaveData, PSNR
import argparse
from skimage.transform import resize


def get_args():
    parser = argparse.ArgumentParser(description='image-dehazing')
    parser.add_argument('--data_dir', type=str, default='dataset/indoor', help='dataset directory')
    parser.add_argument('--save_dir', default='results', help='data save directory')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--n_threads', type=int, default=8, help='number of threads for data loading')
    parser.add_argument('--exp', default='Net1', help='model to select')
    parser.add_argument('--p_factor', type=float, default=0.5, help='perceptual loss factor')
    parser.add_argument('--g_factor', type=float, default=0.5, help='gan loss factor')
    parser.add_argument('--glr', type=float, default=1e-4, help='generator learning rate')
    parser.add_argument('--dlr', type=float, default=1e-4, help='discriminator learning rate')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train')
    parser.add_argument('--lr_step_size', type=int, default=2000, help='period of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='multiplicative factor of learning rate decay')
    parser.add_argument('--patch_gan', type=int, default=30, help='Patch GAN size')
    parser.add_argument('--pool_size', type=int, default=50, help='Buffer size for storing generated samples from G')
    parser.add_argument('--period', type=int, default=1, help='period of printing logs')
    parser.add_argument('--gpu', type=int, required=True, help='gpu index')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint file')
    return parser.parse_args()


def compute_metrics(netG, dataloader, device):
    """ Compute PSNR and SSIM on test dataset """
    netG.eval()
    psnr_sum, ssim_sum, batch_count = 0, 0, 0

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    # Set data_range as per your image format

    with torch.no_grad():
        for images in tqdm(dataloader, total=len(dataloader), desc="Evaluating PSNR & SSIM"):
            input_image, target_image = images
            input_image, target_image = input_image.to(device), target_image.to(device)

            output_image = netG(input_image)

            # Compute PSNR
            psnr_value = PSNR(target_image.cpu().numpy(), output_image.cpu().numpy())

            # Compute SSIM using torchmetrics
            ssim_value = ssim_metric(output_image, target_image)

            psnr_sum += psnr_value
            ssim_sum += ssim_value.item()
            batch_count += 1

    return psnr_sum / batch_count, ssim_sum / batch_count


def train(args):
    print(args)
    wandb.init(project="image-dehazing", entity="sriharikrishnacbe04-psg-college-of-technology", config=vars(args), resume="allow")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG, netD = Generator().to(device), Discriminator().to(device)

    # Loss functions
    l1_loss, l2_loss, bce_loss = nn.L1Loss().to(device), nn.MSELoss().to(device), nn.BCELoss().to(device)

    # Optimizers
    optimizerG, optimizerD = optim.Adam(netG.parameters(), lr=args.glr), optim.Adam(netD.parameters(), lr=args.dlr)
    schedulerG, schedulerD = lr_scheduler.StepLR(optimizerG, step_size=args.lr_step_size, gamma=args.lr_gamma), lr_scheduler.StepLR(optimizerD, step_size=args.lr_step_size, gamma=args.lr_gamma)

    save = SaveData(args.save_dir, args.exp, True)
    save.save_params(args)

    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        netG.load_state_dict(checkpoint['netG'])
        netD.load_state_dict(checkpoint['netD'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        start_epoch = checkpoint['epoch'] + 1

    dataset = MyDataset(args.data_dir, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(50)), batch_size=1, shuffle=False)

    vgg, image_pool = Vgg16(requires_grad=False).to(device), ImagePool(args.pool_size)

    for epoch in range(start_epoch, args.epochs):
        netG.train()
        netD.train()

        d_total_loss, g_total_loss = 0, 0

        for images in tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}"):
            input_image, target_image = images
            input_image, target_image = input_image.to(device), target_image.to(device)

            output_image = netG(input_image)

            netD.requires_grad_(True)
            netD.zero_grad()

            real_loss = bce_loss(netD(target_image), torch.ones_like(netD(target_image)))
            fake_loss = bce_loss(netD(image_pool.query(output_image.detach())), torch.zeros_like(netD(output_image)))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizerD.step()
            d_total_loss += d_loss.item()

            netD.requires_grad_(False)
            netG.zero_grad()

            g_res_loss = l1_loss(output_image, target_image)
            g_per_loss = args.p_factor * l2_loss(vgg(output_image), vgg(target_image))
            g_gan_loss = args.g_factor * bce_loss(netD(output_image), torch.ones_like(netD(output_image)))
            g_loss = g_res_loss + g_per_loss + g_gan_loss
            g_loss.backward()
            optimizerG.step()
            g_total_loss += g_loss.item()

        avg_d_loss = d_total_loss / len(dataloader)
        avg_g_loss = g_total_loss / len(dataloader)

        # Log loss functions
        wandb.log({"Discriminator Loss": avg_d_loss, "Generator Loss": avg_g_loss, "Epoch": epoch})

        schedulerG.step()
        schedulerD.step()

        if epoch % args.period == 0:
            save.save_model(netG, netD, epoch, optimizerG, optimizerD, schedulerG, schedulerD)

            # Compute PSNR and SSIM at checkpoint
            psnr, ssim_score = compute_metrics(netG, test_dataloader, device)
            wandb.log({"PSNR": psnr, "SSIM": ssim_score, "Epoch": epoch})

    wandb.finish()


if __name__ == '__main__':
    args = get_args()
    train(args)
