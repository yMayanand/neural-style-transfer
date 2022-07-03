import os
import argparse
from tqdm import tqdm
from utils import load_image, preprocess_image, save_image
from model import VGG16
from loss import ContentLoss, StyleLoss

import torch
import torch.optim as optim

parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_dir', type=str, help='path to image directory')
parser.add_argument('--style_image', type=str, help='path to style image')
parser.add_argument('--content_image', type=str, help='path to content image')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--steps', type=int, default=50,
                    help='number of optimization steps')
parser.add_argument('--style_size', type=int,
                    default=256, help='style image size')
parser.add_argument('--content_size', type=int,
                    default=256, help='content image size')
parser.add_argument('--content_weight', type=float,
                    default=1e5, help='content weight for loss')
parser.add_argument('--style_weight', type=float,
                    default=1e10, help='style weight for loss')
parser.add_argument('--log_interval', type=int,
                    default=10, help='logging interval')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # path to images
    style_img_path = os.path.join(args.data_dir, args.style_image)
    content_img_path = os.path.join(args.data_dir, args.content_image)

    style_image = load_image(
        style_img_path,
        shape=(args.style_size, args.style_size)
    )

    content_image = load_image(
        content_img_path,
        shape=(args.content_size, args.content_size)
    )

    # preprocess image
    style_tensor = preprocess_image(style_image).to(device)
    content_tensor = preprocess_image(content_image).to(device)

    # initialising image that will be optimized
    opt_image = content_tensor.clone().to(device)
    opt_image.requires_grad = True

    # optimizer
    optimizer = optim.Adam([opt_image], lr=args.lr)

    # model
    vgg = VGG16(requires_grad=False).to(device)

    vgg.eval()
    target_style_features = vgg(style_tensor)
    vgg.eval()
    target_content_features = vgg(content_tensor).relu2_2

    # loss functions

    content_loss = ContentLoss(content_weight=args.content_weight)
    style_loss = StyleLoss(style_weight=args.style_weight, reduction='sum')

    for step in tqdm(range(1, args.steps + 1)):
        vgg.eval()
        image_style_features = vgg(opt_image)
        image_content_features = image_style_features.relu2_2

        s_l = 0
        c_l = 0

        for y, x in zip(target_style_features, image_style_features):
            s_l += style_loss(y, x)

        c_l = content_loss(target_content_features, image_content_features)

        loss = c_l + s_l

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step % args.log_interval) == 0:
            print(f"step: {step}, \
            style_loss: {s_l.item():,}, \
            content_loss: {c_l.item():,}, \
            total_loss: {loss.item():,}")
    
    # saving image
    os.makedirs('./outputs', exist_ok=True)
    save_image('./outputs/img.jpg', opt_image)

args = parser.parse_args()

main(args)
