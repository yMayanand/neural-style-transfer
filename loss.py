import torch
import torch.nn as nn
from utils import gram_matrix

class ContentLoss(nn.Module):
    """computes content loss for neural style transfer task"""
    
    def __init__(self, content_weight=1e5):
        super().__init__()

        self.content_weight = content_weight
        self.aux_loss = nn.MSELoss()

    def forward(self, y, y_target):
        loss = self.aux_loss(y, y_target) * self.content_weight
        return loss

class StyleLoss(nn.Module):
    """computes style loss for neural style transfer task"""

    def __init__(self, style_weight=1e3):
        super().__init__()

        self.style_weight = style_weight
        self.aux_loss = nn.MSELoss()

    def forward(self, y, y_target):
        gram_y = gram_matrix(y)
        gram_y_target = gram_matrix(y_target)
        loss = self.aux_loss(gram_y, gram_y_target) * self.style_weight
        return loss

class TotalVariationLoss(nn.Module):
    """computes total variation loss"""

    def __init__(self, tv_weight=1):
        super().__init__()
        self.tv_weight = tv_weight

    def forward(self, img_batch):
        batch_size = img_batch.shape[0]
        loss = (torch.sum(torch.abs(img_batch[:, :, :, :-1] - img_batch[:, :, :, 1:])) +
                torch.sum(torch.abs(img_batch[:, :, :-1, :] - img_batch[:, :, 1:, :]))) / batch_size

        loss = loss * self.tv_weight
        return loss        