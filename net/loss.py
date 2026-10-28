import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class DSA_loss(nn.Module):
    """
    this loss function should support mse loss and infoNCE loss.
    """

    def __init__(self, loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1), device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function  # -- default CrossEntropy
        self.device = device

        self.if_infoNCE = False

    def mse_loss(self, pred, target):
        """
        Args:
            pred (Tensor): NxC input features.
            target (Tensor): NxC target features.
        """
        N = pred.size(0)
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = 1 - 1 * (pred_norm * target_norm).sum() / N
        return loss

    def forward(self, image_features1, image_features2, logit_scale):
        if self.if_infoNCE is not True:
            b, c, n = image_features1.shape

            feat1 = image_features1.transpose(2, 1).reshape(b, c*n)
            feat2 = image_features2.transpose(2, 1).reshape(b, c*n)

            loss = self.mse_loss(feat1, feat2)

        else:
            # use infoNCE as loss
            b, c, n = image_features1.shape
            feat1 = image_features1.reshape(b, -1)
            feat2 = image_features2.reshape(b, -1)
            image_features1 = F.normalize(feat1, dim=-1)
            image_features2 = F.normalize(feat2, dim=-1)

            logits_per_image1 = logit_scale * image_features1 @ image_features2.T
            logits_per_image2 = logits_per_image1.T
            labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
            loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2

        return loss


class CrossViewContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CrossViewContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feat_a, feat_b):
        feat_a = F.normalize(feat_a, dim=1)
        feat_b = F.normalize(feat_b, dim=1)

        logits = torch.matmul(feat_a, feat_b.t()) / self.temperature
        labels = torch.arange(feat_a.size(0), device=feat_a.device)

        loss_a2b = F.cross_entropy(logits, labels)
        loss_b2a = F.cross_entropy(logits.t(), labels)

        loss = (loss_a2b + loss_b2a) / 2
        return loss


class TempAnnealingCrossViewLoss(CrossViewContrastiveLoss):
    def __init__(self, initial_temperature=0.07, final_temperature=0.01, decay_rate=0.99):
        super(TempAnnealingCrossViewLoss, self).__init__(initial_temperature)
        self.final_temperature = final_temperature
        self.decay_rate = decay_rate

    def forward(self, feat_a, feat_b, epoch=None):
        if epoch is not None:
            temperature = self.temperature * (self.decay_rate ** epoch)
            temperature = max(temperature, self.final_temperature)
        else:
            temperature = self.temperature

        feat_a = F.normalize(feat_a, dim=1)
        feat_b = F.normalize(feat_b, dim=1)

        logits = torch.matmul(feat_a, feat_b.t()) / temperature
        labels = torch.arange(feat_a.size(0), device=feat_a.device)

        loss_a2b = F.cross_entropy(logits, labels)
        loss_b2a = F.cross_entropy(logits.t(), labels)

        loss = (loss_a2b + loss_b2a) / 2
        return loss


def loss_con(x1, x2, epoch=None):
    criterion = TempAnnealingCrossViewLoss()
    loss = criterion(x1, x2, epoch)
    return loss


def loss_ce(outputs, labels):
    loss_func = nn.CrossEntropyLoss()
    loss = 0
    if isinstance(outputs, list):
        for i in outputs:
            loss += loss_func(i, labels)
        loss = loss / len(outputs)
    else:
        loss = loss_func(outputs, labels)
    return loss
