import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)


        # One-hot encode targets
        num_classes = logits.shape[1]
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Compute Dice coefficient
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # Average Dice Loss over classes
        dice_loss = 1 - dice.mean()
        return dice_loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, weights = None):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index = 255, weight = weights)
        self.dice_loss = DiceLoss()


    def forward(self, logits, targets):
        targets = torch.where(targets == 255, torch.tensor(0, dtype=torch.long, device=targets.device), targets)
        ce_loss = self.cross_entropy_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        # print(f"ce: {ce_loss}, dice: {dice_loss}")
        return self.alpha * ce_loss + self.beta * dice_loss
