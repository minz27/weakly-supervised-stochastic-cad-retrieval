import torch
import torch.nn as nn
from pytorch_metric_learning import losses

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        logits = torch.div(
            torch.matmul(
                embeddings, torch.transpose(embeddings, 0, 1)
            ), self.temperature
        )    
        return losses.NTXentLoss(self.temperature)(logits, torch.squeeze(labels))
