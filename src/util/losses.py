import torch
import torch.nn as nn
import torchvision
from pytorch_metric_learning import losses
import numpy as np

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

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0], style_layers=[0,1,2,3]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std    
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        content_loss = 0.0
        style_loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                content_loss_1 = torch.nn.functional.l1_loss(x, y)
                x_content = x.reshape(1,-1).detach().cpu()
                y_content = y.reshape(-1,1).detach().cpu()
                
                dot_product = x_content@y_content
                norm_x = np.linalg.norm(x_content)
                norm_y = np.linalg.norm(y_content)

                content_loss_2 = (np.pi - np.arccos(dot_product/(norm_x*norm_y)))[0][0]

                content_loss += (content_loss_1 + content_loss_2)/2

            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                style_loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        
        # return content_loss[0,0], style_loss
        return content_loss, style_loss

class TripletMarginLossWithNegativeMining(nn.Module):
    def __init__(self, margin=1.0, negative_ratio=0.5):
        super(TripletMarginLossWithNegativeMining, self).__init__()
        self.margin = margin
        self.negative_ratio = negative_ratio
        
    def forward(self, anchor, positive, negative):
        # Compute pairwise distances between anchor, positive, and negative embeddings
        pos_dist = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        neg_dist = torch.sum(torch.pow(anchor.unsqueeze(1) - negative.unsqueeze(0), 2), dim=2)
        
        # Compute the loss for each triplet
        loss = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        
        # Select hard negative examples for each anchor-positive pair
        num_hard_negatives = int(self.negative_ratio * loss.size(0))
        hard_negatives = []
        for i in range(loss.size(0)):
            # Select the k hardest negatives for the current anchor-positive pair
            _, hard_neg_indices = torch.topk(neg_dist[i], k=num_hard_negatives, largest=False)
            hard_negatives.append(negative[hard_neg_indices])
        hard_negatives = torch.cat(hard_negatives, dim=0)
        
        # Compute the loss with the hard negative examples
        hard_neg_dist = torch.sum(torch.pow(anchor.unsqueeze(1) - hard_negatives.unsqueeze(0), 2), dim=2)
        hard_neg_loss = torch.clamp(self.margin + pos_dist.unsqueeze(1) - hard_neg_dist, min=0.0)
        loss = torch.mean(loss + torch.mean(hard_neg_loss, dim=1))
        
        return loss