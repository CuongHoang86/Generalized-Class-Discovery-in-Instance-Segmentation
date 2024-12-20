import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Head(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x_proj = nn.functional.normalize(x_proj, dim=-1, p=2)

        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        # print('logits',logits.shape)
        # exit()
        return x_proj, logits


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        # self.temperature = temperature
        self.contrast_mode = contrast_mode
        # self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None,queue=None,queue_label=None,temperature=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # if len(features.shape) < 3:
        #     raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                      'at least 3 dimensions are required')
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, queue_label.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = features
     

        expanded_temp = temperature.unsqueeze(1)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, queue.T),
            expanded_temp)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

       
        exp_logits = torch.exp(logits) * mask
        # print('hahahaha',exp_logits.sum(1, keepdim=True))
        # exit()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+0.1)

        # print('log_prob',log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # print('mean_log_prob_pos',mean_log_prob_pos)
        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = - mean_log_prob_pos
        # print('loss',loss.shape,anchor_count,batch_size)
        # exit()
        loss = loss.mean()

        # print('loss',loss)
        # exit()

        return loss



def info_nce_logits(features, features1,queue, temperature=1.0, device='cuda'):

    # print('temperature',temperature.shape)
    # features = F.normalize(features, dim=-1, p=2)
    # features1 = F.normalize(features1, dim=-1, p=2)
    
    # Compute similarity between query and key (positive pair)
    pos = torch.bmm(features.view(features.size(0), 1, -1),
                        features1.view(features1.size(0), -1, 1)).squeeze(-1)
    
    # Compute similarity between query and all keys in the queue (negative samples)
    neg = torch.mm(features, queue.transpose(1, 0))
    
    # Combine positive and negative similarities
    # print('positive_sim',pos.shape)
    # print('negative_sim',neg.shape)

    logits = torch.cat((pos, neg), dim=1)

    # print('logits',logits.shape)
    expanded_temp = temperature.unsqueeze(1)

    logits /= expanded_temp

    # print('logits',logits.shape)
    # Labels: the positive pair is the first class (index 0)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    
    # print('labels',labels)
    # Compute the loss using softmax
    loss = F.cross_entropy(logits, labels)
    # print('loss',loss)
    # Update the queue with new momentum features
    # exit()
    return loss

import torchvision
import torchvision.transforms as T


class loss_att(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self):
        super(loss_att, self).__init__()
        # self.temperature = temperature
        # self.intra=SPKDLoss()
        self.tau_plus=0.1
        dimen=112
        self.rezsal1=torchvision.transforms.Resize(dimen, interpolation=T.InterpolationMode.NEAREST)
        self.rezsal2=torchvision.transforms.Resize(int(dimen/2), interpolation=T.InterpolationMode.NEAREST)
        self.rezsal3=torchvision.transforms.Resize(int(dimen/4), interpolation=T.InterpolationMode.NEAREST)
        self.rezsal4=torchvision.transforms.Resize(int(dimen/8), interpolation=T.InterpolationMode.NEAREST)

    def weight_map(self,mask):
        e1,e2,e3=mask.shape

        weights = torch.tensor([[-1., -1., -1.],
                        [-1., 8., -1.],
                        [-1., -1., -1]])
        weights = weights.view(1,1,3, 3).cuda()

        # print('weights',weights.shape)
        mask=mask.unsqueeze(1)
        # print('mask',mask.shape)
        map = F.conv2d(mask, weights)
        mul=1
        dimen=112
        if e2==dimen:
            mul=0.25
        if e2==dimen/2:
            mul=0.5
        if e2==dimen/4:
            mul=0.75
        if e2==dimen/8:
            mul=1

        map=torch.where(map!=0, mul, 1)

        map=F.pad(map,pad=(1,1,1,1),value=1)

        return map

    def mask_loss(self,out,mask,map):
        # print('out',out.shape)
        map1=map.squeeze(1)
        a=(out-mask)**2
        # print(a)
        # print('map',map1.shape)
        a=a*map1

        loss=torch.mean(a)
        # print(a)
        # sys.exit()
        return loss
    def forward(self, haha,mask32):

        mask_atten1=self.rezsal1(mask32).cuda().float()
        mask_atten2=self.rezsal2(mask32).cuda().float()
        mask_atten3=self.rezsal3(mask32).cuda().float()
        mask_atten4=self.rezsal4(mask32).cuda().float()

        map1=self.weight_map(mask_atten1)
        map2=self.weight_map(mask_atten2)
        map3=self.weight_map(mask_atten3)
        map4=self.weight_map(mask_atten4)
        # print('koi',torch.unique(mask))
        # print('haha',haha)
        # sys.exit()

        mask_loss1=self.mask_loss(haha[0],mask_atten1,map1)
        mask_loss2=self.mask_loss(haha[1],mask_atten2,map2)
        mask_loss3=self.mask_loss(haha[2],mask_atten3,map3)
        mask_loss4=self.mask_loss(haha[3],mask_atten4,map4)

        finalloss=(mask_loss1+mask_loss2+mask_loss3+mask_loss4)/4
        # print('finalloss',finalloss)
        # exit()
        return finalloss




def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
    return (weight.t() / torch.sum(weight, 1)).t()

class DistillLoss(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.cluster_loss = nn.KLDivLoss(size_average=False)

    def forward(self, student_output, teacher_output, pointer):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # print('student_output',student_output.shape)
       

        student_output = F.softmax(student_output / 0.1, dim=-1)
        teacher_out = F.softmax(teacher_output / 0.1, dim=-1)

        batch,_=student_output.shape
        target = target_distribution(teacher_out).detach()

        # print('teacher_output',student_output.shape)
        # print('pointer',pointer)
        if pointer!=0:
            cluster_loss = self.cluster_loss((student_output).log(), target[pointer-batch:pointer,:])/student_output.shape[0]
       
        if pointer==0:
            pt=target.shape[0]

            cluster_loss = self.cluster_loss((student_output).log(), target[pt-batch:pt,:])/student_output.shape[0]
        
     
        return cluster_loss
