import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append('/home/cuonghoang/Desktop/codedict/publised-code/data')

from augmentations import get_transform
from get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import Head, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups,loss_att

@torch.no_grad()
def _momentum_update_key_encoder(model,momen_model):
    """
    Momentum update of the key encoder
    """
    for param_q, param_k in zip(
       model.parameters(), momen_model.parameters()
    ):
        param_k.data = param_k.data * 0.9 + param_q.data * (1.0 -  0.9)
    return momen_model

queue_ptr=0
queue_ptr1=0
queue_ptr12=0
queue_ptr13=0
queue_size=1600 #1600

@torch.no_grad()
def dequeue_and_enqueue( queue,keys):
    # gather keys before updating queue
    global queue_ptr 
   
    batch_size = keys.shape[0]
    ptr = int(queue_ptr)
    # replace the keys at ptr (dequeue and enqueue)
    queue[ ptr : ptr + batch_size,:] = keys
    ptr = (ptr + batch_size) % queue_size # move pointer

    queue_ptr= ptr
    return queue

@torch.no_grad()
def dequeue_and_enqueue1( queue,keys):
    # gather keys before updating queue
    global queue_ptr1 
   
    batch_size = keys.shape[0]
    ptr = int(queue_ptr1)
    # replace the keys at ptr (dequeue and enqueue)
    queue[ ptr : ptr + batch_size,:] = keys
    ptr = (ptr + batch_size) % queue_size # move pointer

    queue_ptr1= ptr
    return queue
 
@torch.no_grad()
def dequeue_and_enqueue12( queue,keys):
    # gather keys before updating queue
    global queue_ptr12 
   
    batch_size = keys.shape[0]
    ptr = int(queue_ptr12)
    # replace the keys at ptr (dequeue and enqueue)
    queue[ ptr : ptr + batch_size] = keys
    ptr = (ptr + batch_size) % queue_size # move pointer

    queue_ptr12= ptr
    return queue


@torch.no_grad()
def dequeue_and_enqueue13( queue,keys):
    # gather keys before updating queue
    global queue_ptr13 
   
    batch_size = keys.shape[0]
    ptr = int(queue_ptr13)
    # replace the keys at ptr (dequeue and enqueue)
    queue[ ptr : ptr + batch_size] = keys
    ptr = (ptr + batch_size) % queue_size # move pointer

    queue_ptr13= ptr
    return queue



def train(student,momentum_model, train_loader, unlabelled_train_loader, args,dataset):

    list_score=np.zeros(len(dataset))
    l_normalized=0.07*np.ones(len(dataset))
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )


    cluster_criterion = DistillLoss()
    
    att_loss=loss_att()
    
    queue = torch.zeros(queue_size,256,requires_grad=False).to(device)
    queue2 = torch.zeros(queue_size,1200,requires_grad=False).to(device)
    queue12 = torch.zeros(queue_size,requires_grad=False).to(device)
    queue13 = torch.zeros(queue_size,requires_grad=False).to(device).bool()
    dem=0
    print('---init queue---')
    for batch_idx, batch in enumerate(train_loader):
        images =batch['augmented'].cuda(non_blocking=True)
      
        student_proj, student_out,haha= momentum_model(images)
        # print('student_proj',student_proj.shape)
        # print('student_out',student_out.shape)
        # print('haha',haha.shape)
        # exit()
        dem=dem+1
        # print('student_proj',student_proj.shape)
        # print('student_out',student_out.shape)
        # exit()
        dequeue_and_enqueue(queue,student_proj)
        dequeue_and_enqueue1(queue2,student_out)
        dequeue_and_enqueue12(queue12,batch['meta']['label'])
        dequeue_and_enqueue13(queue13,batch['meta']['mask'].bool())
        # print('queue_ptr',queue_ptr)
        if dem==queue_size/images.shape[0]+1:
            break
        # print('queue',queue)

    for epoch in range(args.epochs):
        # print('epoch',epoch)
        loss_record = AverageMeter()

        student.train()
        # print('list_score',list_score.shape,list_score)
        print('---assigning temp----')
        if epoch>0:
            a = 0.07
            b = 1
            # Min-Max normalization to range [a, b]
           
            # Apply Min-Max normalization to [a, b] range
            hlow = np.percentile(list_score, 10)  # 10th percentile
            hhigh = np.percentile(list_score, 90)  # 90th percentile

           

            # Clip the values in the array so they fall within [hlow, hhigh]
            l_clipped = np.clip(list_score, hlow, hhigh)

            l_min = l_clipped.min()  # Get the minimum value in the array
            l_max = l_clipped.max()  # Get the maximum value in the array

            l_normalized = a + ((l_clipped - l_min) * (b - a)) / (l_max - l_min)
            # print('unique_temp',np.unique(l_normalized))
            # print('---assigning temp----')
            # for batch_idx, batch in enumerate(tqdm(train_loader)):
                # batch['meta']['temp']=l_normalized[batch['meta']['index']]
         
        print('---Training----')
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            # print('batch',batch)
            # exit()
            images, class_labels, mask_lab, bin_mask =batch['image'], batch['meta']['label'],batch['meta']['mask'],batch['semseg']

            # print('queue_ptr12',queue_ptr1)
            # mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = images.cuda(non_blocking=True)
            bin_mask=bin_mask.cuda(non_blocking=True)

            tempt=torch.tensor(l_normalized[batch['meta']['index']],requires_grad=False).cuda(non_blocking=True)
            score_ori=batch['meta']['score'].cuda(non_blocking=True).clone()

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                # print('images',images.shape)
                # exit()
                student_proj, student_out,att_map = student(images)
                student_proj1, student_out1,att_map1 = momentum_model(images)
                
                dequeue_and_enqueue1(queue2,student_out1)
                

                score=torch.matmul(student_proj,queue.T)
                topk_values, topk_indices = torch.topk(score, k=int(queue_size/100), dim=1)
                # print('topk_values',topk_values.shape,score.shape)
                score1=torch.sum(torch.exp(topk_values),dim=1)/torch.sum(torch.exp(score),dim=1)

                # print('hahaha',batch['meta']['score'])
                phi=0.97
                batch['meta']['score']=phi*score_ori+(1-phi)*score1

                list_score[batch['meta']['index']]=batch['meta']['score'].detach().cpu()
                # print(list_score[batch['meta']['index']])
                # print(batch['meta']['score'])

                sup_logits = student_out[mask_lab] 
                sup_labels = class_labels[mask_lab] 

                # print('class_labels',class_labels)_
                # print('mask_lab',mask_lab)

                # print('sup_logits',sup_logits.shape)
                # print('sup_labels',sup_labels)
                # exit()
                atten_loss=att_loss(att_map,bin_mask)

                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                # print('cls_loss',cls_loss)
                # exit()
                # clustering, unsup
                # print('queue_ptr13',queue_ptr1)
                cluster_loss = cluster_criterion(student_out1, queue2,queue_ptr1)
                # avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                # me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                # cluster_loss += args.memax_weight * me_max_loss
                # print('cluster_loss',cluster_loss)
                # exit()
                # represent learning, unsup
                # contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                # contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
                # if epoch>0:
               
                contrastive_loss=info_nce_logits(student_proj,student_proj1,queue.clone(),temperature=tempt)
                # print('contrastive_loss',contrastive_loss)

                # exit()
                # representation learning, sup
                student_proj =student_proj[mask_lab]
                # student_proj = torch.nn.functional.normalize(student_proj, dim=-1)

                sup_con_labels = class_labels[mask_lab]
                
                queue_label=queue12[queue13]


                # print('sup_con_labels',sup_con_labels)
                # print('queue_label',queue_label)
                # exit()

                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels,queue=queue[queue13].clone(),queue_label=queue_label.clone(),temperature=tempt[mask_lab])

                # exit()
                dequeue_and_enqueue(queue,student_proj1)
                dequeue_and_enqueue12(queue12,batch['meta']['label'])
                dequeue_and_enqueue13(queue13,batch['meta']['mask'].cuda(non_blocking=True).bool())
              
                loss = 0
                loss+=atten_loss
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss +=  (1 - args.sup_weight) * contrastive_loss +args.sup_weight * sup_con_loss 

                print(epoch,f"{atten_loss.item():.2f}, {cluster_loss.item():.2f}, {cls_loss.item():.2f}, {contrastive_loss.item():.2f}, {sup_con_loss.item():.2f}")

            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

        
            # # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
         

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

       

        momentum_model=_momentum_update_key_encoder(student,momentum_model)

    torch.save(student.state_dict(), 'model_weights.pth')


from get_datasets import Obdata, get_train_transformations, get_train_dataloader 




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=8, type=int)  #128
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    
    # if args.warmup_model_dir is not None:
    #     args.logger.info(f'Loading weights from {args.warmup_model_dir}')
    #     backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 2048
    args.num_mlp_layers = 3
    args.mlp_out_dim = 1200 #args.num_labeled_classes + args.num_unlabeled_classes

 
    
    args.logger.info('model build')

  
    train_transforms = get_train_transformations()
    dataset = Obdata(root='haha', split='val', transform=train_transforms)
    # print('dataset',len(dataset))
    # print(dataset[0])
    # exit()
    train_loader = get_train_dataloader( dataset,args.batch_size) 




    from resnet import ResNet50

    model = ResNet50(num_classes=1,feat_dim=args.feat_dim, mlp_out_dim=args.mlp_out_dim, num_mlp_layers=args.num_mlp_layers).to(device)

    momentum_model = ResNet50(num_classes=1,feat_dim=args.feat_dim, mlp_out_dim=args.mlp_out_dim, num_mlp_layers=args.num_mlp_layers).to(device)
  
    for param in momentum_model.parameters():
        param.requires_grad = False
    # ----------------------
    # TRAIN
    # ----------------------
    # train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    train(model,momentum_model, train_loader, None, args,dataset)
