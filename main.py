"""
Main Function.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""
import math

import yaml
import argparse
import time
import copy

import numpy as np
import torch
import torchvision

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from data_loader import ImageLoader
from models import Im2Recipe

from utils.metrics import generate_metrics

parser = argparse.ArgumentParser(description='CS7643 Assignment-2 Part 2')
parser.add_argument('--config', default='configs/config_fullmodel.yaml')

if not(torch.cuda.device_count()):
    device = torch.device(*('cpu',0))
else:
    torch.cuda.manual_seed(1234)
    device = torch.device(*('cuda',0))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
     """Computes the precision@k for the specified values of k"""
     batch_size = target.shape[0]
     _, pred = torch.max(output, dim=-1)
     correct = pred.eq(target).sum() * 1.0
     acc = correct / batch_size
     return acc


def train(epoch, data_loader, model, optimizer, criterion, args):
    print('Training')
    iter_time = AverageMeter()
    losses = AverageMeter()
    image_acc = AverageMeter()
    recipe_acc = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()
        # use index 0 if criterion is CosineSimilarity, index 1 for image class
        data = [data[i].to(device) for i in range(len(data))]
        target = [target[i].to(device) for i in range(len(target)-2)]

        #############################################################################
        # TODO: Complete the body of training loop                                  #
        #       1. forward data batch to the model                                  #
        #       2. Compute batch loss                                               #
        #       3. Compute gradients and update model parameters                    #
        #############################################################################
        # Referenced
        # https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#optimizing-the-model-parameters
        out_image, out_recipe, out_image_reg, out_recipe_reg = model(data)
        if args.semantic_reg:
            cos_loss = criterion[0](out_image, out_recipe, target[0])
            image_loss = criterion[1](out_image_reg, target[1])
            recipe_loss = criterion[1](out_recipe_reg, target[2])
            loss = args.cos_weight * cos_loss + args.image_weight * image_loss + args.recipe_weight * recipe_loss
        else:
            loss = criterion[0](out_image, out_recipe, target[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if args.generate_metrics:
            if args.metric_type=='accuracy':
                image_batch_acc = accuracy(out_image_reg, target[1])
                image_acc.update(image_batch_acc, out_image_reg.shape[0])
                recipe_batch_acc = accuracy(out_recipe_reg, target[2])
                recipe_acc.update(recipe_batch_acc, out_recipe_reg.shape[0])
            else:
                if idx == 0:
                    img_store = out_image.data.cpu().numpy()
                    recipe_store = out_recipe.data.cpu().numpy()
                    recipe_id_store = target[-1].data.cpu().numpy()
                else:
                    img_store = np.concatenate((img_store, out_image.data.cpu().numpy()))
                    recipe_store = np.concatenate((recipe_store, out_recipe.data.cpu().numpy()))
                    recipe_id_store = np.concatenate((recipe_id_store, target[-1].data.cpu().numpy()))

        losses.update(loss, out_image.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f} avg)\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f} avg)\t'
                   'Avg Acc (image) {image.avg:.4f}\t'
                   'Avg Acc (recipe) {recipe.avg:.4f}\t'
                   )
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, image=image_acc, recipe=recipe_acc))
    metric_results = None
    if args.generate_metrics:
        if args.metric_type=='accuracy':
            metric_results = (image_acc.avg, recipe_acc.avg)
        else:
            metric_store = {}
            metric_store['image'] = img_store
            metric_store['recipe'] = recipe_store
            metric_store['recipe_id'] = recipe_id_store
            metric_results = generate_metrics(args, metric_store) # returns median, recall

    return losses.avg, metric_results


def validate(epoch, val_loader, model, criterion, args):
    print('Validation')
    iter_time = AverageMeter()
    losses = AverageMeter()
    image_acc = AverageMeter()
    recipe_acc = AverageMeter()

    # num_class = 10
    # cm = torch.zeros(num_class, num_class)
    img_store = recipe_store = recipe_id_store = None
    # evaluation loop
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()
        data = [data[i].to(device) for i in range(len(data))]
        target = [target[i].to(device) for i in range(len(target)-2)]

        #############################################################################
        # TODO: Complete the body of training loop                                  #
        #       HINT: torch.no_grad()                                               #
        #############################################################################
        with torch.no_grad():
            out_image, out_recipe, out_image_reg, out_recipe_reg = model(data)
            if args.semantic_reg:
                cos_loss = criterion[0](out_image, out_recipe, target[0])
                image_loss = criterion[1](out_image_reg, target[1])
                recipe_loss = criterion[1](out_recipe_reg, target[2])
                loss = args.cos_weight * cos_loss + args.image_weight * image_loss + args.recipe_weight * recipe_loss
            else:
                loss = criterion[0](out_image, out_recipe, target[0])
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if args.generate_metrics:
            if args.metric_type=='accuracy':
                image_batch_acc = accuracy(out_image_reg, target[1])
                image_acc.update(image_batch_acc, out_image_reg.shape[0])
                recipe_batch_acc = accuracy(out_recipe_reg, target[2])
                recipe_acc.update(recipe_batch_acc, out_recipe_reg.shape[0])
            else:
                if idx == 0:
                    img_store = out_image.data.cpu().numpy()
                    recipe_store = out_recipe.data.cpu().numpy()
                    recipe_id_store = target[-1].data.cpu().numpy()
                else:
                    img_store = np.concatenate((img_store, out_image.data.cpu().numpy()))
                    recipe_store = np.concatenate((recipe_store, out_recipe.data.cpu().numpy()))
                    recipe_id_store = np.concatenate((recipe_id_store, target[-1].data.cpu().numpy()))
        
        losses.update(loss, out_image.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Avg Acc (image) {image.avg:.4f}\t'
                   'Avg Acc (recipe) {recipe.avg:.4f}\t'
                   )
                  .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, image=image_acc, recipe=recipe_acc))
    metric_results = retrieved = None
    if args.generate_metrics:
        if args.metric_type=='accuracy':
            metric_results = (image_acc.avg, recipe_acc.avg)
        else:
            metric_store = {}
            metric_store['image'] = img_store
            metric_store['recipe'] = recipe_store
            metric_store['recipe_id'] = recipe_id_store
            metric_results = generate_metrics(args, metric_store) # returns median, recall
            retrieved = retrieval(metric_store)

    return losses.avg, metric_results, retrieved


# def adjust_learning_rate(optimizer, epoch, args):
#     epoch += 1
#     if epoch <= args.warmup:
#         lr = args.learning_rate * epoch / args.warmup
#     elif epoch > args.steps[1]:
#         lr = args.learning_rate * 0.01
#     elif epoch > args.steps[0]:
#         lr = args.learning_rate * 0.1
#     else:
#         lr = args.learning_rate
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def im2recipe(args):
    # This is same setup from study
    transform_train = transforms.Compose([
        transforms.Resize(256),  # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),  # we get only the center of that rescaled
        transforms.RandomCrop(224),  # random crop within the center crop
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256), # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224), # we get only the center of that rescaled
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_loader = ImageLoader(args.image_path, transform_train, data_path=args.data_path, partition='train',
                               mismatch=args.mismatch)
    num_images = len(image_loader)
    indexes = np.arange(num_images)
    np.random.shuffle(indexes)
    train_cutoff = int(args.train_percent * num_images)
    val_cutoff = train_cutoff + int(args.val_percent * num_images)
    train_indexes = indexes[:train_cutoff]
    image_loader.all_idx = train_indexes
    val_indexes = indexes[train_cutoff:val_cutoff]
    if torch.cuda.is_available():
        train_loader = torch.utils.data.DataLoader(
            image_loader, batch_size=args.batch_size, sampler=train_indexes,
            num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            ImageLoader(
                args.image_path,
                transform_val,
                data_path=args.data_path,
                partition='val',
                mismatch=args.mismatch,
                all_idx=val_indexes), batch_size=args.batch_size, sampler=val_indexes,
            num_workers=args.workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            image_loader, batch_size=args.batch_size, sampler=train_indexes)
        val_loader = torch.utils.data.DataLoader(
            ImageLoader(
                args.image_path,
                transform_val,
                data_path=args.data_path,
                partition='val',
                mismatch=args.mismatch,
                all_idx=val_indexes), batch_size=args.batch_size, sampler=val_indexes)

    model = Im2Recipe(args)

    cos_criterion = nn.CosineEmbeddingLoss(0.1).to(device)
    # found this in other impl
    if args.semantic_reg:
        # weights = torch.ones(args.num_classes)
        # weights[0] = 0
        # this causes nan to be thrown
        # weights_class = torch.Tensor(args.num_classes).fill_(1)
        # weights_class[0] = 0  # the background class is set to 0, i.e. ignore
        entropy_criterion = nn.CrossEntropyLoss().to(device)
    else:
        entropy_criterion = None
    return (train_loader, val_loader), model, (cos_criterion, entropy_criterion), val_indexes


def recipe2im(args):
    # TODO
    return None, None, None


def retrieval(metric_store):
    img_store = metric_store['image']
    recipe_store = metric_store['recipe']
    sims = np.matmul(img_store, recipe_store.transpose())
    retrieved_id = np.argmax(sims, axis=1)
    retrieved_val = np.max(sims, axis=1)
    return retrieved_id, retrieved_val


def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    loaders, model, criterion, val_indexes = im2recipe(args) if args.model == 'im2recipe' else recipe2im(args)
    print(torch.cuda.is_available())
    model.image_model = torch.nn.DataParallel(model.image_model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best = math.inf
    # best_cm = None
    best_model = best_retrieved = None
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)

        # train loop
        train_loss, _ = train(epoch, loaders[0], model, optimizer, criterion, args)

        val_loss, val_medR, val_retrieval = validate(epoch, loaders[1], model, criterion, args)
        if args.generate_metrics and val_medR is not None:
            val_loss = val_medR[0]

        if val_loss < best:
            best = val_loss
            best_retrieved = val_retrieval
            # best_cm = cm
            best_model = copy.deepcopy(model)

    print('Best Prec @1 Loss: {:.4f}'.format(best))
    # print('Best Prec @1 Acccuracy: {:.4f}'.format(best))
    # per_cls_acc = best_cm.diag().detach().numpy().tolist()
    # for i, acc_i in enumerate(per_cls_acc):
    #     print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    if args.save_best:
        torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + '.pth')
        image_loader = ImageLoader(args.image_path, data_path=args.data_path, partition='val', evaluate=True)
        pairs = []
        for (val_ind, ret_id, ret_val) in zip(val_indexes, best_retrieved[0], best_retrieved[1]):
            given = image_loader[val_ind]
            ret = image_loader[val_indexes[ret_id]]
            pairs.append({'given': given, 'ret': ret, 'val': ret_val})
        pairs.sort(reverse=True, key=lambda p: p['val'])
        results_file = open('image_results.txt', 'w')
        for pair in pairs:
            given = pair['given']
            ret = pair['ret']
            val = pair['val']
            # only save images for now
            results_file.write('Given image: ' + given[0] + ', Retrieved image: ' + ret[0] + ', Val: {0:.4f}'.format(val) + '\n')
        results_file.close()


if __name__ == '__main__':
    main()
