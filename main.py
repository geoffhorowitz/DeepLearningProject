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

parser = argparse.ArgumentParser(description='CS7643 Assignment-2 Part 2')
parser.add_argument('--config', default='configs/config_fullmodel.yaml')


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


# def accuracy(output, target):
#     """Computes the precision@k for the specified values of k"""
#     batch_size = target.shape[0]
#
#     _, pred = torch.max(output, dim=-1)
#
#     correct = pred.eq(target).sum() * 1.0
#
#     acc = correct / batch_size
#
#     return acc


def train(epoch, data_loader, model, optimizer, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    # acc = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()
        # use index 0 if criterion is CosineSimilarity, index 1 for image class
        target = target[0]

        if torch.cuda.is_available():
            data = [data[i].cuda() for i in range(len(data))]
            target = target.cuda()

        #############################################################################
        # TODO: Complete the body of training loop                                  #
        #       1. forward data batch to the model                                  #
        #       2. Compute batch loss                                               #
        #       3. Compute gradients and update model parameters                    #
        #############################################################################
        # Referenced
        # https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#optimizing-the-model-parameters
        out_image, out_recipe = model(data)
        loss = criterion(out_image, out_recipe, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # batch_acc = accuracy(out, target)

        losses.update(loss, out_image.shape[0])
        # acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   # 'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t'
                   )
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses
                          # , top1=acc
                          ))


def validate(epoch, val_loader, model, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    # acc = AverageMeter()

    # num_class = 10
    # cm = torch.zeros(num_class, num_class)
    # evaluation loop
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()
        target = target[0]

        if torch.cuda.is_available():
            data = [data[i].cuda() for i in range(len(data))]
            target = target.cuda()
        #############################################################################
        # TODO: Complete the body of training loop                                  #
        #       HINT: torch.no_grad()                                               #
        #############################################################################
        with torch.no_grad():
            out_image, out_recipe = model(data)
            loss = criterion(out_image, out_recipe, target)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # batch_acc = accuracy(out, target)

        # update confusion matrix
        # _, preds = torch.max(out, 1)
        # for t, p in zip(target.view(-1), preds.view(-1)):
        #     cm[t.long(), p.long()] += 1

        losses.update(loss, out_image.shape[0])
        # acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t')
                  .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses
                          # , top1=acc
                          ))
    # cm = cm / cm.sum(1)
    # per_cls_acc = cm.diag().detach().numpy().tolist()
    # for i, acc_i in enumerate(per_cls_acc):
    #     print("Accuracy of Class {}: {:.4f}".format(i, acc_i))
    #
    # print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    # return acc.avg, cm
    return losses.avg


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


def im2recipe():
    # This is same setup from study
    transform_train = transforms.Compose([
        transforms.Resize(256),  # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),  # we get only the center of that rescaled
        transforms.RandomCrop(224),  # random crop within the center crop
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_loader = ImageLoader('images', transform_train, data_path=args.data_path, partition='test')
    num_images = len(image_loader)
    train_loader = torch.utils.data.DataLoader(
        image_loader, batch_size=args.batch_size, sampler=np.arange(int(0.1*num_images)))

    model = Im2Recipe(args)

    criterion = nn.CosineEmbeddingLoss(0.1)
    # found this in other impl
    # weights = torch.ones(args.num_classes)
    # weights[0] = 0
    # criterion = nn.CrossEntropyLoss(weight=weights)
    return train_loader, model, criterion


def recipe2im():
    # TODO
    return None, None, None


def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    train_loader, model, criterion = im2recipe() if args.model == 'im2recipe' else recipe2im()
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best = math.inf
    # best_cm = None
    best_model = None
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)

        # train loop
        train(epoch, train_loader, model, optimizer, criterion)

        # TODO: validation loop, change back to test_loader
        loss = validate(epoch, train_loader, model, criterion)

        if loss < best:
            best = loss
            # best_cm = cm
            best_model = copy.deepcopy(model)

    print('Best Prec @1 Loss: {:.4f}'.format(best))
    # print('Best Prec @1 Acccuracy: {:.4f}'.format(best))
    # per_cls_acc = best_cm.diag().detach().numpy().tolist()
    # for i, acc_i in enumerate(per_cls_acc):
    #     print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    if args.save_best:
        torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + '.pth')


if __name__ == '__main__':
    main()
