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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data_loader import ImageLoader
from models import Im2Recipe

from utils.metrics import generate_metrics

parser = argparse.ArgumentParser(description='Alphabet Soup Final Project')
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
        recipe_id = target[-1]
        data = [data[i].to(device) for i in range(len(data))]
        target = [target[i].to(device) for i in range(len(target)-2)]

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

        if args.generate_metrics:
            if args.metric_type=='accuracy' or args.metric_type=='both':
                image_batch_acc = accuracy(out_image_reg, target[1])
                image_acc.update(image_batch_acc, out_image_reg.shape[0])
                recipe_batch_acc = accuracy(out_recipe_reg, target[2])
                recipe_acc.update(recipe_batch_acc, out_recipe_reg.shape[0])
            if args.metric_type=='rank' or args.metric_type=='both':
                if idx == 0:
                    img_store = out_image.data.cpu().numpy()
                    recipe_store = out_recipe.data.cpu().numpy()
                    # recipe_id_store = target[-1].data.cpu().numpy()
                    recipe_id_store = recipe_id
                else:
                    img_store = np.concatenate((img_store, out_image.data.cpu().numpy()))
                    recipe_store = np.concatenate((recipe_store, out_recipe.data.cpu().numpy()))
                    # recipe_id_store = np.concatenate((recipe_id_store, target[-1].data.cpu().numpy()))
                    recipe_id_store = np.concatenate((recipe_id_store, recipe_id), axis=0)

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
    metric_results = [(image_acc.avg, recipe_acc.avg)] # at worst, this will just be 0
    if args.generate_metrics:
        if args.metric_type=='rank' or args.metric_type=='both':
            metric_store = {}
            metric_store['image'] = img_store
            metric_store['recipe'] = recipe_store
            metric_store['recipe_id'] = recipe_id_store
            metric_results.append(generate_metrics(args, metric_store)) # returns (median, recall)
        else:
            metric_results.append((1000, 1000))

    return losses.avg, metric_results


def validate(epoch, val_loader, model, criterion, args):
    print('Validation')
    iter_time = AverageMeter()
    losses = AverageMeter()
    image_acc = AverageMeter()
    recipe_acc = AverageMeter()

    img_store = recipe_store = recipe_id_store = None
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()
        recipe_id = target[-1]
        data = [data[i].to(device) for i in range(len(data))]
        target = [target[i].to(device) for i in range(len(target)-2)]

        with torch.no_grad():
            out_image, out_recipe, out_image_reg, out_recipe_reg = model(data)
            if args.semantic_reg:
                cos_loss = criterion[0](out_image, out_recipe, target[0])
                image_loss = criterion[1](out_image_reg, target[1])
                recipe_loss = criterion[1](out_recipe_reg, target[2])
                loss = args.cos_weight * cos_loss + args.image_weight * image_loss + args.recipe_weight * recipe_loss
            else:
                loss = criterion[0](out_image, out_recipe, target[0])

        if args.generate_metrics:
            if args.metric_type=='accuracy' or args.metric_type=='both':
                image_batch_acc = accuracy(out_image_reg, target[1])
                image_acc.update(image_batch_acc, out_image_reg.shape[0])
                recipe_batch_acc = accuracy(out_recipe_reg, target[2])
                recipe_acc.update(recipe_batch_acc, out_recipe_reg.shape[0])
            if args.metric_type=='rank' or args.metric_type=='both':
                if idx == 0:
                    img_store = out_image.data.cpu().numpy()
                    recipe_store = out_recipe.data.cpu().numpy()
                    # recipe_id_store = target[-1].data.cpu().numpy()
                    recipe_id_store = recipe_id
                else:
                    img_store = np.concatenate((img_store, out_image.data.cpu().numpy()))
                    recipe_store = np.concatenate((recipe_store, out_recipe.data.cpu().numpy()))
                    # recipe_id_store = np.concatenate((recipe_id_store, target[-1].data.cpu().numpy()))
                    recipe_id_store = np.concatenate((recipe_id_store, recipe_id), axis=0)
        
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
    retrieved = None
    metric_results = [(image_acc.avg, recipe_acc.avg)] # at worst, this will just be 0
    if args.generate_metrics:
        if args.metric_type=='rank' or args.metric_type=='both':
            metric_store = {}
            metric_store['image'] = img_store
            metric_store['recipe'] = recipe_store
            metric_store['recipe_id'] = recipe_id_store
            retrieved = retrieval(metric_store)
            metric_results.append(generate_metrics(args, metric_store)) # returns (median, recall)
        else:
            metric_results.append((1000, 1000))

    return losses.avg, metric_results, retrieved


def adjust_learning_rate(optimizer, epoch, args):
    try:
        steps = args.steps
    except RuntimeError:
        return
    epoch += 1
    if epoch > steps[1]:
        lr = args.learning_rate * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch > steps[0]:
        lr = args.learning_rate * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def im2recipe(args):
    # Using same results transform setup from paper to reproduce results.
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
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
    entropy_criterion = None
    if args.semantic_reg:
        entropy_criterion = nn.CrossEntropyLoss().to(device)
    return (train_loader, val_loader), model, (cos_criterion, entropy_criterion), val_indexes


def retrieval(metric_store):
    img_store = metric_store['image']
    recipe_store = metric_store['recipe']
    recipe_id_store = metric_store['recipe_id']
    sims = np.matmul(img_store, recipe_store.transpose())
    retrieved_id = recipe_id_store[np.argmax(sims, axis=1)]
    retrieved_val = np.max(sims, axis=1)
    return (recipe_id_store, retrieved_id), retrieved_val


def main(args, tuning_model=False):
    loaders, model, criterion, val_indexes = im2recipe(args)
    print(torch.cuda.is_available())
    model.image_model = torch.nn.DataParallel(model.image_model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best = math.inf
    best_model = best_retrieved = None
    if not tuning_model: 
        results_dict = {'baseline':{'baseline': {0: {}}}}
        results_dict = {'defaults': args}
    
    train_loss_history = np.zeros(args.epochs)
    val_loss_history = np.zeros(args.epochs)
    train_median_history = np.zeros(args.epochs)
    val_median_history = np.zeros(args.epochs)
    train_imacc_history = np.zeros(args.epochs)
    val_imacc_history = np.zeros(args.epochs)
    train_recacc_history = np.zeros(args.epochs)
    val_recacc_history = np.zeros(args.epochs)
    
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train_loss, train_metrics = train(epoch, loaders[0], model, optimizer, criterion, args)

        val_loss, val_metrics, val_retrieval = validate(epoch, loaders[1], model, criterion, args)
        
        (train_acc_image, train_acc_recipe), (train_median, train_recall) = train_metrics
        (val_acc_image, val_acc_recipe), (val_median, val_recall) = val_metrics

        train_loss_history[epoch] = avg_train_loss.item()
        val_loss_history[epoch] = avg_val_loss.item()

        train_median_history[epoch] = train_median
        val_median_history[epoch] = val_median

        train_imacc_history[epoch] = train_acc_image.item()
        val_imacc_history[epoch] = val_acc_image.item()
        train_recacc_history[epoch] = train_acc_recipe.item()
        val_recacc_history[epoch] = val_acc_recipe.item()
        
        
        if args.generate_metrics:
            val_acc, val_medR = val_metrics
            val_loss = val_median

        if args.save_best and val_loss < best:
            best = val_loss
            best_retrieved = val_retrieval
            best_model = copy.deepcopy(model)

        if not tuning_model:
            results_dict['baseline']['baseline'][0] = {}
            results_dict['baseline']['baseline'][0]['id'] = '{}.{}.{}'.format(1, 1, 1)
            results_dict['baseline']['baseline'][0]['train_loss'] = train_loss_history
            results_dict['baseline']['baseline'][0]['train_median'] = train_median_history
            results_dict['baseline']['baseline'][0]['train_imacc'] = train_imacc_history
            results_dict['baseline']['baseline'][0]['train_recacc'] = train_recacc_history
            results_dict['baseline']['baseline'][0]['val_loss'] = val_loss_history
            results_dict['baseline']['baseline'][0]['val_median'] = val_median_history
            results_dict['baseline']['baseline'][0]['val_imacc'] = val_imacc_history
            results_dict['baseline']['baseline'][0]['val_recacc'] = val_recacc_history
            
        
    print('Best Prec @1 Loss: {:.4f}'.format(best))

    if args.save_best:
        torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + '.pth')
        pairs = []
        for (given, ret, ret_val) in zip(best_retrieved[0][0], best_retrieved[0][1], best_retrieved[1]):
            pairs.append({'given': given, 'ret': ret, 'val': ret_val})
        pairs.sort(reverse=True, key=lambda p: p['val'])
        results_file = open('queried_results.txt', 'w')
        for pair in pairs:
            given = pair['given']
            ret = pair['ret']
            val = pair['val']
            # only save id to identify from layer1/2.json later.
            results_file.write('Given Id: ' + given + ', Retrieved Id: ' + ret + ', Val: {0:.4f}'.format(val) + '\n')
        results_file.close()

    if tuning_model:
        return train_loss_history, val_loss_history, train_median_history, val_median_history, train_imacc_history, val_imacc_history, train_recacc_history, val_recacc_history
    else:
        results_dict['baseline']['baseline']['best_median'] = best_median
        results_dict['baseline']['baseline']['best_run'] = best_run
        results_dict['baseline']['baseline']['train_loss_mean'] = train_loss_history
        results_dict['baseline']['baseline']['train_loss_std'] = np.zeros(train_loss_history.shape)
        results_dict['baseline']['baseline']['train_median_mean'] = train_median_history
        results_dict['baseline']['baseline']['train_median_std'] = np.zeros(train_median_history.shape)
        results_dict['baseline']['baseline']['train_imacc_mean'] = train_imacc_history
        results_dict['baseline']['baseline']['train_imacc_std'] = np.zeros(train_imacc_history.shape)
        results_dict['baseline']['baseline']['train_recacc_mean'] = train_recacc_history
        results_dict['baseline']['baseline']['train_recacc_std'] = np.zeros(train_recacc_history.shape)
        results_dict['baseline']['baseline']['val_loss_mean'] = val_loss_history
        results_dict['baseline']['baseline']['val_loss_std'] = np.zeros(val_loss_history.shape)
        results_dict['baseline']['baseline']['val_median_mean'] = val_median_history
        results_dict['baseline']['baseline']['val_median_std'] = np.zeros(val_median_history.shape)
        results_dict['baseline']['baseline']['val_imacc_mean'] = val_imacc_history
        results_dict['baseline']['baseline']['val_imacc_std'] = np.zeros(val_imacc_history.shape)
        results_dict['baseline']['baseline']['val_recacc_mean'] = val_recacc_history
        results_dict['baseline']['baseline']['val_recacc_std'] = np.zeros(val_recacc_history.shape)
        
        f=open('main_results.pkl', 'wb')
        pickle.dump(results_dict, f)
        f.close()
        
        plot_complex_learning_curve(results_dict)
    

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
            
    main(args)
