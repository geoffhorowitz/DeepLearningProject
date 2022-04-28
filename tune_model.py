import math
import os, sys
import random
import time
import copy
import yaml
import argparse
import pickle

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
from main import im2recipe, train, validate

from collections import namedtuple
from plot_methods import plot_simple_learning_curve, plot_complex_learning_curve, plot_complexity_curve


def main(model_inputs=None):
    default_dict = {
        #Train:
        'batch_size': 100,
        'learning_rate': 0.001,
        'reg': 0.0005,
        'epochs': 10,
        'embed_dim': 1024,
        'num_classes': 1048,
        'train_percent': 0.1,
        'val_percent': 0.05,
        'semantic_reg': True,
        'cos_weight': 0.8,
        'image_weight': 0.1,
        'recipe_weight': 0.1,
        'workers': 4,
        'mismatch': 0.5,
        #network:
        'model': 'im2recipe',
        #data:
        'data_path': 'data',
        'image_path': 'data/images',
        'generate_metrics': True,
        'save_best': True,
        #ingredient_lstm:
        'ingredient_lstm_dim': 300,
        'ingredient_embedding_dim': 300, # vocab size 30167 x 300 embedded
        'ingredient_w2v_path': 'data/vocab.bin',
        #recipe_lstm:
        'recipe_lstm_dim': 1024,
        'recipe_embedding_dim': 1024,
        'dropout': .2
    }

    # LSTM
    experiment_dict = {
        #'encoder_emb_size': [32, 64, 128], # 64
        #'encoder_hidden_size': [32, 64, 128], # 128
        #'encoder_dropout': [.2, .5, .8], # .2
        #'encoder_dropout': [0., .1, .2], # .2
        #'decoder_emb_size': [32, 64, 128], # 32 - no signf
        #'decoder_dropout': [.2, .5, .8], # .2
        #'decoder_dropout': [0., .1, .2], # .2
        #'learning_rate': [1e-2, 1e-3, 1e-4], # 1e-3
        #'learning_rate': [5e-1, 5e-2, 1e-2, 1e-3], # 1e-2
        #'epochs': [5, 10, 20]
        'mismatch': [.8, .5, .2]
    }


    runs_per_experiment = 1 # balance b/w trials and hyperparams to test

    results_dict = {'defaults': default_dict}

    for i, (key, val) in enumerate(experiment_dict.items()):
        results_dict[key] = {}
        for j, (exp_val) in enumerate(val):
            results_dict[key][exp_val] = {}
            best_median = 1e10
            n_epochs = default_dict['epochs'] if 'epochs' != key else exp_val
            train_loss_all = np.zeros((runs_per_experiment, n_epochs))
            #train_perp_all = np.zeros((runs_per_experiment, n_epochs))
            train_median_all = np.zeros((runs_per_experiment, n_epochs))
            val_loss_all = np.zeros((runs_per_experiment, n_epochs))
            #val_perp_all = np.zeros((runs_per_experiment, n_epochs))
            val_median_all = np.zeros((runs_per_experiment, n_epochs))

            fh = open('experiments/{}_{}_results.txt'.format(key, exp_val), 'w')

            for k in range(runs_per_experiment):
                input_dict = default_dict.copy()
                input_dict[key] = exp_val

                args = namedtuple("args", input_dict.keys())(*input_dict.values()) # to get it in the same dot callable format

                train_loss, val_loss, train_median, val_median = run(args, model_inputs)

                # save pertinent results for later plotting
                if val_median[-1] < best_median:
                    best_median = val_median[-1]
                    best_run = k

                results_dict[key][exp_val][k] = {}
                results_dict[key][exp_val][k]['id'] = '{}.{}.{}'.format(i+1, j+1, k+1)
                results_dict[key][exp_val][k]['train_loss'] = train_loss
                results_dict[key][exp_val][k]['train_median'] = train_median
                results_dict[key][exp_val][k]['val_loss'] = val_loss
                results_dict[key][exp_val][k]['val_median'] = val_median

                train_loss_all[k, :] = train_loss
                train_median_all[k, :] = train_median
                val_loss_all[k, :] = val_loss
                val_median_all[k, :] = val_median

                # text results
                results_text = '\nexperiment {}={} run {}:'.format(key, exp_val, k) + \
                                '\n\ttrain loss: {}'.format(np.round(train_loss[-1], 4)) + \
                                '\n\ttrain median: {}'.format(np.round(train_median[-1], 4)) + \
                                '\n\tvalid loss: {}'.format(np.round(val_loss[-1], 4)) + \
                                '\n\tvalid median: {}'.format(np.round(val_median[-1], 4))

                print(results_text)

                fh.write(results_text)

                # plots
                #plot_simple_learning_curve(train_loss, train_perp, val_loss, val_perp, (key, exp_val, i, j, k))

            # compute stats
            results_dict[key][exp_val]['best_median'] = best_median
            results_dict[key][exp_val]['best_run'] = best_run
            results_dict[key][exp_val]['train_loss_mean'] = np.mean(train_loss_all, axis=0)
            results_dict[key][exp_val]['train_loss_std'] = np.std(train_loss_all, axis=0)
            results_dict[key][exp_val]['train_median_mean'] = np.mean(train_median_all, axis=0)
            results_dict[key][exp_val]['train_median_std'] = np.std(train_median_all, axis=0)
            results_dict[key][exp_val]['val_loss_mean'] = np.mean(val_loss_all, axis=0)
            results_dict[key][exp_val]['val_loss_std'] = np.std(val_loss_all, axis=0)
            results_dict[key][exp_val]['val_median_mean'] = np.mean(val_median_all, axis=0)
            results_dict[key][exp_val]['val_median_std'] = np.std(val_median_all, axis=0)

            # close file handle
            fh.close()

    # save results dictionary for potential later use
    #with open('experiments/results_dict.json', 'wb') as f:
    #    json.dump(results_dict, f)
    f=open('experiments/results_dict.pkl', 'wb')
    pickle.dump(results_dict, f)
    f.close()

    # run final results
    plot_complex_learning_curve(results_dict, logx_scale=False)
    plot_complexity_curve(results_dict, logx_scale=True)


def run_pickle_data(pickle_file):
    f = open(pickle_file, "rb")
    results_dict = pickle.load(f)
    f.close()

    plot_complex_learning_curve(results_dict, logx_scale=False)
    #plot_complexity_curve(results_dict, logx_scale=True)


def generate_metrics(args, metric_store):
    # metric analysis ref: https://github.com/torralba-lab/im2recipe-Pytorch/blob/master/scripts/rank.py
    # taken from paper to ensure comparability of results
    '''
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    f = open(metric_store, "rb")
    metric_store = pickle.load(f)
    f.close()
    '''
    #print(metric_store.keys())
    im_vecs = metric_store['image']
    instr_vecs = metric_store['recipe']
    names = metric_store['recipe_id']

    random.seed(1234)
    type_embedding = 'images' if args.model == 'im2recipe' else 'recipe'

    # Sort based on names to always pick same samples for medr
    idxs = np.argsort(names)
    names = names[idxs]

    # Ranker
    N = 1000
    idxs = range(N)

    glob_rank = []
    glob_recall = {1:0.0,5:0.0,10:0.0}
    for i in range(10):

        ids = random.sample(range(0,len(names)), N)
        im_sub = im_vecs[ids,:]
        instr_sub = instr_vecs[ids,:]
        ids_sub = names[ids]

        # if params.embedding == 'image':
        if type_embedding == 'image':
            sims = np.dot(im_sub,instr_sub.T) # for im2recipe
        else:
            sims = np.dot(instr_sub,im_sub.T) # for recipe2im

        med_rank = []
        recall = {1:0.0,5:0.0,10:0.0}

        for ii in idxs:

            name = ids_sub[ii]
            # get a column of similarities
            sim = sims[ii,:]

            # sort indices in descending order
            sorting = np.argsort(sim)[::-1].tolist()

            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii)

            if (pos+1) == 1:
                recall[1]+=1
            if (pos+1) <=5:
                recall[5]+=1
            if (pos+1)<=10:
                recall[10]+=1

            # store the position
            med_rank.append(pos+1)

        for i in recall.keys():
            recall[i]=recall[i]/N

        med = np.median(med_rank)
        #print("median", med)

        for i in recall.keys():
            glob_recall[i]+=recall[i]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i]/10

    median = np.average(glob_rank)
    recall = glob_recall
    print("Mean median", median)
    print("Recall", recall)
    return median, recall


def run(args, model_inputs=None):
    loaders, model, criterion = im2recipe(args)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loss_history = np.zeros(args.epochs)
    val_loss_history = np.zeros(args.epochs)
    train_median_history = np.zeros(args.epochs)
    val_median_history = np.zeros(args.epochs)

    for epoch_idx in range(args.epochs):
        avg_train_loss, (train_median, train_recall) = train(epoch_idx, loaders[0], model, optimizer, criterion, args)
        avg_val_loss, (val_median, val_recall) = validate(epoch_idx, loaders[1], model, criterion, args)

        avg_train_loss = avg_train_loss.item()
        avg_val_loss = avg_val_loss.item()
        #train_perplex = np.exp(avg_train_loss)
        #val_perplex = np.exp(avg_val_loss)

        train_loss_history[epoch_idx] = avg_train_loss
        val_loss_history[epoch_idx] = avg_val_loss
        #train_perp_history[epoch_idx] = train_perplex
        #val_perp_history[epoch_idx] = val_perplex
        train_median_history[epoch_idx] = train_median
        val_median_history[epoch_idx] = val_median

    #return train_loss_history, val_loss_history, train_perp_history, val_perp_history, train_median_history, val_median_history
    return train_loss_history, val_loss_history, train_median_history, val_median_history

if not(torch.cuda.device_count()):
    device = torch.device(*('cpu',0))
else:
    torch.cuda.manual_seed(1234)
    device = torch.device(*('cuda',0))

'''
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


def train(epoch, data_loader, model, optimizer, criterion, args):
    print('Training')
    iter_time = AverageMeter()
    losses = AverageMeter()
    # acc = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()
        # use index 0 if criterion is CosineSimilarity, index 1 for image class
        data = [data[i].to(device) for i in range(len(data))]
        target = [target[i].to(device) for i in range(len(target))]

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
            if idx == 0:
                img_store = out_image.data.cpu().numpy()
                recipe_store = out_recipe.data.cpu().numpy()
                recipe_id_store = target[-1].data.cpu().numpy()
            else:
                img_store = np.concatenate((img_store, out_image.data.cpu().numpy()))
                recipe_store = np.concatenate((recipe_store, out_recipe.data.cpu().numpy()))
                recipe_id_store = np.concatenate((recipe_id_store, target[-1].data.cpu().numpy()))
        # batch_acc = accuracy(out, target)

        losses.update(loss, out_image.shape[0])
        # acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f} avg)\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f} avg)\t'
                   # 'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t'
                   )
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses
                          # , top1=acc
                          ))
    metric_results = None
    if args.generate_metrics:
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
    # acc = AverageMeter()

    # num_class = 10
    # cm = torch.zeros(num_class, num_class)
    # evaluation loop
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()
        data = [data[i].to(device) for i in range(len(data))]
        target = [target[i].to(device) for i in range(len(target))]

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
            if idx == 0:
                img_store = out_image.data.cpu().numpy()
                recipe_store = out_recipe.data.cpu().numpy()
                recipe_id_store = target[-1].data.cpu().numpy()
            else:
                img_store = np.concatenate((img_store, out_image.data.cpu().numpy()))
                recipe_store = np.concatenate((recipe_store, out_recipe.data.cpu().numpy()))
                recipe_id_store = np.concatenate((recipe_id_store, target[-1].data.cpu().numpy()))
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
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  )
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
    metric_results = None
    if args.generate_metrics:
        metric_store = {}
        metric_store['image'] = img_store
        metric_store['recipe'] = recipe_store
        metric_store['recipe_id'] = recipe_id_store

        metric_results = generate_metrics(args, metric_store) # returns median, recall

    return losses.avg, metric_results


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
    image_loader = ImageLoader(args.image_path, transform_train, data_path=args.data_path, partition='train')
    num_images = len(image_loader)
    indexes = np.arange(num_images)
    # np.random.shuffle(indexes)
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
    return (train_loader, val_loader), model, (cos_criterion, entropy_criterion)
'''
if __name__ == '__main__':
    #model_inputs = pre_process()
    main(model_inputs=None)
    #run_pickle_data('experiments/results_dict.pkl')
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--config', default='configs/config_fullmodel.yaml')
    #args = parser.parse_args()
    #generate_metrics(args, 'metric_store_0.pkl')
