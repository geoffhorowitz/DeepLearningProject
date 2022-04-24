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

from plot_methods import plot_simple_learning_curve, plot_complex_learning_curve, plot_complexity_curve
from main import


def main(model_inputs):
    default_dict = {
        #Train:
        'batch_size': 100,
        'learning_rate': 0.001,
        'reg': 0.00005,
        'epochs': 15,
        'steps': [6, 8],
        'warmup': 0,
        'momentum': 0.9,
        'embed_dim': 1024,
        'num_classes': 1048,
        'train_percent': 0.1,
        'val_percent': 0.1,
        'semantic_reg': False,
        'cos_weight': 0.98,
        'image_weight': 0.01,
        'recipe_weight': 0.01,
        'workers': 1,
        #network:
        'model': im2recipe,
        #data:
        'data_path': 'data',
        'image_path': 'data/images',
        'save_best': True,
        #ingredient_lstm:
        'ingredient_lstm_dim': 300,
        'ingredient_embedding_dim': 300, # vocab size 30167 x 300 embedded
        'ingredient_w2v_path': 'data/vocab.bin',
        #recipe_lstm:
        'recipe_lstm_dim': 1024,
        'recipe_embedding_dim': 1024,
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
        'epochs': [10]
    }


    runs_per_experiment = 2 # balance b/w trials and hyperparams to test

    results_dict = {'defaults': default_dict}

    for i, (key, val) in enumerate(experiment_dict.items()):
        results_dict[key] = {}
        for j, (exp_val) in enumerate(val):
            results_dict[key][exp_val] = {}
            best_perp = 1e10
            n_epochs = default_dict['epochs'] if 'epochs' != key else exp_val
            train_loss_all = np.zeros((runs_per_experiment, n_epochs))
            train_perp_all = np.zeros((runs_per_experiment, n_epochs))
            val_loss_all = np.zeros((runs_per_experiment, n_epochs))
            val_perp_all = np.zeros((runs_per_experiment, n_epochs))

            fh = open('experiments/{}_{}_results.txt'.format(key, exp_val), 'w')

            for k in range(runs_per_experiment):
                input_dict = default_dict.copy()
                input_dict[key] = exp_val

                args = namedtuple("args", input_dict.keys())(*input_dict.values()) # to get it in the same dot callable format

                train_loss, val_loss, train_perp, val_perp, model = run(args, model_inputs)

                # save pertinent results for later plotting
                if val_perp[-1] < best_perp:
                    best_perp = val_perp[-1]
                    best_run = k

                results_dict[key][exp_val][k] = {}
                results_dict[key][exp_val][k]['id'] = '{}.{}.{}'.format(i+1, j+1, k+1)
                results_dict[key][exp_val][k]['train_loss'] = train_loss
                results_dict[key][exp_val][k]['train_perp'] = train_perp
                results_dict[key][exp_val][k]['val_loss'] = val_loss
                results_dict[key][exp_val][k]['val_perp'] = val_perp

                train_loss_all[k, :] = train_loss
                train_perp_all[k, :] = train_perp
                val_loss_all[k, :] = val_loss
                val_perp_all[k, :] = val_perp

                # text results
                results_text = '\nexperiment {}={} run {}:'.format(key, exp_val, k) + \
                                '\n\ttrain loss: {}'.format(np.round(train_loss[-1], 4)) + \
                                '\n\ttrain perplexity: {}'.format(np.round(train_perp[-1], 4)) + \
                                '\n\tvalid loss: {}'.format(np.round(val_loss[-1], 4)) + \
                                '\n\tvalid perplexity: {}'.format(np.round(val_perp[-1], 4))

                print(results_text)

                fh.write(results_text)

                # plots
                #plot_simple_learning_curve(train_loss, train_perp, val_loss, val_perp, (key, exp_val, i, j, k))

            # compute stats
            results_dict[key][exp_val]['best_perp'] = best_perp
            results_dict[key][exp_val]['best_run'] = best_run
            results_dict[key][exp_val]['train_loss_mean'] = np.mean(train_loss_all, axis=0)
            results_dict[key][exp_val]['train_loss_std'] = np.std(train_loss_all, axis=0)
            results_dict[key][exp_val]['train_perp_mean'] = np.mean(train_perp_all, axis=0)
            results_dict[key][exp_val]['train_perp_std'] = np.std(train_perp_all, axis=0)
            results_dict[key][exp_val]['val_loss_mean'] = np.mean(val_loss_all, axis=0)
            results_dict[key][exp_val]['val_loss_std'] = np.std(val_loss_all, axis=0)
            results_dict[key][exp_val]['val_perp_mean'] = np.mean(val_perp_all, axis=0)
            results_dict[key][exp_val]['val_perp_std'] = np.std(val_perp_all, axis=0)

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


def prepare(args):
    loaders, model, criterion = im2recipe(args) if args.model == 'im2recipe' else recipe2im(args)
    if torch.cuda.is_available():
        model = model.cuda()

    return loaders, model, criterion

def generate_metrics(args, metric_store):
    f = open(metric_store, "rb")
    metric_store = pickle.load(f)
    f.close()

    img_vecs = metric_store['images']
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
        print("median", med)

        for i in recall.keys():
            glob_recall[i]+=recall[i]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i]/10
    print("Mean median", np.average(glob_rank))
    print("Recall", glob_recall)

def run(args, model_inputs=None):
    '''
    TODOs:
        1) add args as input to im2recipe, recipe2im, train, validate functions
        2) add train_loss and avg_train_loss as outputs to train
        3) add avg_val_loss as output to validate
    '''
    #loaders, model, criterion = model_inputs
    loaders, model, criterion = prepare(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loss_history = np.zeros(args.epochs)
    val_loss_history = np.zeros(args.epochs)
    train_perp_history = np.zeros(args.epochs)
    val_perp_history = np.zeros(args.epochs)

    for epoch_idx in range(args.epochs):
        avg_train_loss = train(epoch, loaders[0], model, optimizer, criterion, args)
        avg_val_loss = validate(epoch, loaders[1], model, criterion, args)

        avg_train_loss = avg_train_loss.item()
        avg_val_loss = avg_val_loss.item()
        train_perplex = np.exp(avg_train_loss)
        val_perplex = np.exp(avg_val_loss)

        train_loss_history[epoch_idx] = avg_train_loss
        val_loss_history[epoch_idx] = avg_val_loss
        train_perp_history[epoch_idx] = train_perplex
        val_perp_history[epoch_idx] = val_perplex


    return train_loss_history, val_loss_history, train_perp_history, val_perp_history, seq2seq_model



if __name__ == '__main__':
    model_inputs = pre_process()
    main(model_inputs)
    #run_pickle_data('experiments/results_dict.pkl')
