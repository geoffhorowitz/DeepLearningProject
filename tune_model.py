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
from main import main

from collections import namedtuple
from plot_methods import plot_complex_learning_curve, plot_complexity_curve


def tune_main(model_inputs=None):
    default_dict = {
        #Train:
        'batch_size': 256,
        'learning_rate': 0.0001,
        'epochs': 10,
        'embed_dim': 1024,
        'num_classes': 1048,
        'train_percent': 0.6,
        'val_percent': 0.2,
        'semantic_reg': False,
        'cos_weight': 0.8,
        'image_weight': 0.1,
        'recipe_weight': 0.1,
        'workers': 4,
        'mismatch': 0.5,
        #network:
        'model': 'im2recipe',
        'recipe_model': 'lstm', # lstm or transformer
        #data:
        'data_path': 'data',
        'image_path': 'data/images',
        'generate_metrics': True,
        'metric_type': 'both', # rank or accuracy or both
        'save_best': False,
        #image_model:
        'freeze_image': True,
        #ingredient_lstm:
        'ingredient_lstm_dim': 300,
        'ingredient_embedding_dim': 300, # vocab size 30167 x 300 embedded
        'ingredient_w2v_path': 'data/vocab.bin',
        'ingred_model_variant': 'paper', # base, custom_basic, custom_fusion, paper
        'ingred_dropout': .2,
        #recipe_lstm:
        'recipe_lstm_dim': 1024,
        'recipe_embedding_dim': 1024,
        'recipe_model_variant': 'paper', # base, custom_basic, custom_fusion, paper
        'recipe_dropout': .2,
        #transformer:
        'hidden_dim': 256,
        'num_heads': 4,
        'dim_feedforward': 256, # 2048 original
    }
    

    experiment_dict = {
        #'learning_rate': [1e-2, 1e-3, 1e-4], # 1e-3
        #'epochs': [10, 20],
        'mismatch': [.8, .5, .2],
        #'ingred_model_variant': ['base', 'custom_basic', 'custom_fusion', 'paper']
        #'epochs': [2]
    }


    runs_per_experiment = 2 # balance b/w trials and hyperparams to test

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
            train_imacc_all = np.zeros((runs_per_experiment, n_epochs))
            train_recacc_all = np.zeros((runs_per_experiment, n_epochs))

            val_loss_all = np.zeros((runs_per_experiment, n_epochs))
            #val_perp_all = np.zeros((runs_per_experiment, n_epochs))
            val_median_all = np.zeros((runs_per_experiment, n_epochs))
            val_imacc_all = np.zeros((runs_per_experiment, n_epochs))
            val_recacc_all = np.zeros((runs_per_experiment, n_epochs))


            fh = open('experiments/{}_{}_results.txt'.format(key, exp_val), 'w')

            for k in range(runs_per_experiment):
                input_dict = default_dict.copy()
                input_dict[key] = exp_val

                input_dict['recipe_model_variant'] = input_dict['ingred_model_variant']
                
                args = namedtuple("args", input_dict.keys())(*input_dict.values()) # to get it in the same dot callable format

                train_loss, val_loss, train_median, val_median, train_imacc, val_imacc, train_recacc, val_recacc = run(args, model_inputs)

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
                results_dict[key][exp_val][k]['train_imacc'] = train_imacc
                results_dict[key][exp_val][k]['val_imacc'] = val_imacc
                results_dict[key][exp_val][k]['train_recacc'] = train_recacc
                results_dict[key][exp_val][k]['val_recacc'] = val_recacc

                train_loss_all[k, :] = train_loss
                train_median_all[k, :] = train_median
                train_imacc_all[k, :] = train_imacc
                train_recacc_all[k, :] = train_recacc
                val_loss_all[k, :] = val_loss
                val_median_all[k, :] = val_median
                val_imacc_all[k, :] = val_imacc
                val_recacc_all[k, :] = val_recacc

                # text results
                results_text = '\nexperiment {}={} run {}:'.format(key, exp_val, k) + \
                                '\n\ttrain loss: {}'.format(np.round(train_loss[-1], 4)) + \
                                '\n\ttrain median: {}'.format(np.round(train_median[-1], 4)) + \
                                '\n\tvalid loss: {}'.format(np.round(val_loss[-1], 4)) + \
                                '\n\tvalid median: {}'.format(np.round(val_median[-1], 4)) + \
                                '\n\ttrain_imacc: {}'.format(np.round(train_imacc[-1], 4)) + \
                                '\n\tval_imacc: {}'.format(np.round(val_imacc[-1], 4)) + \
                                '\n\ttrain_recacc: {}'.format(np.round(train_recacc[-1], 4)) + \
                                '\n\tval_recacc: {}'.format(np.round(val_recacc[-1], 4))

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
            results_dict[key][exp_val]['train_imacc_mean'] = np.mean(train_imacc_all, axis=0)
            results_dict[key][exp_val]['train_imacc_std'] = np.std(train_imacc_all, axis=0)
            results_dict[key][exp_val]['val_imacc_mean'] = np.mean(val_imacc_all, axis=0)
            results_dict[key][exp_val]['val_imacc_std'] = np.std(val_imacc_all, axis=0)
            results_dict[key][exp_val]['train_recacc_mean'] = np.mean(train_recacc_all, axis=0)
            results_dict[key][exp_val]['train_recacc_std'] = np.std(train_recacc_all, axis=0)
            results_dict[key][exp_val]['val_recacc_mean'] = np.mean(val_recacc_all, axis=0)
            results_dict[key][exp_val]['val_recacc_std'] = np.std(val_recacc_all, axis=0)


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
    plot_complexity_curve(results_dict, logx_scale=True)


def run(args, model_inputs=None):
    loaders, model, criterion, val_indexes = im2recipe(args)

    model.image_model = torch.nn.DataParallel(model.image_model)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loss_history = np.zeros(args.epochs)
    val_loss_history = np.zeros(args.epochs)
    train_median_history = np.zeros(args.epochs)
    val_median_history = np.zeros(args.epochs)
    train_imacc_history = np.zeros(args.epochs)
    val_imacc_history = np.zeros(args.epochs)
    train_recacc_history = np.zeros(args.epochs)
    val_recacc_history = np.zeros(args.epochs)

    for epoch_idx in range(args.epochs):
        avg_train_loss, train_metrics = train(epoch_idx, loaders[0], model, optimizer, criterion, args)
        avg_val_loss, val_metrics, _ = validate(epoch_idx, loaders[1], model, criterion, args)

        (train_acc_image, train_acc_recipe), (train_median, train_recall) = train_metrics
        (val_acc_image, val_acc_recipe), (val_median, val_recall) = val_metrics

        avg_train_loss = avg_train_loss.item()
        avg_val_loss = avg_val_loss.item()
        #train_perplex = np.exp(avg_train_loss)
        #val_perplex = np.exp(avg_val_loss)

        train_loss_history[epoch_idx] = avg_train_loss
        val_loss_history[epoch_idx] = avg_val_loss

        train_median_history[epoch_idx] = train_median
        val_median_history[epoch_idx] = val_median

        train_imacc_history[epoch_idx] = train_acc_image.item()
        val_imacc_history[epoch_idx] = val_acc_image.item()
        train_recacc_history[epoch_idx] = train_acc_recipe.item()
        val_recacc_history[epoch_idx] = val_acc_recipe.item()

    #return train_loss_history, val_loss_history, train_perp_history, val_perp_history, train_median_history, val_median_history
    return train_loss_history, val_loss_history, train_median_history, val_median_history, train_imacc_history, val_imacc_history, train_recacc_history, val_recacc_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alphabet Soup Final Project')
    parser.add_argument('--config', default='configs/config_tuning.yaml')
    model_args = parser.parse_args()

    tune_main(model_inputs=None)
    #run_pickle_data('experiments/results_dict.pkl')
