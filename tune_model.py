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

from plot_methods import plot_simple_learning_curve, plot_complex_learning_curve, plot_complexity_curve
from main import


def main(model_inputs):
    default_dict = {
        'encoder_emb_size': 64, #32 original
        'encoder_hidden_size': 64,
        'encoder_dropout': 0.1, #0.2 original

        'decoder_emb_size': 32,
        'decoder_hidden_size': 64,
        'decoder_dropout': 0.1, #0.2 original

        'learning_rate': 1e-2, # 1e-3 original
        'model_type': "LSTM",
        'epochs': 10,
        'translate': False
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


    runs_per_experiment = 3 # balance b/w trials and hyperparams to test

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

                # additional argument contraints
                input_dict['decoder_hidden_size'] = input_dict['encoder_hidden_size']

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
        train_loss, avg_train_loss = train(epoch, loaders[0], model, optimizer, criterion, args)
        val_loss, avg_val_loss = validate(epoch, loaders[1], model, criterion, args)

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
