import os
import random
import numpy as np
import sys

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
        if type_embedding == 'images':
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


def accuracy(pred, target):
    ndx_max = np.argmax(pred, axis=1)
    acc = (np.equal(ndx_max, target)/target.shape[0]).sum()
    return acc
