import yaml
import argparse
import copy

import numpy as np
import torch
import torchvision

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from data_loader import ImageLoader
from lstm_model import IngredModel

import sys
sys.path.append('..')
from main import train, validate, adjust_learning_rate

parser = argparse.ArgumentParser(description='CS7643 Assignment-2 Part 2')
parser.add_argument('--config', default='./config_lstm.yaml')


def im2recipe():
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
    train_loader = torch.utils.data.DataLoader(image_loader, batch_size=args.batch_size, sampler=np.arange(int(0.01*num_images)))
    print('loaded data...')
    model = IngredModel(args)
    print('instantiated model...')
    criterion = nn.CosineEmbeddingLoss(0.1)
    return train_loader, model, criterion


def recipe2im():
    return None, None, None

def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('loaded args...')
    train_loader, model, criterion = im2recipe() if args.model == 'im2recipe' else recipe2im()
    print('data loader complete...')
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print('instantiated optimizer...')
    best = 0.0
    best_cm = None
    best_model = None
    for epoch in range(args.epochs):
        print('epoch {}'.format(epoch))
        adjust_learning_rate(optimizer, epoch, args)
        print('learning rate adjusted')

        # train loop
        train(epoch, train_loader, model, optimizer, criterion)
        print('training complete')

        # validation loop, change back to test_loader
        acc, cm = validate(epoch, train_loader, model, criterion)
        print('validation complete')
        if acc > best:
            best = acc
            best_cm = cm
            best_model = copy.deepcopy(model)

    print('Best Prec @1 Acccuracy: {:.4f}'.format(best))
    per_cls_acc = best_cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    if args.save_best:
        torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + '.pth')


if __name__ == '__main__':
    main()
