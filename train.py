import argparse
import json
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from datasets import load_class_names, prepare_loader, separate_class
from models import construct_model
from models.network_v1 import NetworkV1, NetworkV2
from test import test_v1, test_v2


def train_v2(ep, model, optimizer, lr_scheduler, train_loader, device, config):
    lr_scheduler.step()
    model.train()

    loss_meter = 0
    acc_meter = 0
    make_acc_meter = 0
    type_acc_meter = 0

    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target, make_target, type_target in train_loader:
        data = data.to(device)
        target = target.to(device)
        make_target = make_target.to(device)
        type_target = type_target.to(device)

        optimizer.zero_grad()

        pred, make_pred, type_pred = model(data)

        loss_main = F.cross_entropy(pred, target)
        loss_make = F.cross_entropy(make_pred, make_target)
        loss_type = F.cross_entropy(type_pred, type_target)

        loss = loss_main + config['make_loss'] * loss_make + config['type_loss'] * loss_type
        loss.backward()

        optimizer.step()

        acc = pred.max(1)[1].eq(target).float().mean()
        make_acc = make_pred.max(1)[1].eq(make_target).float().mean()
        type_acc = type_pred.max(1)[1].eq(type_target).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        make_acc_meter += make_acc.item()
        type_acc_meter += type_acc.item()

        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} '
              f'Make: {make_acc_meter / i:.4f} '
              f'Type: {type_acc_meter / i:.4f} '
              f'({elapsed:.2f}s)', end='\r')

    print()
    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)
    make_acc_meter /= len(train_loader)
    type_acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,
        'train_make_acc': make_acc_meter,
        'train_type_acc': type_acc_meter,
        'train_time': elapsed
    }

    return trainres


def train_v1(ep, model, optimizer, lr_scheduler, train_loader, device, config):
    lr_scheduler.step()
    model.train()

    loss_meter = 0
    acc_meter = 0
    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        pred = model(data)

        loss = F.cross_entropy(pred, target)
        loss.backward()

        optimizer.step()

        acc = pred.max(1)[1].eq(target).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} ({elapsed:.2f}s)', end='\r')

    print()
    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,
        'train_time': elapsed
    }

    return trainres


def get_exp_dir(config):
    exp_dir = f'logs/{config["arch"]}_{config["imgsize"][0]}_{config["epochs"]}_v{config["version"]}'

    if config['finetune']:
        exp_dir += '_finetune'
        
    os.makedirs(exp_dir, exist_ok=True)

    exps = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    files = set(map(int, exps))
    if len(files):
        exp_id = min(set(range(1, max(files) + 2)) - files)
    else:
        exp_id = 1

    exp_dir = os.path.join(exp_dir, str(exp_id))
    os.makedirs(exp_dir, exist_ok=True)

    json.dump(config, open(exp_dir + '/config.json', 'w'))

    return exp_dir


def load_weight(model, path, device):
    sd = torch.load(path)
    model.load_state_dict(sd)


def main(args):
    device = torch.device('cuda')

    config = {
        'batch_size': args.batch_size,
        'test_batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'epochs': args.epochs,
        'imgsize': (args.imgsize, args.imgsize),
        'arch': args.arch,
        'version': args.version,
        'make_loss': args.make_loss,
        'type_loss': args.type_loss,
        'finetune': args.finetune,
        'path': args.path
    }

    exp_dir = get_exp_dir(config)

    class_names = load_class_names()
    num_classes = len(class_names)
    v2_info = separate_class(class_names)
    num_makes = len(v2_info['make'].unique())
    num_types = len(v2_info['model_type'].unique())

    model = construct_model(config, num_classes, num_makes, num_types)

    if config['finetune']:
        load_weight(model, config['path'], device)

    model = model.to(device)

    optimizer = optim.SGD(model.parameters(),
                          lr=config['lr'],
                          momentum=config['momentum'],
                          weight_decay=config['weight_decay'])

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  [100, 150],
                                                  gamma=0.1)

    train_loader, test_loader = prepare_loader(config)

    best_acc = 0
    res = []

    if config['version'] == 1:
        train_fn = train_v1
        test_fn = test_v1
    else:
        train_fn = train_v2
        test_fn = test_v2

    for ep in range(1, config['epochs'] + 1):
        trainres = train_fn(ep, model, optimizer, lr_scheduler, train_loader, device, config)
        valres = test_fn(model, test_loader, device, config)
        trainres.update(valres)

        if best_acc < valres['val_acc']:
            best_acc = valres['val_acc']
            torch.save(model.state_dict(), exp_dir + '/best.pth')

        res.append(trainres)

    print(f'Best accuracy: {best_acc:.4f}')
    res = pd.DataFrame(res)
    res.to_csv(exp_dir + '/history.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and finetuning script for Cars classification task')

    # training arg
    parser.add_argument('--batch-size', default=32, type=int,
                        help='training batch size (default: 32)')
    parser.add_argument('--epochs', default=40, type=int,
                        help='training epochs (default: 40)')
    parser.add_argument('--arch', default='resnext50', choices=['resnext50',
                                                                'resnet34',
                                                                'mobilenetv2'],
                        help='Architecture (default: resnext50)')
    parser.add_argument('--imgsize', default=400, type=int,
                        help='Input image size (default: 400)')
    parser.add_argument('--version', default=1, type=int, choices=[1, 2],
                        help='Classification version (default: 1)\n'
                             '1. Cars Model only\n'
                             '2. Cars Model + Make + Car Type')
    parser.add_argument('--finetune', default=False, action='store_true',
                        help='whether to finetune from 400x400 to 224x224 (default: False)')
    parser.add_argument('--path',
                        help='required if it is a finetune task (default: None)')

    # optimizer arg
    parser.add_argument('--lr', default=0.01, type=float,
                        help='SGD learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', default=0.0001, type=float,
                        help='SGD weight decay (default: 0.0001)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum (default: 0.9)')

    # multi-task learning arg
    parser.add_argument('--make-loss', default=0.2, type=float,
                        help='loss$_{make}$ lambda')
    parser.add_argument('--type-loss', default=0.2, type=float,
                        help='loss$_{type}$ lambda')

    args = parser.parse_args()
    main(args)
