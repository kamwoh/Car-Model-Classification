import os

import pandas as pd
import scipy.io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_anno(path):
    mat = scipy.io.loadmat(path)
    return mat


def load_class_names(path='data/devkit/cars_meta.mat'):
    cn = load_anno(path)['class_names']
    cn = cn.tolist()[0]
    cn = [str(c[0].item()) for c in cn]
    return cn


def load_annotations_v1(path):
    ann = load_anno(path)['annotations'][0]
    ret = {}

    for idx in range(len(ann)):
        x1, y1, x2, y2, target, imgfn = ann[idx]

        r = {
            'x1': x1.item(),
            'y1': y1.item(),
            'x2': x2.item(),
            'y2': y2.item(),
            'target': target.item() - 1,
            'filename': imgfn.item()
        }

        ret[idx] = r

    return ret


def load_annotations_v2(path, v2_info):
    ann = load_anno(path)['annotations'][0]
    ret = {}
    make_codes = v2_info['make'].astype('category').cat.codes
    type_codes = v2_info['model_type'].astype('category').cat.codes

    for idx in range(len(ann)):
        x1, y1, x2, y2, target, imgfn = ann[idx]

        r = {
            'x1': x1.item(),
            'y1': y1.item(),
            'x2': x2.item(),
            'y2': y2.item(),
            'target': target.item() - 1,
            'make_target': make_codes[target.item() - 1].item(),
            'type_target': type_codes[target.item() - 1].item(),
            'filename': imgfn.item()
        }

        ret[idx] = r
    return ret


def separate_class(class_names):
    arr = []
    for idx, name in enumerate(class_names):
        splits = name.split(' ')
        make = splits[0]
        model = ' '.join(splits[1:-1])
        model_type = splits[-2]

        if model == 'General Hummer SUV':
            make = 'AM General'
            model = 'Hummer SUV'

        if model == 'Integra Type R':
            model_type = 'Type-R'

        if model_type == 'Z06' or model_type == 'ZR1':
            model_type = 'Convertible'

        if 'SRT' in model_type:
            model_type = 'SRT'

        if model_type == 'IPL':
            model_type = 'Coupe'

        year = splits[-1]
        arr.append((idx, make, model, model_type, year))

    arr = pd.DataFrame(arr, columns=['target', 'make', 'model', 'model_type', 'year'])
    return arr


class CarsDatasetV1(Dataset):
    def __init__(self, imgdir, anno_path, transform, size):
        self.annos = load_annotations_v1(anno_path)
        self.imgdir = imgdir
        self.transform = transform
        self.resize = transforms.Resize(size)
        self.cache = {}

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        r = self.annos[idx]

        target = r['target']

        if idx not in self.cache:
            fn = r['filename']

            img = Image.open(os.path.join(self.imgdir, fn))
            img = img.convert('RGB')
            img = self.resize(img)
            
            self.cache[idx] = img
        else:
            img = self.cache[idx]
        
        img = self.transform(img)

        return img, target


class CarsDatasetV2(Dataset):
    def __init__(self, imgdir, anno_path, transform, size):
        self.class_names = load_class_names()
        self.v2_info = separate_class(self.class_names)
        self.annos = load_annotations_v2(anno_path, self.v2_info)
        self.imgdir = imgdir
        self.transform = transform
        self.resize = transforms.Resize(size)
        self.cache = {}

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        r = self.annos[idx]

        target = r['target']
        make_target = r['make_target']
        type_target = r['type_target']

        if idx not in self.cache:
            fn = r['filename']

            img = Image.open(os.path.join(self.imgdir, fn))
            img = img.convert('RGB')
            img = self.resize(img)
            
            self.cache[idx] = img
        else:
            img = self.cache[idx]
        
        img = self.transform(img)

        return img, target, make_target, type_target


def prepare_loader(config):
    train_imgdir = 'data/cars_train'
    test_imgdir = 'data/cars_test'

    train_annopath = 'data/devkit/cars_train_annos.mat'
    test_annopath = 'data/devkit/cars_test_annos_withlabels.mat'

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )

    CarsDataset = CarsDatasetV1 if config['version'] == 1 else CarsDatasetV2

    train_dataset = CarsDataset(train_imgdir, train_annopath, train_transform, config['imgsize'])
    test_dataset = CarsDataset(test_imgdir, test_annopath, test_transform, config['imgsize'])

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              pin_memory=False,
                              num_workers=12)
    test_loader = DataLoader(test_dataset,
                             batch_size=config['test_batch_size'],
                             shuffle=False,
                             pin_memory=False,
                             num_workers=12)

    return train_loader, test_loader
