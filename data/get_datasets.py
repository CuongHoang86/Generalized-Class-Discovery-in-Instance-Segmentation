from data_utils import MergedDataset

from cifar import get_cifar_10_datasets, get_cifar_100_datasets
# from herbarium_19 import get_herbarium_datasets
# from stanford_cars import get_scars_datasets
# from imagenet import get_imagenet_100_datasets, get_imagenet_1k_datasets
# from cub import get_cub_datasets
# from fgvc_aircraft import get_aircraft_datasets

from copy import deepcopy
import pickle
import os
from PIL import Image
from config import osr_split_dir
from torchvision.transforms.transforms import ToPILImage

get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    # 'imagenet_100': get_imagenet_100_datasets,
    # 'imagenet_1k': get_imagenet_1k_datasets,
    # 'herbarium_19': get_herbarium_datasets,
    # 'cub': get_cub_datasets,
    # 'aircraft': get_aircraft_datasets,
    # 'scars': get_scars_datasets
}



def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=False)
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def get_class_splits(args):

    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']

    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'imagenet_1k':

        args.image_size = 224
        args.train_classes = range(500)
        args.unlabeled_classes = range(500, 1000)
    
    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)

    else:

        raise NotImplementedError

    return args

from torch._six import string_classes
import torch
int_classes = int
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
import collections
import torchvision.transforms as transforms
import custom_transforms as custom_tr


def collate_custom(batch):
    # print(batch[0],type(batch[0]))
    if isinstance(batch[0], np.int64):
        # print(1)
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
      
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        # print(3)
        return np.stack(batch, 0)

    elif isinstance(batch[0], int_classes):
        # print(4)
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        # print(5)
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], string_classes):
        # print(6)
        return batch

    elif isinstance(batch[0], collections.Mapping):
        # print(7)
        batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        return batch_modified

    elif isinstance(batch[0], collections.Sequence):
        # print(8)
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))

def get_train_transformations():
    return transforms.Compose([
        custom_tr.RandomHorizontalFlip(),
        custom_tr.ToTensor(),
        custom_tr.Resize(size=224),
        custom_tr.RandomResizedCrop(size=224),
        transforms.RandomApply([custom_tr.ColorJitter(jitter=[0.4,0.4,0.4,0.1])], p=0.8),
        # custom_tr.RandomGrayscale(),
        custom_tr.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])])


def get_train_dataloader( dataset,batch_size):
    return torch.utils.data.DataLoader(dataset, num_workers=1, 
            batch_size=batch_size, pin_memory=True, 
            collate_fn=collate_custom, drop_last=True, shuffle=True)

class Obdata(data.Dataset):
    
   
    def __init__(self, root='',
                 split='val', transform=None, download=False, ignore_classes=[]):
        # Set paths
        self.root = root
        # valid_splits = ['trainaug', 'train', 'val']
        # assert(split in valid_splits)
        self.split = split

        # if split == 'trainaug':
        #     _semseg_dir = os.path.join(self.root, 'SegmentationClassAug')
        # else:
        #     _semseg_dir = os.path.join(self.root, 'SegmentationClass')

        _image_dir = os.path.join('/home/cuonghoang/Desktop/codedict/GCC-master2/new_image')
        _semseg_dir = os.path.join('/home/cuonghoang/Desktop/codedict/GCC-master2/new_sal')

        # Download
        # if download:
        #     self._download()

        # Transform
        self.transform = transform

        # Splits are pre-cut
        # print("Initializing dataloader for PASCAL VOC12 {} set".format(''.join(self.split)))
        split_file = os.path.join('/home/cuonghoang/Desktop/codedict/GCC-master2/image.txt')
        split_label = os.path.join('/home/cuonghoang/Desktop/codedict/GCC-master2/mask_label.txt')

        # split_file = os.path.join('image (copy).txt')
        # split_label = os.path.join('label (copy).txt')

        self.images = []
        self.semsegs = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        with open(split_label, "r") as f1:
            self.lines1 = f1.read().splitlines()
            # print(lines)

        for ii, line in enumerate(lines):
            # Images
            _image = os.path.join(_image_dir, line + ".png")
            assert os.path.isfile(_image)
            self.images.append(_image)

            # Semantic Segmentation
            _semseg = os.path.join(_semseg_dir, line + '.png')
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

        assert(len(self.images) == len(self.semsegs))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

        # List of classes which are remapped to ignore index.
        # self.ignore_classes = [self.VOC_CATEGORY_NAMES.index(class_name) for class_name in ignore_classes]
        # print(' self.ignore_classes', self.ignore_classes)

    def update_neighbors(self, indices,indices1):
        print ("update neighbors!!!")
        self.indices = indices
        self.indices1 = indices1

    def update_distance(self, distance):
        print ("update distance!!!")
        self.distance = distance
    def __getitem__(self, index):  #main function
        sample = {}

        # Load image
        _img = self._load_img(index)
        # print(type(_img))
        sample['image'] = _img
        sample['augmented'] = _img
        sample['neighbor'] = _img
        
        # sample['target'] = self.all.index(int(self.lines1[index]))

        # Load pixel-level annotations
        _semseg = self._load_semseg(index)
        # print('unique',np.unique(_semseg))
        if _semseg.shape != _img.shape[:2]:
            _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        sample['semseg'] = _semseg
        sample['semseg_aug'] = _semseg
        sample['semseg_nei'] = _semseg
        
        mask_lab=1

        if int(self.lines1[index])==-1:
            mask_lab=0

        sample['meta'] = {'im_size': (_img.shape[0], _img.shape[1]),
                          'image_file': self.images[index],
                          'index': index,
                          'image': os.path.basename(self.semsegs[index]).split('.')[0],
                          'label':int(self.lines1[index]),
                          'mask':mask_lab,
                        
                          'score':1}
            
        if self.transform is not None:
            sample = self.transform(sample)

        
        
        

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        # _img=torch.tensor(_img)
        # _img1 = Image.open(self.images[index])
        return _img

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index]))
        
        # for ignore_class in self.ignore_classes:
        _semseg[_semseg == 255] = 1
        # _semseg=torch.tensor(_semseg)
       
        return _semseg

    # def get_img_size(self, idx=0):
    #     img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[idx] + '.jpg'))
    #     return list(reversed(img.size))

    def __str__(self):
        return 'VOC12(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES
