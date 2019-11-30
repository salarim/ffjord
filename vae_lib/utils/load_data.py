from __future__ import print_function

import torch
import torch.utils.data as data_utils
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle
from scipy.io import loadmat

import numpy as np

import os


def load_static_mnist(args, **kwargs):
    """
    Dataloading function for static mnist. Outputs image data in vectorized form: each image is a vector of size 784
    """
    args.dynamic_binarization = False
    args.input_type = 'binary'

    args.input_size = [1, 28, 28]

    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])

    with open(os.path.join('data', 'MNIST_static', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    x_train = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('data', 'MNIST_static', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    x_val = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('data', 'MNIST_static', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = lines_to_np_array(lines).astype('float32')

    # shuffle train data
    np.random.shuffle(x_train)

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))
    y_val = np.zeros((x_val.shape[0], 1))
    y_test = np.zeros((x_test.shape[0], 1))

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args


def load_freyfaces(args, **kwargs):
    # set args
    args.input_size = [1, 28, 20]
    args.input_type = 'multinomial'
    args.dynamic_binarization = False

    TRAIN = 1565
    VAL = 200
    TEST = 200

    # start processing
    with open('data/Freyfaces/freyfaces.pkl', 'rb') as f:
        data = pickle.load(f, encoding="latin1")[0]

    data = data / 255.

    # NOTE: shuffling is done before splitting into train and test set, so test set is different for every run!
    # shuffle data:
    np.random.seed(args.freyseed)

    np.random.shuffle(data)

    # train images
    x_train = data[0:TRAIN].reshape(-1, 28 * 20)
    # validation images
    x_val = data[TRAIN:(TRAIN + VAL)].reshape(-1, 28 * 20)
    # test images
    x_test = data[(TRAIN + VAL):(TRAIN + VAL + TEST)].reshape(-1, 28 * 20)

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))
    y_val = np.zeros((x_val.shape[0], 1))
    y_test = np.zeros((x_test.shape[0], 1))

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, args


def load_omniglot(args, **kwargs):
    n_validation = 1345

    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = True

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='F')

    omni_raw = loadmat(os.path.join('data', 'OMNIGLOT', 'chardata.mat'))

    # train and test data
    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # shuffle train data
    np.random.shuffle(train_data)

    # set train and validation data
    x_train = train_data[:-n_validation]
    x_val = train_data[-n_validation:]

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))
    y_val = np.zeros((x_val.shape[0], 1))
    y_test = np.zeros((x_test.shape[0], 1))

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args


def load_caltech101silhouettes(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='F')

    caltech_raw = loadmat(os.path.join('data', 'Caltech101Silhouettes', 'caltech101_silhouettes_28_split1.mat'))

    # train, validation and test data
    x_train = 1. - reshape_data(caltech_raw['train_data'].astype('float32'))
    np.random.shuffle(x_train)
    x_val = 1. - reshape_data(caltech_raw['val_data'].astype('float32'))
    np.random.shuffle(x_val)
    x_test = 1. - reshape_data(caltech_raw['test_data'].astype('float32'))

    y_train = caltech_raw['train_labels']
    y_val = caltech_raw['val_labels']
    y_test = caltech_raw['test_labels']

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args


def load_mymnist(args, **kwargs):
    args.dynamic_binarization = False
    args.input_type = 'binary'
    args.input_size = [1, 28, 28]
    args.num_labels = 10
    args.labels = list(range(10))

    train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

    test_dataset = datasets.MNIST(root='data', 
                                train=False, 
                                transform=transforms.ToTensor())


    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False)

    return train_loader, test_loader, test_loader, args



def inf_train_gen_with_labels(data, rng=None, num_samples=200, labels=[]):
    if rng is None:
        rng = np.random.RandomState()

    if data == "gaussians":
        if len(labels) == 0:
            labels = [8]
        scale = 1.
        dataset = []
        dataset_labels = []
        for label in labels:
            centers = [(np.cos(i*2*np.pi/label),np.sin(i*2*np.pi/label)) for i in range(label)]
            centers = [(scale * x, scale * y) for x, y in centers]
            for i in range(num_samples//len(labels)):
                point = rng.randn(2) * np.sin(2*np.pi/label)/2
                idx = rng.randint(label)
                center = centers[idx]
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
                dataset_labels.append(labels.index(label))

        dataset = torch.tensor(dataset).float()
        dataset /= 1.414
        dataset_labels = torch.tensor(dataset_labels).long()
        return dataset, dataset_labels
    else: 
        return inf_train_gen_with_labels("gaussians", rng, num_samples, labels)

def load_synthetic(args, **kwargs):
    
    from torch.utils.data import Dataset, DataLoader

    class SyntheticData(Dataset):
        def __init__(self, data_type, num_samples, labels):
            self.type = data_type
            self.labels = labels
            self.num_samples = num_samples
            self.data, self.data_classes = inf_train_gen_with_labels(data=self.type, num_samples=self.num_samples, labels=self.labels)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return self.data[idx], self.data_classes[idx]
    
    args.dynamic_binarization = False
    args.input_type = 'synthetic'
    args.input_size = [2]
    args.num_labels = 5
    
    labels = [1, 2, 3, 4, 5]
    train_dataset = SyntheticData(data_type='gaussians', num_samples=20000, labels=labels)
    val_dataset = SyntheticData(data_type='gaussians', num_samples=5000, labels=labels)
    test_dataset = SyntheticData(data_type='gaussians', num_samples=2000, labels=labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args

def load_dataset(args, **kwargs):

    if args.dataset == 'mnist':
        train_loader, val_loader, test_loader, args = load_static_mnist(args, **kwargs)
    elif args.dataset == 'caltech':
        train_loader, val_loader, test_loader, args = load_caltech101silhouettes(args, **kwargs)

    elif args.dataset == 'freyfaces':
        train_loader, val_loader, test_loader, args = load_freyfaces(args, **kwargs)
    elif args.dataset == 'omniglot':
        train_loader, val_loader, test_loader, args = load_omniglot(args, **kwargs)
    elif args.dataset == 'mymnist':
        train_loader, val_loader, test_loader, args = load_mymnist(args, **kwargs)
    elif args.dataset == 'synthetic':
        train_loader, val_loader, test_loader, args = load_synthetic(args, **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')

    return train_loader, val_loader, test_loader, args
