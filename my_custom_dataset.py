from PIL import Image
from os.path import join
from torch.utils.data import Dataset

dataset_root_path_default = 'datasets/CUB'


# generate 2 dictionary that convert class name to class index and vice versa
def get_class_dicts( root=dataset_root_path_default):
    class2int = dict()
    int2class = dict()
    with open(join(root, 'classes.txt')) as f :
        lines = f.readlines()
        for l in lines:
            index , class_name = int(l.split('.')[0].strip()), l.strip()
            class2int[class_name] = index
            int2class[index] = class_name
        return class2int, int2class

# generate first 90% images in training_labels.txt
class trainset(Dataset):
    def __init__(self, transform=None, root=dataset_root_path_default):
        self.root = root
        self.transform=transform
        self.class2int, self.int2class = get_class_dicts(root)
        self.labels = dict()
        self.files = []
        with open(join(root,'training_labels.txt')) as f:
            for l in f.readlines():
                file , label = l.split(' ')[0].strip(), int(l.split(' ')[1].split('.')[0].strip())
                self.labels[file] = label
                self.files.append(file)
    def __len__(self):
        return int(len(self.files)*0.9)
    def __getitem__(self, idx):
        label = self.labels[self.files[idx]]
        img = Image.open(join(self.root,'training_images', self.files[idx]))
        if self.transform:
            img = self.transform(img)
        return img, label

# generate last 10% of the images in training_labels.txt
# This class is used as validation set in the code
class testset(Dataset):
    def __init__(self, transform=None, root=dataset_root_path_default):
        self.root = root
        self.transform=transform
        self.class2int, self.int2class = get_class_dicts(root)
        self.labels = dict()
        self.files = []
        with open(join(root,'training_labels.txt')) as f:
            for l in f.readlines():
                file , label = l.split(' ')[0].strip(), int(l.split(' ')[1].split('.')[0].strip())
                self.labels[file] = label
                self.files.append(file)
    def __len__(self):
        # this line is different from class trainset
        return len(self.files) - int(len(self.files)*0.9)
    def __getitem__(self, idx):
        # this line is different from class trainset
        idx = idx + int(len(self.files)*0.9)
        label = self.labels[self.files[idx]]
        img = Image.open(join(self.root,'training_images', self.files[idx]))
        if self.transform:
            img = self.transform(img)
        return img, label

# generate testing (images_name, image) pair in 'testing_img_order.txt'
# no ground truth label, evaluation only
class evalset(Dataset):
    def __init__(self, transform=None, root=dataset_root_path_default):
        self.root = root
        self.transform = transform
        self.class2int, self.int2class = get_class_dicts(root)
        self.files = []
        with open(join(root,'testing_img_order.txt')) as f:
            for file in f.readlines():
                self.files.append(file.strip())
    def __len__(self):
            return len(self.files)
    def __getitem__(self, index):
            img = Image.open(join(self.root,'testing_images',self.files[index]))
            if self.transform:
                img = self.transform(img)
            return self.files[index], img
