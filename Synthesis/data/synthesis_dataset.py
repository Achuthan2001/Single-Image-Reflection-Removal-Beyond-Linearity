import os.path
from data.image_folder import make_dataset
import torch.utils.data as data
from PIL import Image
import random
import torchvision.transforms as transforms
import cv2
import numpy as np


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

class SynthesisDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.loadSize = opt.loadSize
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.B_paths = make_dataset(self.dir_B)            

        self.B_paths = sorted(self.B_paths)
            
        self.B_size = len(self.B_paths)

        if opt.phase == 'test':
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
            self.A_paths = make_dataset(self.dir_A)
            self.A_paths = sorted(self.A_paths)
            self.A_size = len(self.A_paths)

    def get_transform(self, img):
        width = img.size[0]
        height = img.size[1]

        transform_list = []
        if width < height:
            transform_list.append(transforms.RandomCrop((width, width)))
        else:
            transform_list.append(transforms.RandomCrop((height, height)))
        transform_list.append(transforms.Resize((self.loadSize, self.loadSize)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),
                                                   (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)
        img = transform(img)

        return img

    def __getitem__(self, index):
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]            

        B_img = Image.open(B_path).convert('RGB')
        B_img_origin = B_img

        reflection_type = 'defocused'
        if self.opt.phase == 'test':
            reflection_type = self.opt.type

        # Defocused reflection
        if reflection_type == 'defocused':
            B_img = np.asarray(B_img)
            k_sz = np.linspace(5,10,80)
            sigma = k_sz[np.random.randint(0, len(k_sz))]
            sz = int(2*np.ceil(2*sigma)+1)
            B_img = cv2.GaussianBlur(B_img,(sz,sz),sigma,sigma,0)
            B_img = Image.fromarray(B_img.astype(np.uint8))

        B = self.get_transform(B_img)
        B_origin = self.get_transform(B_img_origin)

        if self.opt.phase == 'test':
            index_A = random.randint(0, self.A_size - 1)
            A_path = self.A_paths[index_A]
            A_img = Image.open(A_path).convert('RGB')
            A = self.get_transform(A_img)

        if self.opt.phase == 'train':
            return {'B': B, 'B_origin': B_origin, 'B_paths': B_path}
        else:
            return {'A': A, 'A_origin': A_origin, 'B': B,
                    'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.B_size

    def name(self):
        return 'SynthesisDataset'
