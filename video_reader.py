import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import zipfile
import io
import numpy as np
import random
import re
import pickle
from glob import glob
import json

from videotransforms.tensor_transforms import GroupNormalize
from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip, CenterCrop, TenCrop
from videotransforms.volume_transforms import ClipToTensor


class Split():
    def __init__(self):
        self.gt_a_list = []
        self.videos = []
    
    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)
        
        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def get_max_video_len(self):
        max_len = 0
        for v in self.videos:
            l = len(v)
            if l > max_len:
                max_len = l

        return max_len

    def __len__(self):
        return len(self.gt_a_list)

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.get_item_counter = 0

        self.data_dir = args.path
        self.classInd = args.classInd
        self.vid2cls = {}
        self.seq_len = args.seq_len
        self.train = True
        self.tensor_transform = transforms.ToTensor()
        self.norm = None
        self.img_size = args.img_size
        self.img_norm = args.img_norm

        self.annotation_path = args.traintestlist
        #self.dataname = args.traintestlist.split()

        self.single_img = False

        self.way=args.way
        self.shot=args.shot
        self.query_per_class=args.query_per_class

        self.train_split = Split()
        self.test_split = Split()

        self.setup_transforms()

        self._select_fold()
        
        if self.classInd:
            classInd = open(self.classInd, 'r')
            js = classInd.read()
            self.vid2cls = json.loads(js)
            classInd.close()
        self.read_dir()


    def setup_transforms(self):
        if self.single_img:
            self.transform = {}
            trsfm = transforms.Compose([
                            transforms.Resize((84,84)),
                            transforms.ToTensor(),
                        ])

            self.transform['train'] = trsfm
            self.transform['test'] = trsfm

        else:
            video_transform_list = []
            video_test_list = []
            
            if self.img_size == 84:
                video_transform_list.append(Resize(96))
                video_test_list.append(Resize(96))
            elif self.img_size == 224:
                video_transform_list.append(Resize(256))
                video_test_list.append(Resize(256))


            
            if self.args.dataset != 'ssv2' and self.args.dataset != 'ssv2_cmn':
                #print('Add Flip augmentation')
                video_transform_list.append(RandomHorizontalFlip())
            else:
                print('Remove Flip augmentation for SSv2 dataset')
            video_transform_list.append(RandomCrop(self.img_size))
            


            video_test_list.append(CenterCrop(self.img_size))

            # if self.args.dataset == 'kinetics':
            #     print('Add data normalization for Kinetics')
            #     self.norm = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            self.transform = {}
            self.transform["train"] = Compose(video_transform_list)
            self.transform["test"] = Compose(video_test_list)
            #

    def read_dir(self):
                
        class_folders = os.listdir(self.data_dir)
        class_folders.sort()
        self.class_folders = class_folders
        if self.args.dataset == 'ssv2' or self.args.dataset == 'ssv2_cmn':
            for video_folder in class_folders:
                #print(video_folder)
                c = self.get_train_or_test_db(video_folder)
                if c == None:
                    continue
                imgs = os.listdir(os.path.join(self.data_dir, video_folder))
                if len(imgs) < self.seq_len:
                    continue            
                imgs.sort()
                #print(len(imgs))
                paths = [os.path.join(self.data_dir, video_folder, img) for img in imgs]
                paths.sort()
                if self.classInd:
                    class_id = self.vid2cls[video_folder]
                else:
                    class_id =  class_folders.index(video_folder)
                c.add_vid(paths, class_id)
        else:
            for class_folder in class_folders:
                video_folders = os.listdir(os.path.join(self.data_dir, class_folder))
                #print('len', len(video_folders))
                video_folders.sort()
                #print('video folders', video_folders)
                if self.args.debug_loader:
                    video_folders = video_folders[0:1]
                for video_folder in video_folders:
                    #print(video_folder)
                    c = self.get_train_or_test_db(video_folder)
                    if c == None:
                        continue
                    imgs = os.listdir(os.path.join(self.data_dir, class_folder, video_folder))
                    #print(imgs)
                    #print(imgs)
                    if len(imgs) < self.seq_len:
                        continue            
                    imgs.sort()
                    paths = [os.path.join(self.data_dir, class_folder, video_folder, img) for img in imgs]
                    paths.sort()
                    class_id =  class_folders.index(class_folder)
                    c.add_vid(paths, class_id)
        print("loaded {}".format(self.data_dir))
        print("train: {}, test: {}".format(len(self.train_split), len(self.test_split)))

    def get_train_or_test_db(self, split=None): 
        if split is None:
            get_train_split = self.train
        else:
            if split in self.train_test_lists["train"] or split.lower() in self.train_test_lists["train"]:
                get_train_split = True
            elif split in self.train_test_lists["test"] or split.lower() in self.train_test_lists["test"]:
                #print(split)
                get_train_split = False
            else:
                #print(split)
                return None
        if get_train_split:
            return self.train_split
        else:
            return self.test_split
            #return self.test_split
   
    def _select_fold(self):
        lists = {}
        for name in ["train", "test"]:
            fname = "{}list{:02d}.txt".format(name, self.args.split)
            f = os.path.join(self.annotation_path, fname)
            selected_files = []
            with open(f, "r") as fid:
                data = fid.readlines()
                #print(data)
                data = [x.replace(' ', '_').lower() for x in data] #注意这里把所有视频名里的字母大写都转换为小写了
                data = [x.strip().split(" ")[0] for x in data] # class_name/classID
                data = [os.path.splitext(os.path.split(x)[1])[0] for x in data] # classID
                # print(data)
                # if "kinetics" in self.args.path:
                #    data = [x[0:11] for x in data] 

                selected_files.extend(data)
            lists[name] = selected_files
        self.train_test_lists = lists
        #print(len(self.train_test_lists['train']))
        #print(self.train_test_lists['test'])

    def __len__(self):
        c = self.get_train_or_test_db()
        return 1000000
        #return len(c)
   
    def get_split_class_list(self):
        c = self.get_train_or_test_db()
        classes = list(set(c.gt_a_list))
        classes.sort()
        return classes
    
    def read_single_image(self, path):
        with Image.open(path) as i:
            i.load()
            return i
    
    
    def get_seq(self, label, idx=-1):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx) 
        n_frames = len(paths)
        if self.train:
            excess_frames = n_frames - self.seq_len
            excess_pad = int(min(5, excess_frames / 2))
            if excess_pad < 1:
                start = 0
                end = n_frames - 1
            else:
                start = random.randint(0, excess_pad)
                end = random.randint(n_frames-1 -excess_pad, n_frames-1)
        else:
            start = 1
            end = n_frames - 2

        if end - start < self.seq_len:
            end = n_frames - 1
            start = 0
        else:
            pass

        idx_f = np.linspace(start, end, num=self.seq_len)
        idxs = [int(f) for f in idx_f]
        
        if self.seq_len == 1:
            idxs = [random.randint(start, end-1)]
        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]
            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            imgs = torch.stack(imgs)
            if self.norm:
                imgs = self.norm(imgs)
        return imgs, vid_id

    def get_single_img(self, index):
        c = self.get_train_or_test_db().cursor()
        try:
            path, label = c.execute("SELECT path, gt_a FROM frames WHERE id=?", (int(index+1), )).fetchall()[0]
            img = self.read_single_image(path)
        except:
            print(index)
            exit(1)
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]
            img = transform(img)
            return img, label

    """returns dict of support and target images and labels"""
    def __getitem__(self, index):

        if self.single_img:
            return self.get_single_img(index)

        c = self.get_train_or_test_db()
        classes = c.get_unique_classes() # train or test set上的所有类
        batch_classes = random.sample(classes, self.way) # 随机挑选way类
 

        if self.train:
            n_queries = self.args.query_per_class
        else:
            n_queries = self.args.query_per_class_test


        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []

        for bl, bc in enumerate(batch_classes):
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)
            #print('n_total', n_total)

            for idx in idxs[0:self.args.shot]:
                vid, vid_id = self.get_seq(bc, idx)
                support_set.append(vid)
                support_labels.append(bl)
            for idx in idxs[self.args.shot:]:
                vid, vid_id = self.get_seq(bc, idx)
                target_set.append(vid)
                target_labels.append(bl)
                real_target_labels.append(bc)
        
        s = list(zip(support_set, support_labels))
        #random.shuffle(s)
        support_set, support_labels = zip(*s)
        
        t = list(zip(target_set, target_labels, real_target_labels))
        random.shuffle(t)
        target_set, target_labels, real_target_labels = zip(*t)
        
        
        support_set = torch.cat(support_set)
        target_set = torch.cat(target_set)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes) 
        
        return {"support_set":support_set, "support_labels":support_labels, "target_set":target_set, \
                "target_labels":target_labels, "real_target_labels":real_target_labels, "batch_class_list": batch_classes}


if __name__ == '__main__':
    class ArgsObject(object):
        def __init__(self):
            # self.trans_linear_in_dim = 512
            # self.trans_linear_out_dim = 128
            self.dataset = 'hmdb'
            #self.traintestlist = 'splits/ssv2_CMN/'
            #self.path = '/mnt/data/sjtu/ssv2/frames'
            self.traintestlist = 'splits/hmdb_ARN/'
            self.path = '/home/sjtu/data/HMDB51/jpg'
            # self.traintestlist = 'splits/kinetics_CMN/'
            # self.path = '/home/sjtu/data/kinetics-FSL/'
            self.split = 3
            self.classInd = None

            self.way = 5
            self.shot = 1
            self.query_per_class = 5
            self.query_per_class_test = 5
            self.trans_dropout = 0.1
            self.seq_len = 8 
            self.img_size = 84
            self.method = "resnet18"
            self.num_gpus = 1
            self.temp_set = [2,3]    
            self.debug_loader = False
            self.img_norm = None

    args = ArgsObject()
    dataloader = VideoDataset(args)
    dataloader = iter(dataloader)
    data = next(dataloader)