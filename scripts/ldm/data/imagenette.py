import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import requests as req
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset

import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light


def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())
    


class ImagenetteBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self.keep_orig_class_label = self.config.get("keep_orig_class_label", False)
        self.process_images = True  # if False we skip loading & processing images and self.data contains filepaths
        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_idx_to_synset()
        self._prepare_human_to_integer_label()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths):
        ignore = set([
            #"n06596364_9591.JPEG", #dont want to filter anythin?
        ])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
        if "sub_indices" in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
            self.synset2idx = synset2idx(path_to_yaml=self.idx2syn)
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    def _prepare_synset_to_human(self):
        SIZE = 2655750
        URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
        self.human_dict = os.path.join(self.root, "synset_human.txt")
        if (not os.path.exists(self.human_dict) or
                not os.path.getsize(self.human_dict)==SIZE):
            download(URL, self.human_dict)

    def _prepare_idx_to_synset(self):
        URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
        self.idx2syn = os.path.join(self.root, "index_synset.yaml")
        if (not os.path.exists(self.idx2syn)):
            download(URL, self.idx2syn)

    def _prepare_human_to_integer_label(self):
        URL = "https://heibox.uni-heidelberg.de/f/2362b797d5be43b883f6/?dl=1"
        self.human2integer = os.path.join(self.root, "imagenet1000_clsidx_to_labels.txt")
        if (not os.path.exists(self.human2integer)):
            download(URL, self.human2integer)
        with open(self.human2integer, "r") as f:
            lines = f.read().splitlines()
            assert len(lines) == 1000
            self.human2integer_dict = dict()
            for line in lines:
                value, key = line.split(":")
                self.human2integer_dict[key] = int(value)

    def _load(self):
        print("debug load")
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)
            #print(self.relpaths)
            self.relpaths = self._filter_relpaths(self.relpaths)
            #print(self.relpaths)
            print("Removed {} files from filelist during filtering.".format(l1 - len(self.relpaths)))

        self.synsets = [p.split("/")[0] for p in self.relpaths]
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        print(unique_synsets)
        
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        if not self.keep_orig_class_label:
            self.class_labels = [class_dict[s] for s in self.synsets]
        else:
            self.class_labels = [self.synset2idx[s] for s in self.synsets]
            
        #print(self.class_labels)

        with open(self.human_dict, "r") as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        self.human_labels = [human_dict[s] for s in self.synsets]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
        }

        if self.process_images:
            self.size = retrieve(self.config, "size", default=256)
            self.data = ImagePaths(self.abspaths,
                                   labels=labels,
                                   size=self.size,
                                   random_crop=self.random_crop,
                                   )
        else:
            self.data = self.abspaths


class ImagenetteTrain(ImagenetteBase):
    NAME = "imagenette_train"
    URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    FILES = ["imagenette2-320.tgz"]
    SIZES = [34247816]

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        super().__init__(**kwargs)

    def _prepare(self):
#        if self.data_root:
#            self.root = os.path.join(self.data_root, self.NAME)
#        else:
#            cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
#            self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)

        self.root = os.path.join("/oscar", "scratch", "aghandik","FourierImageDiffuse", "data", self.NAME)
        #self.root = os.path.join(os.path.expanduser("~/FourierImageDiffuse"), self.NAME)
        print("PRELIMINARY PATHHHHHH")
        print(self.root)
        
        self.datadir = os.path.join(self.root, "train")
        self.txt_filelist = os.path.join(self.root, "train_filelist.txt")
        self.expected_length = 9469  # Total number of training images in Imagenette
        self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",
                                    default=True)

        if not tdu.is_prepared(self.root):
            # prep
            
            print("GOING AGIANNNNNNNN")

            print(f"Preparing dataset {self.NAME} in {self.root}")
            paths_to_check = [self.root, self.datadir]
            for path in paths_to_check:
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)

            archive_path = os.path.join(self.root, self.FILES[0])
            if not os.path.exists(archive_path) or os.path.getsize(archive_path) != self.SIZES[0]:
                print(f"Downloading {self.URL} to {archive_path}")
                response = req.get(self.URL, stream=True)
                response.raise_for_status()
                with open(archive_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)

            with tarfile.open(archive_path, "r:gz") as tar:
                members = [m for m in tar.getmembers() if m.name.startswith("imagenette2-320/train")]
                tar.extractall(path=self.root, members=members)
                
            # Move extracted images to self.datadir
            extracted_train_dir = os.path.join(self.root, "imagenette2-320", "train")
            if os.path.exists(extracted_train_dir):
                print(f"Moving extracted files from {extracted_train_dir} to {self.datadir}")
                if not os.path.exists(self.datadir):
                    os.rename(extracted_train_dir, self.datadir)
                else:
                    # Handle case where `self.datadir` already exists (merge directories)
                    for item in os.listdir(extracted_train_dir):
                        src = os.path.join(extracted_train_dir, item)
                        dst = os.path.join(self.datadir, item)
                        if os.path.isdir(src):
                            shutil.move(src, dst)
                        else:
                            os.replace(src, dst)
                            
            #delete empty directory 
            parent_folder = os.path.join(self.root, "imagenette2-320")
            if os.path.exists(parent_folder):
                print(f"Removing empty folder: {parent_folder}")
                shutil.rmtree(parent_folder)
    
            # Write file paths to `train_filelist.txt`
            
            print(f"Writing file paths to {self.txt_filelist}")
            with open(self.txt_filelist, "w") as f:
                for dirpath, dirnames, filenames in os.walk(self.datadir):
                    for filename in filenames:
                        if filename.endswith(".JPEG"):  # Only include JPEG files
                            relative_path = os.path.relpath(os.path.join(dirpath, filename), start=self.datadir)
                            f.write(relative_path + "\n")
                            
        
            print(f"Dataset prepared: {self.datadir} and file list written to {self.txt_filelist}")

            tdu.mark_prepared(self.root)
            
        


class ImagenetteValidation(ImagenetteBase):
    NAME = "imagenette_val"
    URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    FILES = ["imagenette2-320.tgz"]
    SIZES = [34247816]

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.data_root = data_root
        self.process_images = process_images
        super().__init__(**kwargs)

    def _prepare(self):
#        if self.data_root:
#            self.root = os.path.join(self.data_root, self.NAME)
#        else:
#            cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
#            self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)

        self.root = os.path.join("/oscar", "scratch", "aghandik","FourierImageDiffuse", "data", self.NAME)

        self.datadir = os.path.join(self.root, "val")
        self.txt_filelist = os.path.join(self.root, "val_filelist.txt")
        self.expected_length = 3925  # Total number of validation images in Imagenette
        self.random_crop = retrieve(self.config, "ImageNetValidation/random_crop",
                                    default=False)

        if not tdu.is_prepared(self.root):
            # prep
            print(f"Preparing dataset {self.NAME} in {self.root}")
            paths_to_check = [self.root, self.datadir]
            for path in paths_to_check:
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)

            archive_path = os.path.join(self.root, self.FILES[0])
            if not os.path.exists(archive_path) or os.path.getsize(archive_path) != self.SIZES[0]:
                print(f"Downloading {self.URL} to {archive_path}")
                response = req.get(self.URL, stream=True)
                response.raise_for_status()
                with open(archive_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)

            
            with tarfile.open(archive_path, "r:gz") as tar:
                members = [m for m in tar.getmembers() if m.name.startswith("imagenette2-320/val")]
                tar.extractall(path=self.root, members=members)

            # Move extracted images to self.datadir
            extracted_train_dir = os.path.join(self.root, "imagenette2-320", "val")
            if os.path.exists(extracted_train_dir):
                print(f"Moving extracted files from {extracted_train_dir} to {self.datadir}")
                if not os.path.exists(self.datadir):
                    os.rename(extracted_train_dir, self.datadir)
                else:
                    # Handle case where `self.datadir` already exists (merge directories)
                    for item in os.listdir(extracted_train_dir):
                        src = os.path.join(extracted_train_dir, item)
                        dst = os.path.join(self.datadir, item)
                        if os.path.isdir(src):
                            shutil.move(src, dst)
                        else:
                            os.replace(src, dst)
                            
                            
            #delete empty directory 
            parent_folder = os.path.join(self.root, "imagenette2-320")
            if os.path.exists(parent_folder):
                print(f"Removing empty folder: {parent_folder}")
                shutil.rmtree(parent_folder)
    
            # Write file paths to `train_filelist.txt`
            print(f"Writing file paths to {self.txt_filelist}")
            with open(self.txt_filelist, "w") as f:
                for dirpath, dirnames, filenames in os.walk(self.datadir):
                    for filename in filenames:
                        if filename.endswith(".JPEG"):  # Only include JPEG files
                            relative_path = os.path.relpath(os.path.join(dirpath, filename), start=self.datadir)
                            f.write(relative_path + "\n")
        
            
            print(f"Dataset prepared: {self.datadir} and file list written to {self.txt_filelist}")
            
            tdu.mark_prepared(self.root)



class ImageNetSR(Dataset):
    def __init__(self, size=None,
                 degradation=None, downscale_f=4, min_crop_f=0.5, max_crop_f=1.,
                 random_crop=True):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        self.base = self.get_base()
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

        else:
            interpolation_fn = {
            "cv_nearest": cv2.INTER_NEAREST,
            "cv_bilinear": cv2.INTER_LINEAR,
            "cv_bicubic": cv2.INTER_CUBIC,
            "cv_area": cv2.INTER_AREA,
            "cv_lanczos": cv2.INTER_LANCZOS4,
            "pil_nearest": PIL.Image.NEAREST,
            "pil_bilinear": PIL.Image.BILINEAR,
            "pil_bicubic": PIL.Image.BICUBIC,
            "pil_box": PIL.Image.BOX,
            "pil_hamming": PIL.Image.HAMMING,
            "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn)

            else:
                self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                          interpolation=interpolation_fn)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example = self.base[i]
        image = Image.open(example["file_path_"])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)

        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)

        else:
            self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)

        image = self.cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]

        if self.pil_interpolation:
            image_pil = PIL.Image.fromarray(image)
            LR_image = self.degradation_process(image_pil)
            LR_image = np.array(LR_image).astype(np.uint8)

        else:
            LR_image = self.degradation_process(image=image)["image"]

        example["image"] = (image/127.5 - 1.0).astype(np.float32)
        example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32)

        return example


class ImageNetSRTrain(ImageNetSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        with open("data/imagenet_train_hr_indices.p", "rb") as f:
            indices = pickle.load(f)
        dset = ImageNetTrain(process_images=False,)
        return Subset(dset, indices)


class ImageNetSRValidation(ImageNetSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        with open("data/imagenet_val_hr_indices.p", "rb") as f:
            indices = pickle.load(f)
        dset = ImageNetValidation(process_images=False,)
        return Subset(dset, indices)
