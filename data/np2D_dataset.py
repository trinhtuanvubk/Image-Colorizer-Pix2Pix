import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class Np2D_Seismic(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.seismic_dir = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.seismic_2D_paths = sorted(make_dataset(self.seismic_dir, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image



    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        """
        percentile = 25
        # apply the same transform to both src and tgt
        transform = torchvision_T.Compose([torchvision_T.Resize([256,256], 
            torchvision_T.InterpolationMode.BICUBIC),
            torchvision_T.ToTensor(),
            torchvision_T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # read a image given a random integer index
        image_path = self.seismic_2D_paths[index]

        seismic = np.load(image_path, allow_pickle=True)
        src_seismic = seismic.copy()
        # rescale volume
        # minval = np.percentile(seismic, 2)
        # maxval = np.percentile(seismic, 98)
        # seismic = np.clip(seismic, minval, maxval)
        axis = np.random.choice(['i_line', 'x_line'], 1)[0]

        if axis == 'i_line':
            sample_size = np.round(seismic.shape[0]*(percentile/100)).astype('int')
            if sample_start is None:
                sample_start = np.random.choice([5, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270], 1)[0]
            sample_end = sample_start+sample_size

            target_mask = np.zeros(seismic.shape).astype('bool')
            target_mask[sample_start:sample_end] = True

            # target = seismic[sample_start:sample_end, :, :].copy()
            seismic[target_mask] = np.nan
        
        else:
            sample_size = np.round(seismic.shape[1]*(percentile/100)).astype('int')
            if sample_start is None:
            sample_start = np.random.choice([5, 15, 30, 45], 1)[0]
            sample_end = sample_start+sample_size

            target_mask = np.zeros(seismic.shape).astype('bool')
            target_mask[:, sample_start:sample_end] = True

            # target = seismic[:, sample_start:sample_end].copy()
            seismic[target_mask] = np.nan

        seismic = np.expand_dims(seismic, axis=2)
        seismic = Image.fromarray(np.uint8(seismic), 'L')
        seismic = transform(seismic)

        src_seismic = np.expand_dims(src_seismic, axis=2)
        src_seismic = Image.fromarray(np.uint8(src_seismic), 'L')
        src_seismic = transform(src_seismic)

        return seismic, src_seismic

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.seismic_2D_paths)
