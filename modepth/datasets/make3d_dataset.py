import os
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from torch.utils import data

# from path_my import Path
from scipy.io import loadmat

# ref https://github.com/nianticlabs/monodepth2/issues/392

def get_input_depth_make3d(path):
    m = loadmat(path)
    pos3dgrid = m['Position3DGrid']
    depth = pos3dgrid[:, :, 3]
    return depth

def get_input_img(path, color=True):
    """Read the image in KITTI."""
    img = Image.open(path)
    if color:
        img = img.convert('RGB')
    return img


class Make3DDataset(data.Dataset):
    DATA_NAME_DICT = {
        'color': ('Test134', 'img-', 'jpg'),
        'depth': ('Gridlaserdata', 'depth_sph_corr-', 'mat')
    }

    def __init__(self,
                 dataset_mode,
                 split_file,
                 normalize_params=[0.411, 0.432, 0.45],
                 use_godard_crop=True,
                 full_size=None,
                 resize_before_crop=False
                 ):
        super().__init__()
        self.init_opts = locals()
        self.dataset_mode = dataset_mode
        self.dataset_dir = Path.get_path_of('make3d')
        self.split_file = split_file
        self.use_godard_crop = use_godard_crop
        self.full_size = full_size
        self.resize_before_crop = resize_before_crop

        self.file_list = self._get_file_list(split_file)

        # Initializate transforms
        self.to_tensor = tf.ToTensor()
        self.normalize = tf.Normalize(mean=normalize_params, std=[1, 1, 1])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, f_idx):

        file_info = self.file_list[f_idx]
        base_path = os.path.join(self.dataset_dir, '{}',
                                 '{}' + file_info + '.{}')
        inputs = {}
        color_l_path = base_path.format(*self.DATA_NAME_DICT['color'])
        inputs['color_s_raw'] = get_input_img(color_l_path)

        depth_path = base_path.format(*self.DATA_NAME_DICT['depth'])
        inputs['depth'] = get_input_depth_make3d(depth_path)

        for key in list(inputs):
            if 'color' in key:
                raw_img = inputs[key]
                if self.resize_before_crop:
                    self.color_resize = tf.Resize(self.full_size,
                                                  interpolation=Image.ANTIALIAS)

                    img = self.to_tensor(self.color_resize(raw_img))
                    if self.use_godard_crop:
                        top = int((self.full_size[0] - self.full_size[1] / 2) / 2) + 1
                        bottom = int((self.full_size[0] + self.full_size[1] / 2) / 2) + 1
                        img = img[:, top:bottom,:]
                    inputs[key.replace('_raw', '')] =\
                        self.normalize(img)
                else:
                    if self.use_godard_crop:
                        raw_img = raw_img.crop((0, 710, 1704, 1562))
                    img = self.to_tensor(raw_img)
                    if self.full_size is not None:
                        # for outputting the same image with cv2
                        img = img.unsqueeze(0)
                        img = F.interpolate(img, self.full_size, mode='nearest')
                        img = img.squeeze(0)
                    inputs[key.replace('_raw', '')] =\
                        self.normalize(img)

            elif 'depth' in key:
                raw_depth = inputs[key]
                if self.use_godard_crop:
                    raw_depth = raw_depth[17:38, :]
                depth = torch.from_numpy(raw_depth.copy()).unsqueeze(0)
                inputs[key] = depth

        # delete raw data
        inputs.pop('color_s_raw')
        inputs['file_info'] = [file_info]
        return inputs

    def _get_file_list(self, split_file):
        with open(split_file, 'r') as f:
            files = f.readlines()
            filenames = []
            for f in files:
                file_name = f.replace('\n', '')
                filenames.append(file_name)
        return filenames
    
    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path
    
    def dataset_info(self):
        infos = []
        infos.append('    -{} Datasets'.format(self.dataset_mode))
        infos.append('      get {} of data'.format(len(self)))
        for key, val in self.init_opts.items():
            if key not in ['self', '__class__']:
                infos.append('        {}: {}'.format(key, val))
        return infos