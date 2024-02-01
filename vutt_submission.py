import os
import torch
from PIL import Image
import glob
import torchvision.transforms as torchvision_T
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
import ntpath
import numpy as np
from util import util
import cv2
from tqdm import tqdm

# Option
opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1  
opt.phase = 'test'
opt.isTrain = False
opt.model="pix2pix"


device = torch.device("cpu")
# Preprocess
transform = torchvision_T.Compose([torchvision_T.Resize([256,256], torchvision_T.InterpolationMode.BICUBIC),
             torchvision_T.ToTensor(),
             torchvision_T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# MODEL PROCESS
model = Pix2PixModel(opt)
print(model)
print(opt)
model.setup(opt)
if opt.eval:
    model.eval()

# INPUT IMAGE
root = "/home/aittgp/vutt/workspace/patch_planet/Image-Colorizer-Pix2Pix/test_data"
# root = "/home/aittgp/vutt/workspace/patch_planet/patch_the_planet_starter_ntbk/example_data"
os.makedirs("vutt_submission", exist_ok=True)
seismics_dir = glob.glob(os.path.join(root,"*"))

for seismic_path in tqdm(seismics_dir):
    # print(seismic_path)
    short_path = ntpath.basename(seismic_path)
    name = os.path.splitext(short_path)[0]
    # image = Image.open(image_path).convert('RGB')
    seismic_data = np.load(seismic_path, allow_pickle=True)
    # minval = np.percentile(seismic_data, 2)
    # maxval = np.percentile(seismic_data, 98)
    # seismic_data = np.clip(seismic_data, minval, maxval)
    # seismic_data = ((seismic_data - minval) / (maxval - minval)) * 255
    image_test = seismic_data[...,500]
    nan_mask = np.isnan(image_test)
    rows, cols = np.where(nan_mask)
    x, y, w, h = min(cols), min(rows), max(cols) - min(cols), max(rows) - min(rows)
    # print(x,y,w,h)
    
    miss_seismic = []

    for i in range(seismic_data.shape[-1]):
    # for i in range(500,550):
        image = seismic_data[...,i]
        cv2.imwrite("cv_1.jpg", image)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        cv2.imwrite("cv_.jpg", image)
        # print(image.shape)
        image = Image.fromarray(np.uint8(image), 'RGB')
        image.save("pil_.jpg")

        image = transform(image)
        image = torch.unsqueeze(image, dim=0)
        image = image.to(device)
        # print(image.shape)

        output = model.netG(image)
        out_image = util.tensor2im(output)
        cv2.imwrite("cv_all.jpg", out_image)
        # out_image = out_image[y:y+h, x:x+w,:]
        # cv2.imwrite("cv_miss.jpg", out_image)
        # print(out_image.shape)
        image_pil = Image.fromarray(out_image)
        image_pil = image_pil.resize((300, 300), Image.BICUBIC)
        image_pil = image_pil.convert('L')
        image_pil = image_pil.crop((x, y, x+w+1, y+h+1))
        image_pil.save("cv_miss.jpg")

        miss_seismic.append(image_pil)


    miss_seismic = np.stack(miss_seismic, axis=0)
    # print(miss_seismic.shape)
    permute_seismic = np.transpose(miss_seismic, (2,0,1))
    print(permute_seismic.shape)
    os.makedirs("./np_output", exist_ok=True)
    np_path = os.path.join("./np_output", f'{name}.npy')
    with open(np_path, 'wb') as f:
        np.save(f, permute_seismic)
        
hihi = np.load("/home/aittgp/vutt/workspace/patch_planet/Image-Colorizer-Pix2Pix/np_output/seismic-92551497.npy")
print(hihi.shape)

