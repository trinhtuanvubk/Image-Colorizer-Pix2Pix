from models.pix2pix_model import Pix2PixModel

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from data.base_dataset import *
from PIL import Image
import torch
from util.util import * 

opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
opt.preprocess = "crop_and_resize"
opt.name = "facadeedxts_pix2pix"
opt.model = "pix2pix"
# dataset = create_dataset(opt) 

model = Pix2PixModel(opt)
# model.set_input(input)
model.setup(opt)
# model.eval()

def OnlyImageLoader(opt, image_path):
    if image_path is None: 
        image_path = opt.image_path 

    img = Image.open(image_path).convert('RGB')
    transform_params = get_params(opt, img.size)
    transforms = get_transform(opt, transform_params, grayscale=(opt.input_nc == 1) )
    img = transforms(img)

    return img

image_path = "./own_data/A/test/otomegesekai_ch72_07-2.png"
img = OnlyImageLoader(opt, image_path)
img = torch.unsqueeze(img, 0)
print(img)
print(img.shape)
    
fake = model.netG(img.to("cpu"))


_fake = tensor2im(fake)
print(fake.shape)
save_image(_fake, "fake.png")


_real = tensor2im(img)
save_image(_real, "real.png")
    