import os
import torch
from PIL import Image
import glob
import torchvision.transforms as torchvision_T
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
import ntpath

from util import util

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
root = "/home/aittgp/vutt/workspace/patch_planet/Image-Colorizer-Pix2Pix/own_miss_data/A/test"
os.makedirs("vutt_result", exist_ok=True)
images_dir = glob.glob(os.path.join(root,"*"))

for image_path in images_dir:
    print(image_path)
    short_path = ntpath.basename(image_path)
    name = os.path.splitext(short_path)[0]
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = torch.unsqueeze(image, dim=0)
    image = image.to(device)
    print(image.shape)
    output = model.netG(image)
    
    out_image = util.tensor2im(output)

    save_path = os.path.join("./vutt_result", f'{name}.jpg')
    print(save_path)
    util.save_image(out_image, save_path, aspect_ratio=opt.aspect_ratio)

   