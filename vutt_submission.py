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


device = torch.device("cuda")
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
root = "/home/sangdt/research/voice/planet/Image-Colorizer-Pix2Pix/own_miss_data/A/test"
os.makedirs("vutt_submission", exist_ok=True)
seismics_dir = glob.glob(os.path.join(root,"*"))

for seismic_path in seismics_dir:
    print(seismic_path)
    short_path = ntpath.basename(seismic_path)
    name = os.path.splitext(short_path)[0]
    # image = Image.open(image_path).convert('RGB')
    seismic_data = np.load(seismic_path, allow_pickle=True)
    image_test = seismic_data[...,0]
    rows, cols = np.where(nan_mask)
    x, y, w, h = min(cols), min(rows), max(cols) - min(cols), max(rows) - min(rows)
    
    miss_seismic = []

    for i in range(len(seismic_data.shape[-1])):
        image = seismic_data[...,i]

        image = np.repeat(seismic_data[:, :, np.newaxis], 3, axis=2)
        image = Image.fromarray(image, 'RGB')

        image = transform(image)
        image = torch.unsqueeze(image, dim=0)
        image = image.to(device)
        print(image.shape)

        output = model.netG(image)
        out_image = util.tensor2im(output)

        miss_part = out_image[y:y+h, x:x+w,:]

        miss_seismic.append(miss_part)

    miss_seismic = np.stack(miss_seismic, axis=0)

    os.makedirs("./np_output", exist_ok=True)
    np_path = os.path.join("./np_output", f'{name}.npy')
    with open(np_path, 'wb') as f:
        np.save(f, miss_seismic)

