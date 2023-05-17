import os, torch, cv2, shutil
import numpy as np
import skimage.color as sc
from datetime import datetime
from PIL import Image


def save_img(x, colors=3, value_range=255):
    if colors == 3:
        x = x.mul(value_range).clamp(0, value_range).round()
        x = x.numpy().astype(np.uint8)
        x = x.transpose((1, 2, 0))
        x = Image.fromarray(x)
    elif colors == 1:
        x = x[0, :, :].mul(value_range).clamp(0, value_range).round().numpy().astype(np.uint8)
        x = Image.fromarray(x).convert('L')
    return x


def crop_center(img, croph, cropw):
    h, w, c = img.shape

    if h < croph:
        img = cv2.copyMakeBorder(img, int(np.ceil((croph - h)/2)), int(np.ceil((croph - h)/2)), 0, 0, cv2.BORDER_DEFAULT)
    if w < cropw:
        img = cv2.copyMakeBorder(img, 0, 0, int(np.ceil((cropw - w)/2)), int(np.ceil((cropw - w)/2)), cv2.BORDER_DEFAULT)
    h, w, c = img.shape

    starth = h//2-(croph//2)
    startw = w//2-(cropw//2)
    return img[starth:starth+croph, startw:startw+cropw, :]


def quantize(img, rgb_range):
    return img.mul(rgb_range).clamp(0, rgb_range).round().div(rgb_range)


def rgb2ycbcrT(rgb):
    rgb = rgb.numpy().transpose(1, 2, 0) / 255
    yCbCr = sc.rgb2ycbcr(rgb)
    return torch.Tensor(yCbCr[:, :, 0])


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = 255*img1.astype(np.float64)
    img2 = 255*img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_SSIM_Y(input, target, rgb_range, shave):
    '''calculate SSIM
    the same outputs as MATLAB's
    '''

    c, h, w = input.size()
    input = quantize(input, rgb_range)
    target = quantize(target[:, 0:h, 0:w], rgb_range)
    if c > 1:
        input = rgb2ycbcrT(input).view(1, h, w)
        target = rgb2ycbcrT(target).view(1, h, w)
    input = input[0, shave:(h - shave), shave:(w - shave)]
    target = target[0, shave:(h - shave), shave:(w - shave)]
    return ssim(input.numpy(), target.numpy())


def calc_PSNR_Y(input, target, rgb_range, shave):
    c, h, w = input.size()
    input = quantize(input, rgb_range)
    target = quantize(target[:, 0:h, 0:w], rgb_range)
    if c > 1:
        input_Y = rgb2ycbcrT(input)
        target_Y = rgb2ycbcrT(target)
        diff = (input_Y - target_Y).view(1, h, w)
    else:
        diff = input - target
    diff = diff[:, shave:(h - shave), shave:(w - shave)]
    mse = diff.pow(2).mean()
    psnr = -10 * np.log10(mse)

    return psnr.data.numpy()


def calc_SSIM(input, target, rgb_range, shave):
    '''calculate SSIM
    the same outputs as MATLAB's
    '''

    c, h, w = input.shape
    input = quantize(input, rgb_range)
    target = quantize(target[:, 0:h, 0:w], rgb_range)
    input = input[:, shave:(h - shave), shave:(w - shave)]
    target = target[:, shave:(h - shave), shave:(w - shave)]
    ssim_value = 0
    for i in range(c):
        ssim_value += ssim(input[i, :, :].numpy(), target[i, :, :].numpy())
    return ssim_value / c


def calc_PSNR(input, target, rgb_range, shave):
    c, h, w = input.shape
    input = quantize(input, rgb_range)
    target = quantize(target[:, 0:h, 0:w], rgb_range)
    diff = input - target
    diff = diff[:, shave:(h - shave), shave:(w - shave)]
    mse = diff.pow(2).mean()
    psnr = -10 * np.log10(mse)

    return psnr.data.numpy()



def print_args(args):
    if args.train.lower() == 'train':
        args.save_img = False

        args.model_path = 'models/MSID_X{}'.format(args.scale) + datetime.now().strftime("_%Y%m%d_%H%M%S")

        args.resume = args.model_path + '/Checkpoints/checkpoint_epoch_0.pth'

        if not os.path.exists(args.model_path + '/Checkpoints/'):
            os.makedirs(args.model_path + '/Checkpoints')

        print(args)

    elif args.train.lower() == 'test':
        args.save_img = True

        args.model_path = 'models/MSID_X{}'.format(args.scale)
        args.resume = 'models/MSID_x{}.pth'.format(args.scale)
        # args.resume = 'models/MSID_X4_20230517_165608/Checkpoints/checkpoint_epoch_2.pth'


    return args
