import os, time, torch
import skimage.color as sc
import imageio
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import utils
from tqdm import tqdm
import data.common as common


def train(training_dataloader, optimizer, model, epoch, writer, args):
    criterion = nn.L1Loss(size_average=False).to(args.device)

    model.train()
    torch.cuda.empty_cache()

    with tqdm(total=len(training_dataloader), ncols=224) as pbar:
        for iteration, (LR_img, HR_img) in enumerate(training_dataloader):

            LR_img = Variable(LR_img).to(args.device)
            HR_img = Variable(HR_img).to(args.device)

            SR_img = model(LR_img)
            loss = criterion(SR_img, HR_img)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            time.sleep(0.1)
            pbar.update(1)
            pbar.set_postfix(Epoch=epoch,
                             LeaRate='{:.3e}'.format(optimizer.param_groups[0]['lr']),
                             Loss='{:.4f}'.format(loss))

            niter = (epoch - 1) * len(training_dataloader) + iteration
            
            if (niter + 1) % 200 == 0:
                writer.add_scalar('Train-Loss', loss, niter)

    torch.cuda.empty_cache()


def test(source_path, result_path, model, args, f_csv=None):
    model.eval()
    count = 0
    Avg_PSNR = 0
    Avg_SSIM = 0
    Avg_Time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    filename = os.listdir(source_path)
    filename.sort()
    val_length = len(filename)
    torch.cuda.empty_cache()

    with torch.no_grad():
        with tqdm(total=val_length, ncols=224) as pbar:
            for idx_img in range(val_length):
                img_name = filename[idx_img]
                HR_img = imageio.imread(os.path.join(source_path, img_name))
                img_name, ext = os.path.splitext(img_name)
                source_path_LQ = source_path.split('HR')[0] + 'LR_bicubic/X{}'.format(args.scale)
                LR_img = imageio.imread(os.path.join(source_path_LQ, img_name + '.png'))

                HR_img = common.set_channel(HR_img, args.n_colors)
                HR_img = common.np2Tensor(HR_img, args.value_range)
                LR_img = common.set_channel(LR_img, args.n_colors)
                LR_img = common.np2Tensor(LR_img, args.value_range)

                # c, h, w = HR_img.shape
                LR_img = Variable(LR_img[None]).to(args.device)
                H, W = HR_img.shape[1:]
                HR_img = HR_img[:, :(H - H % args.scale), :(W - W % args.scale)]

                start.record()
                SR_img = model(LR_img)
                end.record()
                torch.cuda.synchronize()
                Time = start.elapsed_time(end)
                Avg_Time += Time
                count += 1

                SR_img = SR_img.data[0].cpu().clamp(0, 1)

                PSNR = utils.calc_PSNR_Y(SR_img, HR_img, rgb_range=args.value_range, shave=args.scale)
                Avg_PSNR += PSNR

                if f_csv:
                    SSIM = utils.calc_SSIM_Y(SR_img, HR_img, rgb_range=args.value_range, shave=args.scale)
                    Avg_SSIM += SSIM
                    f_csv.writerow([img_name, PSNR, SSIM, Time])

                if args.save_img:
                    SR_img = utils.save_img(SR_img, 3, 255)
                    SR_img.save(result_path + '/{}.png'.format(img_name))

                time.sleep(0.1)
                pbar.update(1)
                pbar.set_postfix(PSNR='{:.2f}dB'.format(Avg_PSNR / count),
                                 SSIM='{:.4f}'.format(Avg_SSIM / count),
                                 TIME='{:.1f}ms'.format(Avg_Time / count),
                                 )
    torch.cuda.empty_cache()

    if f_csv:
        f_csv.writerow(['Avg', Avg_PSNR / count, Avg_SSIM / count, Avg_Time / count])

    return Avg_PSNR / count, Avg_SSIM / count, Avg_Time / count

