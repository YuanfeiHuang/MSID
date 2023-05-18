import shutil, csv, os
from torch.utils.data import DataLoader
import data.dataloaders as Dataloaders
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from option import args
import utils
import train
from src.MSID_arch import MSID


def main():
    global opt
    opt = utils.print_args(args)
    if torch.cuda.is_available() and opt.cuda:
        opt.device = torch.device('cuda:{}'.format(opt.GPU_ID))
        torch.cuda.manual_seed(opt.seed)
    else:
        opt.device = torch.device('cpu')
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print("===> Building model")
    model = MSID(upscale=opt.scale)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).to(opt.device)
        para = filter(lambda x: x.requires_grad, model.module.parameters())
    else:
        model = model.to(opt.device)
        para = filter(lambda x: x.requires_grad, model.parameters())

    from thop import profile
    sz_H, sz_W = 320, 180
    input = torch.FloatTensor(1, opt.n_colors, sz_H, sz_W).to(opt.device)
    FLOPs, Params = profile(model, inputs=(input, ), verbose=False)
    print('-------------Complexity-------------')
    print('\tParam = {:.3f}K\n\tFLOPs = {:.3f}G on {}'.format(Params * 1e-3, FLOPs * 1e-9, input.shape))
    torch.cuda.empty_cache()

    if os.path.isfile(opt.resume):
        print('===> Loading Checkpoint from {}'.format(opt.resume))
        ckp = torch.load(opt.resume)['params']
        model.load_state_dict(ckp, strict=False)
    else:
        print('===> No Checkpoint in {}'.format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = opt.optimizer([{'params': para, 'initial_lr': opt.lr}],
                              lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          last_epoch=opt.start_epoch,
                                          step_size=opt.lr_gamma_1,
                                          gamma=opt.lr_gamma_2)

    if opt.train.lower() == 'train':
        model.train()
        if os.path.exists(opt.model_path + '/' + 'runs'):
            shutil.rmtree(opt.model_path + '/' + 'runs')
        writer = SummaryWriter(opt.model_path + '/runs')

        start_epoch = opt.start_epoch if opt.start_epoch >= 0 else 0

        for data_test in opt.data_test:
            print('===> Testing on {}'.format(data_test))
            result_path = opt.model_path + '/Results/{}'.format(data_test)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            PSNR, SSIM, Time = train.test(opt.dir_data + 'Test/{}/HR'.format(data_test), result_path, model, opt)
            writer.add_scalar('PSNR_{}/'.format(data_test), PSNR, 0)
            writer.add_scalar('SSIM_{}/'.format(data_test), SSIM, 0)

        print('===> Building Training dataloader on {}'.format(opt.data_train))
        trainset = Dataloaders.dataloader(opt)
        train_dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                      num_workers=opt.threads, pin_memory=False)

        for epoch in range(start_epoch + 1, opt.n_epochs + 1):
            print('===> Training on {} dataset'.format(opt.data_train))
            train.train(train_dataloader, optimizer, model, epoch, writer, opt)

            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            scheduler.step()

            for data_test in opt.data_test:
                print('===> Testing on {}'.format(data_test))
                result_path = opt.model_path + '/Results/{}'.format(data_test)
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                PSNR, SSIM, Time = train.test(opt.dir_data + 'Test/{}/HR'.format(data_test), result_path, model, opt)
                writer.add_scalar('PSNR_{}/'.format(data_test), PSNR, epoch)
                writer.add_scalar('SSIM_{}/'.format(data_test), SSIM, epoch)

            model_path = opt.model_path + '/Checkpoints/checkpoint_epoch_{}.pth'.format(epoch)
            torch.save({'params': model.state_dict()}, model_path)
            print('Checkpoint saved to {}'.format(model_path))
            torch.cuda.empty_cache()
        writer.close()

    elif opt.train.lower() == 'test':
        for data_test in opt.data_test:
            print('===> Testing on {}'.format(data_test))
            result_path = opt.model_path + '/Results/{}'.format(data_test)
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            with open(opt.model_path + '/Results/Results_{}.csv'.format(data_test), 'w', newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(
                    ['image_name', 'PSNR', 'SSIM', 'Time (ms)'])
                train.test(opt.dir_data + 'Test/{}/HR'.format(data_test), result_path, model, opt, f_csv)

if __name__ == '__main__':
    main()
