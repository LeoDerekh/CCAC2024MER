import time
from os.path import join

from ema import EMA
from model.model_utils import *
from option.option import Options
from utils.utils import *
import torch
import os
from torch.utils.tensorboard import SummaryWriter


def train(opt):
    start_time = time.time()
    
    train_loader, val_loader = load_me_data(opt)

    model = get_model(opt)

    criterionCE = torch.nn.CrossEntropyLoss().to(opt.device)
    torch.nn.DataParallel(criterionCE, opt.gpu_ids)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(opt, pg)

    scheduler = get_scheduler(opt, optimizer)

    train_losses = []
    val_losses = []

    best_UF1 = -1
    best_UAR = -1
    best_ACC = -1
    best_epoch = 0
    best_class_accuracies = None

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=opt.log_dir)

    if opt.use_ema:
        # 初始化
        ema = EMA(model, opt.ema_decay)
        ema.register()

    for epoch in range(1, opt.epochs + 1):
        print("Epoch {}".format(epoch))
        # train
        train_loss, train_UF1, train_UAR, train_ACC, train_class_accuracies = train_one_epoch(opt=opt,
                                                                                              model=model,
                                                                                              criterion=criterionCE,
                                                                                              optimizer=optimizer,
                                                                                              data_loader=train_loader,
                                                                                              device=opt.device,
                                                                                              epoch=epoch,
                                                                                              ema=ema if opt.use_ema else None)
        print(
            'Train Epoch {} => train_UF1: {:.4f}, train_UAR: {:.4f}, train_ACC: {:.4f}, train_class_accuracies: {}'.format(
                epoch,
                train_UF1,
                train_UAR,
                train_ACC,
                train_class_accuracies))

        train_losses.append(train_loss)

        # Log training loss and accuracy to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('UF1/train', train_UF1, epoch)
        writer.add_scalar('ACC/train', train_ACC, epoch)

        scheduler.step()
        if opt.use_ema:
            ema.apply_shadow()
        # validate
        val_loss, val_UF1, val_UAR, val_ACC, val_class_accuracies = evaluate(opt=opt,
                                                                             model=model,
                                                                             criterion=criterionCE,
                                                                             data_loader=val_loader,
                                                                             device=opt.device,
                                                                             epoch=epoch,)
        if opt.use_ema:
            ema.restore()
        print(
            'Validation Epoch {} => val_UF1: {:.4f}, val_UAR: {:.4f}, val_ACC: {:.4f}, val_class_accuracies: {}'.format(
                epoch,
                val_UF1,
                val_UAR,
                val_ACC,
                val_class_accuracies))
        print()
        val_losses.append(val_loss)

        # Log validation loss and accuracy to TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('UF1/val', val_UF1, epoch)
        writer.add_scalar('ACC/val', val_ACC, epoch)

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('UF1', {'train': train_UF1, 'val': val_UF1}, epoch)
        writer.add_scalars('ACC', {'train': train_ACC, 'val': val_ACC}, epoch)

        if opt.use_ema:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
            }
        else:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }
        torch.save(checkpoint, os.path.join(opt.ckpt_dir, 'model_%s.pth' % epoch))

        # save best checkpoint
        # note that best_UAR, best_ACC, best_epoch are best when UF1 is best
        if val_UF1 > best_UF1:
            best_UF1 = val_UF1
            best_UAR = val_UAR
            best_ACC = val_ACC
            best_class_accuracies = val_class_accuracies
            best_epoch = epoch
            checkpoint = {
                'epoch': best_epoch,
                'state_dict': model.state_dict(),
            }
            torch.save(checkpoint, os.path.join(opt.ckpt_dir, 'best_model.pth'))

    print('best_epoch: {} => best_UF1: {:.4f}, best_UAR: {:.4f}, best_ACC: {:.4f}, best_class_accuracies: {}'.format(
        best_epoch,
        best_UF1,
        best_UAR,
        best_ACC,
        best_class_accuracies))

    # save info
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f"Elapsed time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds")

    # info to save in results directory
    info_dict = {'model': opt.model, 'num_classes': opt.num_classes, 'pretrained': opt.pretrained,
                 'best_UF1': best_UF1, 'best_UAR': best_UAR, 'best_ACC': best_ACC,
                 'best_class_accuracies': str(best_class_accuracies), 'best_epoch': best_epoch,
                 'epochs': opt.epochs, 'batch_size': opt.batch_size, 'n_workers': opt.n_workers,
                 'lr': opt.lr, 'optimizer': opt.optimizer, 'weight_decay': opt.weight_decay, 'lr_policy': opt.lr_policy,
                 'scale': opt.scale_factor, 'use_ema': opt.use_ema, 'ema_decay': opt.ema_decay,
                 'ckpt_dir': opt.ckpt_dir, 'gpu_ids': str(opt.gpu_ids), 'lucky_seed': opt.lucky_seed,
                 'data_path': opt.data_path, 'start_time': start_time, 'end_time': end_time,
                 'elapsed_time': f"{hours}h{minutes}m{seconds:.2f}s"}
    result_path = join(opt.results, "result_B.csv")
    save_info_append(result_path, info_dict)

    # Close the TensorBoard writer
    writer.close()

    print('[THE END]')


if __name__ == '__main__':
    opt = Options().parse()
    train(opt)
