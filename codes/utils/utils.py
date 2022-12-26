import os
import csv
import logging
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch.nn.functional as F
from PIL import ImageFile
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image







#-----------------------------------------------------------------------------
class WM_Dataset(Dataset):
    def __init__(self, opt):
        super(WM_Dataset, self).__init__()
        # train and val
        if opt['train/test'] == 'train':
            self.num_of_load = opt['datasets']['nDatasets']['num']
            path = opt['path']['train_folder']
        else:   # test only
            self.num_of_load = opt['datasets']['test']['num']
            path = opt['path']['test_folder']
        #
        imgs=os.listdir(path)
        self.imgs=[os.path.join(path,k) for k in imgs]
        self.input_transforms = transforms.Compose([
            transforms.CenterCrop((opt['datasets']['H'], opt['datasets']['W'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
    def __getitem__(self, index):
        #
        data = self.imgs[index]
        img = Image.open(data)   
        img = img.convert('RGB') 
        img = self.input_transforms(img)  
        return img

    def __len__(self):
        return self.num_of_load


def train_val_loaders(opt):
    # Dataset init
    data_input = WM_Dataset(opt)
    # Split dataset
    train_size = int(opt['datasets']['nDatasets']['nTrain'] * len(data_input))
    test_size = len(data_input) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data_input, [train_size, test_size])
    # Dataloader
    train_loader = DataLoader(dataset = train_dataset, batch_size=opt['train']['batch_size'], shuffle=True, num_workers=opt['train']['num_workers'])
    val_loader = DataLoader(dataset = test_dataset,  batch_size=opt['train']['batch_size'], shuffle=False, num_workers=opt['train']['num_workers'])
    
    return train_loader, val_loader


def test_loader(opt):
    # Dataset init
    data_input = WM_Dataset(opt)
    # Dataloader
    test_loader = DataLoader(dataset = data_input,  batch_size=opt['train']['batch_size'], shuffle=False, num_workers=opt['train']['num_workers'])
    return test_loader

# ---------------------------------------------------------------

def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def write_losses(file_name, epoch, loss1=0, loss2=0, loss3=0):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row_to_write = ['{:.0f}'.format(epoch)] + ['{:.4f}'.format(loss1)] + ['{:.4f}'.format(loss2)] + ['{:.4f}'.format(loss3)]
        writer.writerow(row_to_write)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_image(
    tensor,
    fp,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range = None,
    scale_each: bool = False,
    pad_value: int = 0,
    format = None,
) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)



def _normalize(input_tensor):
    min_val, max_val = torch.min(input_tensor), torch.max(input_tensor)
    return (input_tensor-min_val) / (max_val-min_val)



def save_images(cover_img, watermarking_img, noised_img, img_fake, epoch, current_step, folder, time_now_NewExperiment, opt, resize_to=None):
    #
    cover_img = cover_img.cpu()
    watermarking_img = watermarking_img.cpu()
    noised_img = noised_img.cpu()
    img_fake = img_fake.cpu()
    # scale values to range [0, 1] from original range of [-1, 1]
    cover_img = (cover_img + 1) / 2
    watermarking_img = (watermarking_img + 1) / 2
    noised_img = (noised_img + 1) / 2
    diff_w2co = _normalize(torch.abs(cover_img - watermarking_img))
    diff_w2no = _normalize(torch.abs(noised_img - watermarking_img))
    #
    if resize_to is not None:
        cover_img = F.interpolate(cover_img, size=resize_to)
        watermarking_img = F.interpolate(watermarking_img, size=resize_to)
        diff_w2co = F.interpolate(diff_w2co, size=resize_to)
        diff_w2no = F.interpolate(diff_w2no, size=resize_to)
    #
    stacked_images = torch.cat([cover_img, watermarking_img, noised_img, diff_w2co, diff_w2no], dim=0)
    filename = os.path.join(folder, 'epoch-{}-step-{}-{}.png'.format(epoch, current_step, time_now_NewExperiment))
    saveFormat = opt['train']['saveFormat']
    if opt['train']['saveStacked']:
        save_image(stacked_images, filename + saveFormat, cover_img.shape[0], normalize=False)
    else:
        save_image(watermarking_img, filename + '-watermarking' + saveFormat, normalize=False)




def save_tensor_images(input, folder_name=None):
    #
    img0 = input[0,:,:,:]
    img0 = img0.unsqueeze(0)
    img0 = img0.cpu()
    img0 = (img0 + 1) / 2
    #
    img0 = img0.reshape(img0.shape[1],img0.shape[0],img0.shape[2], img0.shape[3])
    #
    folder = '/.../debug/{}'.format(folder_name)
    mkdir(folder)
    saveFormat = '.png'
    stacked_images = img0
    save_image(stacked_images, folder + saveFormat, input.shape[1], normalize=False)


#
def func_loss_RecMsg(RecMsgLoss, message, msg_fake_1, msg_fake_2):
    if msg_fake_1 != None:
        loss_RecMsg  =  RecMsgLoss(message, msg_fake_1)
    elif msg_fake_2 != None:
        loss_RecMsg  =  RecMsgLoss(message, msg_fake_2)

    return loss_RecMsg


#
def loss_lamd(current_step, opt):
    #
    if opt['loss']['option'] == 'lamd':
        # loss weight 
        lw_Rec      = opt['loss']['lamd']['Rec']    # [1, 5, 10, 20, 40] 
        lw_Eec      = opt['loss']['lamd']['Eec']    # [1, 5, 10, 20, 40] 
        lw_Msg      = opt['loss']['lamd']['Msg'] 
        #
        lamd_ms_Rec = opt['loss']['lamd']['milestones_Rec']      # [1000, 5000, 10000, 20000] 
        lamd_ms_Enc = opt['loss']['lamd']['milestones_Eec']      # [1000, 5000, 10000, 20000] 
        lamd_ms_Msg = opt['loss']['lamd']['milestones_Msg']
        #
        length_rec      = len(lw_Rec)
        length_enc      = len(lw_Eec)
        length_msg      = len(lw_Msg)
        # loss weight RecImg
        for i in range(length_rec):
            if current_step <= lamd_ms_Rec[i]:
                lwRec = lw_Rec[i]
                break
            elif lamd_ms_Rec[i] < current_step <= lamd_ms_Rec[i + 1]:
                lwRec = lw_Rec[i + 1]
                break
        # loss weight EncImg
        for i in range(length_enc):
            if current_step <= lamd_ms_Enc[i]:
                lwEnc = lw_Eec[i]
                break
            elif lamd_ms_Enc[i] < current_step <= lamd_ms_Enc[i + 1]:
                lwEnc = lw_Eec[i + 1]
                break
        # loss weight Msg
        for i in range(length_msg):
            if current_step <= lamd_ms_Msg[i]:
                lwMsg = lw_Msg[i]
                break
            elif lamd_ms_Msg[i] < current_step <= lamd_ms_Msg[i + 1]:
                lwMsg = lw_Msg[i + 1]
                break
        #
        lossWeight = {} #{'lwRec', 'lwEnc', 'lwMsg', 'lwFre', 'lwOffset', 'lwGAN'}
        lossWeight['lwRec'] = lwRec           # loss weight RecImg
        lossWeight['lwEnc'] = lwEnc          # loss weight EecImg
        lossWeight['lwMsg'] = lwMsg          # loss weight msg    

    return lossWeight


def func_loss(lw, loss_RecImg, loss_encoded, loss_RecMsg):
    #
    train_loss = 0
    #
    if lw['lwRec'] != 0 :
        train_loss = train_loss + lw['lwRec'] * loss_RecImg
    if lw['lwEnc'] != 0 :
        train_loss = train_loss + lw['lwEnc'] * loss_encoded
    if lw['lwMsg'] != 0 :
        train_loss = train_loss + lw['lwMsg'] * loss_RecMsg

    return train_loss





def func_mean_filter_None(base, input, type= 'add'):
    #
    if type == 'add':
        if base == None or input == None:
            return None
        else:
            return  base+input.item()
    #
    if type == 'div':
        if base == None or input == None:
            return 'None'
        else:
            return base/input


def bitWise_accurary(msg_fake, message, opt):
    #
    if msg_fake == None:
        return None, None
    else:
        #
        if opt['datasets']['msg']['mod_a']:     # msg in [0, 1]
            DecodedMsg_rounded = msg_fake.detach().cpu().numpy().round().clip(0, 1)
        elif opt['datasets']['msg']['mod_b']:   # msg in [-1, 1]
            DecodedMsg_rounded = msg_fake.detach().cpu().numpy().round().clip(-1, 1)
            DecodedMsg_rounded, message = (DecodedMsg_rounded + 1) / 2, (message + 1) / 2 
        #
        diff = DecodedMsg_rounded - message.detach().cpu().numpy()
        count = np.sum(np.abs(diff))
        #
        accuracy = (1 - count / (opt['train']['batch_size'] * opt['network']['message_length'])) 
        BitWise_AvgErr = count / (opt['train']['batch_size'] * opt['network']['message_length'])
        #
        return accuracy * 100, BitWise_AvgErr



def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def filter_None_value(input):
    return 'None' if input == None else input 


def log_info(lw, current_epoch, total_epochs, current_step, Lr_current, psnr_wm2co, psnr_no2co, psnr_rec2co, psnr_wm2no, \
            ssim_wm2co, ssim_rec2co, BitWise_AvgErr1, BitWise_AvgErr2, train_loss=None, loss_RecImg=None, \
                loss_RecMsg=None, loss_encoded=None, noise_choice=None):
    #
    BitWise_AvgErr1 = filter_None_value(BitWise_AvgErr1)
    BitWise_AvgErr2 = filter_None_value(BitWise_AvgErr2)

    logging.info('epoch: {}/{}'.format(current_epoch, total_epochs))
    logging.info('step:{}:'.format(current_step))
    logging.info('lr:{}'.format('{:.7f}'.format(Lr_current)))
    logging.info('lw:{}/{}/{}'.format('{}'.format(lw['lwRec']), '{}'.format(lw['lwEnc']), '{}'.format(lw['lwMsg'])))
    logging.info('noise_choice:{}'.format('{}'.format(noise_choice)))
    logging.info('L_RecMsg:{}'.format('{:.6f}'.format(loss_RecMsg.item())))
    logging.info('L_encoded:{}'.format('{:.6f}'.format(loss_encoded.item())))
    logging.info('Ssim_wm/rec:{}'.format('{:.4f}/{:.4f}'.format(ssim_wm2co.item(), ssim_rec2co.item())))
    logging.info('Psnr_wm/no/rec/wm2no: {}'.format('{:.1f}/{:.1f}/{:.1f}/{:.1f}'.format(psnr_wm2co.item(), psnr_no2co.item(), psnr_rec2co.item(), psnr_wm2no.item())))
    logging.info('msg_AvgErr_1_2: {}/{}'.format(BitWise_AvgErr1, BitWise_AvgErr2))
    logging.info('---------------------------------------------------------------------------------------------')



def log_info_test(current_step, total_steps, Lr_current, psnr_wm2co, psnr_no2co, psnr_rec2co, psnr_wm2no,\
            ssim_wm2co, ssim_rec2co, BitWise_AvgErr1, BitWise_AvgErr2, noise_choice):
    #
    BitWise_AvgErr1 = filter_None_value(BitWise_AvgErr1)
    BitWise_AvgErr2 = filter_None_value(BitWise_AvgErr2)

    logging.info('step:{}/{}:'.format(current_step, total_steps))
    logging.info('lr:{}'.format('{:.7f}'.format(Lr_current)))
    logging.info('noise_choice:{}'.format('{}'.format(noise_choice)))
    logging.info('Ssim_wm/rec:{}'.format('{:.4f}/{:.4f}'.format(ssim_wm2co.item(), ssim_rec2co.item())))
    logging.info('Psnr_wm/no/rec/wm2no: {}'.format('{:.1f}/{:.1f}/{:.1f}/{:.1f}'.format(psnr_wm2co.item(), \
        psnr_no2co.item(), psnr_rec2co.item(), psnr_wm2no.item())))
    logging.info('msg_AvgErr_1_2: {}/{}'.format(BitWise_AvgErr1, BitWise_AvgErr2))
    logging.info('---------------------------------------------------------------------------------------------')



def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)

