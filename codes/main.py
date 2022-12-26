import torch
from utils.yml import parse_yml, dict_to_nonedict, set_random_seed
import random
import os
import time
import torch
import utils.utils as utils
import logging
import sys
from utils.yml import dict2str
from models.Network import Network




#---------------------------------------------------------------------------------------------------
def main():

    name = str("CIN")

    # Read config
    yml_path = '.../codes/options/opt.yml'
    option_yml = parse_yml(yml_path)

    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(option_yml)
    
    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    set_random_seed(seed)
    
    # cudnn
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = opt["train"]["os_environ"]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # path (log, experiment results and loss)
    time_now_NewExperiment = time.strftime("%Y-%m-%d-%H:%M", time.localtime()) 
    if opt['subfolder'] != None:
        subfolder_name = opt['subfolder'] + '/-'
    else:
        subfolder_name = ''
    #
    folder_str = opt['path']['logs_folder'] + name + '/' + subfolder_name + str(time_now_NewExperiment) + '-' + opt['train/test']
    log_folder = folder_str + '/logs'
    img_w_folder_tra = folder_str  + '/img/train'
    img_w_folder_val = folder_str  + '/img/val'
    img_w_folder_test = folder_str + '/img/test'
    loss_w_folder = folder_str  + '/loss'
    path_checkpoint = folder_str  + '/path_checkpoint'
    opt_folder = folder_str  + '/opt'
    opt['path']['folder_temp'] = folder_str  + '/temp'
    #
    path_in = {'log_folder':log_folder, 'img_w_folder_tra':img_w_folder_tra, \
                    'img_w_folder_val':img_w_folder_val,'img_w_folder_test':img_w_folder_test,\
                        'loss_w_folder':loss_w_folder, 'path_checkpoint':path_checkpoint, \
                            'opt_folder':opt_folder, 'time_now_NewExperiment':time_now_NewExperiment}

    # create logger
    utils.mkdir(log_folder)
    logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(log_folder, f'{name}-{time_now_NewExperiment}.log')),
                        logging.StreamHandler(sys.stdout)
                    ])
                    
    # log option_yml
    utils.mkdir(opt_folder)
    utils.setup_logger('base', opt_folder, 'train_' + name, level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(dict2str(opt))
    
    # seed
    logging.info('\nSeed = {}'.format(seed))
    
    # load data
    train_data, val_data = utils.train_val_loaders(opt)      # from one folders (train only)

    # log datesets number
    file_count_tr = len(train_data.dataset)
    file_count_val = len(val_data.dataset)
    logging.info('\nTrain_Img_num {} \nval_Img_num {}'.format(file_count_tr, file_count_val))
    
    # step config
    total_epochs = opt['train']['epoch']
    start_epoch = opt['train']['set_start_epoch']
    start_step = opt['train']['start_step']

    # log BPP
    Bpp = opt['network']['message_length'] / (opt['network']['H'] * opt['network']['W'] * opt['network']['input']['in_img_nc'])
    logging.info('BPP = {:.4f}\n'.format(Bpp))

    # log
    logging.info('\nStarting epoch {}\nstart_step {}'.format(start_epoch, start_step))
    logging.info('Batch size = {}\n'.format(opt['train']['batch_size']))
    
    #
    network = Network(opt, device, path_in)



# ------------------------------------------------------------------------------------------------------------------
    for current_epoch in range(start_epoch, total_epochs + 1):
        #
        if opt['train/test'] == 'train':
            network.train(train_data, current_epoch)
            if current_epoch % opt['train']['val']['per_epoch'] == 0:
                network.validation(val_data, current_epoch)
        #
        if opt['train/test'] == 'test':
            test_data = utils.test_loader(opt) 
            network.test(test_data, current_epoch)
            print('main: Test end !')
            break




# ------------------------------------------------------------------------
if __name__ == '__main__':
    main()











