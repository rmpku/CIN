
import os
import torch
import torch.nn as nn
import utils.utils as utils




class Checkpoint(nn.Module):
    def __init__(self, path, opt):
        super(Checkpoint, self).__init__()
        self.path = path
        self.opt = opt
        utils.mkdir(self.path)

    def save(self, model, current_step, epoch, model_name):
        state = {"{}".format(model_name): model.state_dict()}
        save_filename = 'epoch-{}_step-{}-{}.pth'.format(epoch, current_step, model_name)
        save_path = os.path.join(self.path, save_filename)
        torch.save(state, save_path)

    def resume_training(self, model, modle_name, resume_state): 
        if self.opt['train']['resume']['All'] == True:
            model.load_state_dict(resume_state['{}'.format(modle_name)])
        elif self.opt['train']['resume']['Partial'] == True:
            model_dict = model.state_dict()
            pretrained_dict = {i: j for k, v in resume_state.items() for i, j in resume_state[k].items() if i in model_dict} 
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else: 
            print('Resume option error !')
            exit()
        return model


