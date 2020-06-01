from __future__ import print_function

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from torch.optim.lr_scheduler import StepLR

import sys, os, time
sys.path.append('utils')
sys.path.append('models')
from utils.data import CelebA, pairCelebA, RandomNoiseGenerator
from torch.utils.data import DataLoader
from models.model import Generator, Discriminator
import argparse
import numpy as np
from utils.logger import Logger
import imageio

from config import train_args, argument_report
from summary import VisdomSummary


class PGGAN():
    def __init__(self, G, D, data, noise, opts, vissum):
        self.G = G
        self.D = D
        self.data = data
        self.noise = noise
        self.opts = opts
        self.current_time = time.strftime('%Y-%m-%d_%H%M%S')
        self.logger = Logger('./logs/' + self.current_time + "/")
        self.use_cuda = len(self.opts['gpu']) > 0
        self.latent_size = 512
        self.device = torch.device('cuda:{}'.format(self.opts['gpu'][0]))
        #os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        # batch size map keyed by resolution_level
        self.bs_map = {2**R: self.get_bs(2**R) for R in range(2, 11)} 
        self.rows_map = {32: 8, 16: 4, 8: 4, 4: 2, 2: 2}

        self.restore_model()
        self.vissum = vissum

        # save opts
        if self.opts['restore_dir'] == '' or self.opts['which_file'] == '':
                with open(os.path.join(self.opts['exp_dir'], self.time+'_'+self.opts['vis_env'],
                        'options_%s.txt'%self.current_time), 'w') as f:
                    for k, v in self.opts.items():
                        print('%s: %s' % (k, v), file=f)
                    print('batch_size_map: %s' % self.bs_map, file=f)

    def restore_model(self):
        exp_dir = self.opts['restore_dir']
        which_file = self.opts['which_file']  # 128x128-fade_in-105000
        self.current_time = time.strftime('%Y-%m-%d_%H%M%S')
        if exp_dir == '' or which_file == '':
            self.time = self.current_time
            self._from_resol = self.opts['first_resol']
            self._phase = 'stabilize'
            self._epoch = 0
            self.is_restored = False
            self.opts['sample_dir'] = os.path.join(self.opts['exp_dir'], 
                                        self.current_time+'_'+self.opts['vis_env'],
                                        'samples')
            self.opts['ckpt_dir'] = os.path.join(self.opts['exp_dir'], 
                                        self.current_time+'_'+self.opts['vis_env'], 
                                        'ckpts')
            os.makedirs(self.opts['sample_dir'])
            os.makedirs(self.opts['ckpt_dir'])
            return 
        else:
            prnit("#####################load pk##################")
            pattern = which_file.split('-')
            self._from_resol = int(pattern[0].split('x')[0])
            self._phase = pattern[1]
            self._epoch = int(pattern[2])
            tmp = exp_dir.split('/')
            self.opts['exp_dir'] = '/'.join(tmp[:-1])
            self.time = tmp[-1]
            self.opts['sample_dir'] = os.path.join('exp',exp_dir, 'samples')
            self.opts['ckpt_dir'] = os.path.join('exp',exp_dir, 'ckpts')
            assert os.path.exists(self.opts['sample_dir']) and os.path.exists(self.opts['ckpt_dir'])

            G_model = os.path.join(self.opts['ckpt_dir'], which_file+'-G.pth')
            D_model = os.path.join(self.opts['ckpt_dir'], which_file+'-D.pth')
            assert os.path.exists(G_model) and os.path.exists(D_model)

            self.G.load_state_dict(torch.load(G_model))
            self.D.load_state_dict(torch.load(D_model))
            self.is_restored = True
            print('Restored from dir: %s, pattern: %s' % (exp_dir, which_file))

    def get_bs(self, resolution):
        R = int(np.log2(resolution))
        if R < 7:
            bs = 32 / 2**(max(0, R-4))
        else:
            bs = 8 / 2**(min(2, R-7))
        return int(bs)

    def register_on_gpu(self):
        self.G.to(self.device)
        self.D.to(self.device)

    def create_optimizer(self):
        self.optim_G = optim.Adam(self.G.parameters(), lr=self.opts['g_lr_max'], 
                                betas=(self.opts['beta1'], self.opts['beta2']))
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.opts['d_lr_max'], 
                                betas=(self.opts['beta1'], self.opts['beta2']))
        
    def create_criterion(self):
        self.BCELoss = nn.BCELoss()
        # w is for gan
        if self.opts['gan'] == 'lsgan':
            # sigmoid is applied here
            self.adv_criterion = lambda p,t,w: torch.mean((p-t)**2)  
        elif self.opts['gan'] == 'wgan_gp':
            self.adv_criterion = lambda p,t,w: (-2*t+1) * torch.mean(p)
        elif self.opts['gan'] == 'gan':
            self.adv_criterion = lambda p,t,w: -w*(torch.mean(t*torch.log(p+1e-8)) \
                                            + torch.mean((1-t)*torch.log(1-p+1e-8)))
        else:
            raise ValueError('Invalid/Unsupported GAN: %s.' % self.opts['gan'])

    def compute_adv_loss(self, prediction, target, w):
        return self.adv_criterion(prediction, target, w)

    def compute_additional_g_loss(self):
        return 0.0

    def compute_additional_d_loss(self):  
        # drifting loss and gradient penalty, weighting inside this function
        return 0.0

    def _get_data(self, d):
        return d.item() if isinstance(d, Variable) else d

    def compute_G_loss(self):
        g_adv_loss = self.compute_adv_loss(self.d_fake, True, 1)
        g_add_loss = self.compute_additional_g_loss()
        self.g_adv_loss = self._get_data(g_adv_loss)
        self.g_add_loss = self._get_data(g_add_loss)
        g_sd_loss = self.BCELoss(self.d_fake, self.real_label)
        return g_adv_loss + g_add_loss + g_sd_loss

    def compute_D_loss(self):
        self.d_adv_loss_real = self.compute_adv_loss(self.d_real, True, 0.5)
        self.d_adv_loss_fake = self.compute_adv_loss(self.d_fake, False, 0.5)\
                                * self.opts['fake_weight']
        d_adv_loss = self.d_adv_loss_real + self.d_adv_loss_fake
        d_add_loss = self.compute_additional_d_loss()
        self.d_adv_loss = self._get_data(d_adv_loss)
        self.d_add_loss = self._get_data(d_add_loss)

        d_sd_loss = self.BCELoss(self.d_real, self.real_label) \
                    + self.BCELoss(self.d_fake, self.fake_label)
        return d_adv_loss + d_add_loss + d_sd_loss

    def _rampup(self, epoch, rampup_length):
        if epoch < rampup_length:
            p = max(0.0, float(epoch)) / float(rampup_length)
            p = 1.0 - p
            return np.exp(-p*p*5.0)
        else:
            return 1.0

    def _rampdown_linear(self, epoch, num_epochs, rampdown_length):
        if epoch >= num_epochs - rampdown_length:
            return float(num_epochs - epoch) / rampdown_length
        else:
            return 1.0

    '''Update Learning rate
    '''
    def update_lr(self, cur_nimg):
        for param_group in self.optim_G.param_groups:
            lrate_coef = self._rampup(cur_nimg / 1000.0, self.opts['rampup_kimg'])
            lrate_coef *= self._rampdown_linear(cur_nimg/1000.0, self.opts['total_kimg'], 
                                                self.opts['rampdown_kimg'])
            param_group['lr'] = lrate_coef * self.opts['g_lr_max']
        for param_group in self.optim_D.param_groups:
            lrate_coef = self._rampup(cur_nimg / 1000.0, self.opts['rampup_kimg'])
            lrate_coef *= self._rampdown_linear(cur_nimg/1000.0, self.opts['total_kimg'], 
                                                self.opts['rampdown_kimg'])
            param_group['lr'] = lrate_coef * self.opts['d_lr_max']

    def postprocess(self):
        # TODO: weight cliping or others
        pass
        
    def _numpy2var(self, x):
        var = torch.from_numpy(x)
        if self.use_cuda:
            var = var.to(self.device, non_blocking=True)
        return var

    def _var2numpy(self, var):
        if self.use_cuda:
            return var.cpu().data.numpy()
        return var.data.numpy()

    def compute_noise_strength(self):
        if self.opts.get('no_noise', False):
            return 0

        if hasattr(self, '_d_'):
            self._d_ = self._d_*0.9+np.clip(torch.mean(self.d_real).item(),
                                            0.0,1.0)*0.1
        else:
            self._d_ = 0.0
        strength = 0.2 * max(0, self._d_ - 0.5)**2
        return strength

    def preprocess(self, real1, real2):
        self.real1 = self._numpy2var(real1)
        self.real2 = self._numpy2var(real2)

    def forward_G(self, cur_level):
        self.d_fake = self.D(self.fake1, self.fake2, cur_level=cur_level)

    def forward_D(self, cur_level, detach=True):
        self.fake1, self.fake2 = self.G(self.z1, self.z2, cur_level=cur_level)
        # print('z1  : ', self.z1[0].min().item(), self.z1[0].max().item())
        # print('fake: ', self.fake1[0].min().item(), self.fake1[0].max().item())
        # print('real: ', self.real1[0].min().item(), self.real1[0].max().item())
        vissum.image2d('fake-x'+str(cur_level),'fake',
                torch.cat([self.fake1[[0]],self.fake2[[0]]]), nrow=2)
        vissum.image2d('real-x'+str(cur_level),'real',
                torch.cat([self.real1[[0]],self.real2[[0]]]), nrow=2)

        strength = self.compute_noise_strength()
        self.d_real = self.D(self.real1, self.real2, cur_level=cur_level, 
                            gdrop_strength=strength)
        self.d_fake = self.D(self.fake1.detach(), self.fake2.detach(), 
                            cur_level=cur_level)

    def backward_G(self):
        g_loss = self.compute_G_loss()
        g_loss.backward()
        self.optim_G.step()
        self.g_loss = self._get_data(g_loss)

    def backward_D(self, retain_graph=False):
        d_loss = self.compute_D_loss()
        d_loss.backward(retain_graph=retain_graph)
        self.optim_D.step()
        self.d_loss = self._get_data(d_loss)

    def report(self, it, num_it, phase, resol):
        formation = 'Iter[%d|%d], %s, %s, G: %.3f, D: %.3f, G_adv: %.3f, G_add: %.3f, D_adv: %.3f, D_add: %.3f'
        values = (it, num_it, phase, resol, self.g_loss, self.d_loss, self.g_adv_loss,
                self.g_add_loss, self.d_adv_loss, self.d_add_loss)
        print(formation % values)

    def tensorboard(self, it, num_it, phase, resol, samples):
        # (1) Log the scalar values
        prefix = str(resol)+'/'+phase+'/'
        info = {prefix + 'G_loss': self.g_loss,
                prefix + 'G_adv_loss': self.g_adv_loss,
                prefix + 'G_add_loss': self.g_add_loss,
                prefix + 'D_loss': self.d_loss,
                prefix + 'D_adv_loss': self.d_adv_loss,
                prefix + 'D_add_loss': self.d_add_loss,
                prefix + 'D_adv_loss_fake': self._get_data(self.d_adv_loss_fake),
                prefix + 'D_adv_loss_real': self._get_data(self.d_adv_loss_real)}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, it)

        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in self.G.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary('G/' + prefix +tag, self._var2numpy(value), it)
            if value.grad is not None:
                self.logger.histo_summary('G/' + prefix +tag + '/grad', 
                                        self._var2numpy(value.grad), it)

        for tag, value in self.D.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary('D/' + prefix + tag, self._var2numpy(value), it)
            if value.grad is not None:
                self.logger.histo_summary('D/' + prefix + tag + '/grad',
                                          self._var2numpy(value.grad), it)

        # (3) Log the images
        # info = {'images': samples[:10]}
        # for tag, images in info.items():
        #     logger.image_summary(tag, images, it)
    
    def create_fix_noizes(self, batch_size, d_i):
        fixed_z_i = torch.Tensor(batch_size, int(d_i/2), 1, 1).uniform_(-1, 1)
        fixed_zo1 = torch.Tensor(batch_size, int(d_i/2), 1, 1).uniform_(-1, 1)
        fixed_zo2 = torch.Tensor(batch_size, int(d_i/2), 1, 1).uniform_(-1, 1)
        fixed_zo3 = torch.Tensor(batch_size, int(d_i/2), 1, 1).uniform_(-1, 1)
        fixed_zo4 = torch.Tensor(batch_size, int(d_i/2), 1, 1).uniform_(-1, 1)

        fixed_z1 = torch.cat([fixed_z_i, fixed_zo1], 1).to(self.device, non_blocking=True)
        fixed_z2 = torch.cat([fixed_z_i, fixed_zo2], 1).to(self.device, non_blocking=True)
        fixed_z3 = torch.cat([fixed_z_i, fixed_zo3], 1).to(self.device, non_blocking=True)
        fixed_z4 = torch.cat([fixed_z_i, fixed_zo4], 1).to(self.device, non_blocking=True)
        return fixed_z1, fixed_z2, fixed_z3, fixed_z4


    def train_phase(self, R, phase, batch_size, cur_nimg, from_it, total_it):
        assert total_it >= from_it
        resol = 2 ** (R+1)
        # R=1, phase=stabilize, batch_size=32, cur_nimg=0, from_it=0, total_it=18750

        # self.fixed_z1, self.fixed_z2, self.fixed_z3, self.fixed_z4 = self.create_fix_noizes(batch_size, 512)
        for it in range(from_it, total_it):
            if phase == 'stabilize':
                cur_level = R
            else:
                cur_level = R + total_it/float(from_it)
            cur_resol = 2 ** int(np.ceil(cur_level+1))

            # get a batch noise and real images
            x1, x2 = self.data(batch_size, cur_resol, cur_level)
            
            z_i = torch.Tensor(batch_size, 256, 1,1).uniform_(-1, 1)
            z_o1 = torch.Tensor(batch_size, 256, 1,1).uniform_(-1, 1)
            z_o2 = torch.Tensor(batch_size, 256, 1,1).uniform_(-1, 1)
            self.z1 = torch.cat([z_i,z_o1],1).to(self.device, non_blocking=True)
            self.z2 = torch.cat([z_i,z_o2],1).to(self.device, non_blocking=True)
            self.real_label = torch.ones(batch_size).to(self.device, non_blocking=True)
            self.fake_label = torch.zeros(batch_size).to(self.device, non_blocking=True)

            # ===preprocess===
            self.preprocess(x1, x2)
            self.update_lr(cur_nimg)

            # ===update D===
            self.optim_D.zero_grad()
            self.forward_D(cur_level, detach=True)
            self.backward_D()

            # ===update G===
            self.optim_G.zero_grad()
            self.forward_G(cur_level)
            self.backward_G()

            # ===report ===
            self.report(it, total_it, phase, cur_resol)

            cur_nimg += batch_size

            # ===generate sample images===
            samples1,samples2 = [],[]
            if (it % self.opts['sample_freq'] == 0) or it == total_it-1:
                samples1, samples2 = self.sample()
                samples1 = np.array(samples1, dtype=np.uint8) 
                samples2 = np.array(samples2, dtype=np.uint8)
                imageio.imwrite(os.path.join(self.opts['sample_dir'],
                                'fake1_%dx%d-%s-%s.png' % (cur_resol, 
                                cur_resol, phase, str(it).zfill(6))), samples1)
                imageio.imwrite(os.path.join(self.opts['sample_dir'],
                                'fake2_%dx%d-%s-%s.png' % (cur_resol, 
                                cur_resol, phase, str(it).zfill(6))), samples2)

            # ===tensorboard visualization===
            if (it % self.opts['sample_freq'] == 0) or it == total_it - 1:
                self.tensorboard(it, total_it, phase, cur_resol, samples1)
                self.tensorboard(it, total_it, phase, cur_resol, samples2)

            # ===save model===
            if (it % self.opts['save_freq'] == 0 and it > 0) or it == total_it-1:
                self.save(os.path.join(self.opts['ckpt_dir'], 
                '%dx%d-%s-%s' % (cur_resol, cur_resol, phase, str(it).zfill(6))))
            
            if it == total_it-1:
                self.vissum.image2d('save_fake1_'+str(cur_resol), 'fake1', self.fake1[0])
                self.vissum.image2d('save_fake2_'+str(cur_resol), 'fake2', self.fake2[0])
            self.vissum.save()

        
    def train(self):
        # prepare
        self.create_optimizer()
        self.create_criterion()
        # self.register_on_gpu()

        to_level = int(np.log2(self.opts['target_resol']))
        from_level = int(np.log2(self._from_resol))
        assert 2**to_level == self.opts['target_resol'] and 2**from_level == self._from_resol and to_level >= from_level >= 2

        train_kimg = int(self.opts['train_kimg'] * 1000)
        transition_kimg = int(self.opts['transition_kimg'] * 1000)

        # to_level=8, from_level=2, train_kimg=600000, transition_kimg=600000
        print('Start training..')
        for R in range(from_level-1, to_level):
            batch_size = self.bs_map[2 ** (R+1)] # 32
            phases = {'stabilize':[0, train_kimg//batch_size], 
                    'fade_in':[train_kimg//batch_size+1, 
                        (transition_kimg+train_kimg)//batch_size]}
            # {'stabilize': [0, 18750], 'fade_in': [18751, 37500]}
            if self.is_restored and R == from_level-1:
                phases[self._phase][0] = self._epoch + 1
                if self._phase == 'fade_in':
                    del phases['stabilize']

            for phase in ['stabilize', 'fade_in']:
                if phase in phases:
                    _range = phases[phase]
                    self.train_phase(R, phase, batch_size, _range[0]*batch_size,
                                     _range[0], _range[1])

    def sample(self):
        batch_size = self.z1.size(0)
        n_row = self.rows_map[batch_size]
        n_col = int(np.ceil(batch_size / float(n_row)))
        samples1,samples2 = [],[]
        i = j = 0
        for row in range(n_row):
            one_row, two_row = [], []
            # fake
            for col in range(n_col):
                one_row.append(self.fake1[i].cpu().data.numpy())
                two_row.append(self.fake2[i].cpu().data.numpy())
                i += 1
            # real
            for col in range(n_col):
                one_row.append(self.real1[j].cpu().data.numpy())
                two_row.append(self.real2[j].cpu().data.numpy())
                j += 1
            samples1 += [np.concatenate(one_row, axis=2)]
            samples2 += [np.concatenate(two_row, axis=2)]
        samples1 = np.concatenate(samples1, axis=1).transpose([1, 2, 0])
        samples2 = np.concatenate(samples2, axis=1).transpose([1, 2, 0])

        half = samples1.shape[1] // 2
        samples1[:, :half, :] = samples1[:, :half, :] - np.min(samples1[:, :half, :])
        samples1[:, :half, :] = samples1[:, :half, :] / np.max(samples1[:, :half, :])
        samples1[:, half:, :] = samples1[:, half:, :] - np.min(samples1[:, half:, :])
        samples1[:, half:, :] = samples1[:, half:, :] / np.max(samples1[:, half:, :])
        half = samples2.shape[1] // 2
        samples2[:, :half, :] = samples2[:, :half, :] - \
            np.min(samples2[:, :half, :])
        samples2[:, :half, :] = samples2[:, :half, :] / \
            np.max(samples2[:, :half, :])
        samples2[:, half:, :] = samples2[:, half:, :] - \
            np.min(samples2[:, half:, :])
        samples2[:, half:, :] = samples2[:, half:, :] / \
            np.max(samples2[:, half:, :])
        return samples1, samples2

    def save(self, file_name):
        g_file = file_name + '-G.pth'
        d_file = file_name + '-D.pth'
        torch.save(self.G.state_dict(), g_file)
        torch.save(self.D.state_dict(), d_file)

if __name__ == '__main__':
    FG = train_args()
    opts = {k:v for k,v in FG._get_kwargs()}
    # torch setting
    device = torch.device('cuda:{}'.format(FG.gpu[0]))
    torch.cuda.set_device(FG.gpu[0])

    # visdom setting
    vissum = VisdomSummary(port=FG.vis_port, env=FG.vis_env)

    # Dimensionality of the latent vector.
    latent_size = 512
    # Use sigmoid activation for the last layer?
    sigmoid_at_end = FG.gan in ['lsgan', 'gan']
    if hasattr(FG, 'no_tanh'):
        tanh_at_end = False
    else:
        tanh_at_end = True

    G = Generator(num_channels=3, latent_size=latent_size, resolution=FG.target_resol, 
                    fmap_max=latent_size, fmap_base=8192, tanh_at_end=tanh_at_end).to(device)
    D = Discriminator(num_channels=3, mbstat_avg=FG.mbstat_avg, resolution=FG.target_resol, 
                    fmap_max=latent_size, fmap_base=8192, sigmoid_at_end=sigmoid_at_end).to(device)

    print(G)
    print(D)
    # exit()
    
    if len(FG.gpu) != 1:
        G = torch.nn.DataParallel(G, FG.gpu)
        D = torch.nn.DataParallel(D, FG.gpu)

    data = pairCelebA(FG.celeba_dir, FG.img_dir, vissum)
    noise = RandomNoiseGenerator(latent_size, 'gaussian') #(32, 512)
    pggan = PGGAN(G, D, data, noise, opts, vissum)
    pggan.train()
