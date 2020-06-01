from __future__ import absolute_import, division, print_function
import argparse
import os

def argument_report(arg, end='\n'):
    d = arg.__dict__
    keys = d.keys()
    report = '{:15}    {}'.format('running_fold', d['running_fold'])+end
    report += '{:15}    {}'.format('memo', d['memo'])+end
    for k in sorted(keys):
        if k == 'running_fold' or k == 'memo':
            continue
        report += '{:15}    {}'.format(k, d[k])+end
    return report


def _base_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpu', default=(0,1,2,3), type=int, nargs='+', help='gpu(s) to use.')
    parser.add_argument('--gan', default='lsgan', type=str, help='model: lsgan/wgan_gp/gan, currently only support lsgan or gan with no_noise option.')
    parser.add_argument('--drift', default=1e-3, type=float, help='drift, only available for wgan_gp.')
    parser.add_argument('--mbstat_avg', default='all', type=str, help='MinibatchStatConcatLayer averaging strategy (Which dimensions to average the statistic over?)')
    parser.add_argument('--exp_dir', default='./exp', type=str, help='experiment dir.')
    parser.add_argument('--ckpt_dir', default='',
                        type=str, help='experiment dir.')
    parser.add_argument('--no_noise', action='store_true', help='do not add noise to real data.')
    parser.add_argument('--no_tanh', action='store_true', help='do not use tanh in the last layer of the generator.')
    parser.add_argument('--restore_dir', default='', type=str, help='restore from which exp dir.')
    parser.add_argument('--celeba_dir', default='./datasets/celeba', type=str, help='restore from which exp dir.')
    parser.add_argument('--img_dir', default='./datasets/CelebA-HQ', type=str, help='restore from which exp dir.')
    parser.add_argument('--outf', default='./results', type=str, help='restore from which exp dir.')
    parser.add_argument('--which_file', default='', type=str, help='restore from which file, e.g. 128x128-fade_in-105000.')
    parser.add_argument('--load_pkl', default='false', type=str, help='load model pickle.')

    parser.add_argument('--vis_env', type=str, default='SD-PGGAN')
    parser.add_argument('--vis_port', type=int, default=10002)
    # parser.add_argument('--data_root', type=str, default='data')
    # parser.add_argument('--save_dir', type=str, default='results')
    # parser.add_argument('--checkpoint_root', type=str, default='checkpoint')
    # parser.add_argument('--labels', type=str, nargs='+', default=('AD','CN'))
    # parser.add_argument('--fold', type=int, default=5)
    # parser.add_argument('--running_fold', type=int, default=0)
    # parser.add_argument('--memo', type=str, default='')
    # parser.add_argument('--subject_ids_path', type=str,
    #                     default=os.path.join('data', 'subject_ids.pkl'))
    # parser.add_argument('--diagnosis_path', type=str,
    #                     default=os.path.join('data', 'diag nosis.pkl'))


    return parser


def train_args():
    parser = argparse.ArgumentParser(parents=[_base_parser()])
    parser.add_argument('--train_kimg', default=600, type=float, help='#*1000 real samples for each stabilizing training phase.')
    parser.add_argument('--transition_kimg', default=600, type=float, help='#*1000 real samples for each fading in phase.')
    parser.add_argument('--total_kimg', default=10000, type=float, help='total_kimg: a param to compute lr.')
    parser.add_argument('--rampup_kimg', default=10000, type=float, help='rampup_kimg.')
    parser.add_argument('--rampdown_kimg', default=10000, type=float, help='rampdown_kimg.')
    parser.add_argument('--g_lr_max', default=1e-3, type=float, help='Generator learning rate')
    parser.add_argument('--d_lr_max', default=1e-3, type=float, help='Discriminator learning rate')
    parser.add_argument('--fake_weight', default=0.1, type=float, help="weight of fake images' loss of D")
    parser.add_argument('--beta1', default=0, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.99, type=float, help='beta2 for adam')
    parser.add_argument('--first_resol', default=4, type=int, help='first resolution')
    parser.add_argument('--target_resol', default=256, type=int, help='target resolution')
    parser.add_argument('--sample_freq', default=500, type=int, help='sampling frequency.')
    parser.add_argument('--save_freq', default=5000, type=int, help='save model frequency.')
    # parser.add_argument('--num_epoch', type=int, default=10000)
    # parser.add_argument('--batch_size', type=int, default=20)
    # parser.add_argument('--l2_decay', type=float, default=0.0005)
    # parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--lrG', type=float, default=0.0002)
    # parser.add_argument('--lrD', type=float, default=0.0002)
    # parser.add_argument('--lr_gamma', type=float, default=0.999)
    # parser.add_argument('--beta1', type=float, default=0.5)
    # parser.add_argument('--beta2', type=float, default=0.999)
    # parser.add_argument('--z', type=int, default=128)
    # parser.add_argument('--d_code', type=int, default=2)
    # parser.add_argument('--c_code', type=int, default=2)
    # parser.add_argument('--axis', type=int, default=0)
    # parser.add_argument('--isize', type=int, default=79)
    # parser.add_argument('--SUPERVISED', type=str, default='True')
    # parser.add_argument('--lr_adam', type=float, default=1e-4)
    # parser.add_argument('--std', type=float, default=0.02, help='for weight')

    args = parser.parse_args()
    return args
    
