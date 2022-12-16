import os
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import numpy as np

from tvae.data.dsprites import get_dataloader
from tvae.containers.tvae import TVAE
from tvae.models.mlp import Encoder_Chairs, Decoder_Chairs
from tvae.containers.encoder import Gaussian_Encoder
from tvae.containers.decoder import Bernoulli_Decoder, Gaussian_Decoder
from tvae.containers.grouper import Stationary_Capsules_1d
from tvae.utils.logging import configure_logging, get_dirs
from tvae.utils.train_loops import train_epoch, validate_epoch, eval_epoch

class DynBinarizedMNIST(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        #x = (x > torch.distributions.Uniform(0, 1).sample(x.shape).to(x.device)).float()
        return x, 0

def create_model(n_caps, cap_dim, mu_init, n_transforms, k_time, k_space):
    s_dim = n_caps * cap_dim
    group_kernel = (k_time, k_space, 1)
    z_encoder = Gaussian_Encoder(Encoder_Chairs(s_dim=s_dim, n_cin=3, n_hw=64),
                                 loc=0.0, scale=1.0)

    u_encoder = Gaussian_Encoder(Encoder_Chairs(s_dim=s_dim, n_cin=3, n_hw=64),                                
                                 loc=0.0, scale=1.0)

    decoder = Gaussian_Decoder(Decoder_Chairs(s_dim=s_dim, n_cout=3, n_hw=64))

    pad_fix_0 = (group_kernel[0]+1) % 2
    pad_fix_1 = (group_kernel[1]+1) % 2
    grouper = Stationary_Capsules_1d(
                      nn.ConvTranspose3d(in_channels=1, out_channels=1,
                                          kernel_size=group_kernel, 
                                          padding=(2*(group_kernel[0] // 2)-pad_fix_0, 
                                                   2*(group_kernel[1] // 2)-pad_fix_1,
                                                   2*(group_kernel[2] // 2)),
                                          stride=(1,1,1), padding_mode='zeros', bias=False),
                      lambda x: F.pad(x, (group_kernel[2] // 2, group_kernel[2] // 2,
                                          group_kernel[1] // 2-pad_fix_1, group_kernel[1] // 2,
                                          group_kernel[0] // 2-pad_fix_0, group_kernel[0] // 2), 
                                          mode='circular'),
                    n_caps=n_caps, cap_dim=cap_dim, n_transforms=n_transforms,
                    mu_init=mu_init)

    
    return TVAE(z_encoder, u_encoder, decoder, grouper)


def main():
    config = {
        'wandb_on': True,
        'lr': 1e-3,
        #'momentum': 0.9,
        'batch_size': 64,
        'max_epochs': 200,
        'eval_epochs': 200,
        #'dataset': 'DSprites',
        #'seq_transforms': ['posX', 'posY', 'orientation', 'scale'],
        #'avail_transforms': ['posX', 'posY', 'orientation', 'scale', 'shape'],
        'seed': 1,
        'n_caps': 8,
        'cap_dim': 16,
        'n_transforms': 8,
        'max_transform_len': 30,
        'mu_init': 30.0,
        'k_time': 4,
        'k_space': 5,
        'n_is_samples': 100
        }

    name = 'Bubbles_Sprites_L=1/4_K=5'

    config['savedir'], config['data_dir'], config['wandb_dir'] = get_dirs()

    savepath = os.path.join(config['savedir'], name)
    #preprocessor = Preprocessor(config)
    #train_loader, val_loader, test_loader = preprocessor.get_dataloaders(batch_size=config['batch_size'])

    data_train = DynBinarizedMNIST(torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/sprites/Sprites_train_torch_131.pt'), map_location="cpu")['data'][:-1000])#[:10000]
    data_val = DynBinarizedMNIST(torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/sprites/Sprites_train_torch_131.pt'), map_location="cpu")['data'][-1000:])#[:5000]
    data_test = DynBinarizedMNIST(torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/sprites/Sprites_test_torch_131.pt'), map_location="cpu")['data'])

    #kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=config['batch_size'], 
                                #sampler=self.train_sampler,
                                #drop_last=True
                                )
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=config['batch_size'], 
                                #sampler=self.valid_sampler,
                                shuffle=False,
                                #drop_last=False
                                )
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=config['batch_size'], 
                                shuffle=False,
                                #drop_last=False
                                )

    model = create_model(n_caps=config['n_caps'], cap_dim=config['cap_dim'], mu_init=config['mu_init'], 
                         n_transforms=config['n_transforms'], k_time=config['k_time'], k_space=config['k_space'])
    model.to('cuda')

    log, checkpoint_path = configure_logging(config, name, model)
    # model.load_state_dict(torch.load(checkpoint_path))

    optimizer = optim.Adam(model.parameters(), 
                           lr=config['lr'],
                           eps=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[50, 100, 125, 150],
            gamma=0.5,
            #verbose=True
        )

    best_total_loss = 100000000000
    best_model = deepcopy(model)

    for e in range(1, config['max_epochs']+1):
        log('Epoch', e)

        total_loss, total_neg_logpx_z, total_kl, total_eq_loss, num_batches = train_epoch(model, optimizer, 
                                                                     train_loader, log,
                                                                     savepath, e, eval_batches=3000,
                                                                     plot_weights=False,
                                                                     plot_fullcaptrav=True,
                                                                     wandb_on=config['wandb_on'])
        total_val_loss, _, _, _, _ = validate_epoch(model=model, val_loader=val_loader, epoch=e)

        if total_val_loss < best_total_loss:
            print("keeping best")
            best_model = deepcopy(model)
            best_total_loss = total_val_loss

        log("Epoch Avg Loss", total_loss / num_batches)
        log("Epoch Avg -LogP(x|z)", total_neg_logpx_z / num_batches)
        log("Epoch Avg KL", total_kl / num_batches)
        log("Epoch Avg EQ Loss", total_eq_loss / num_batches)
        scheduler.step()

        torch.save(model.state_dict(), checkpoint_path)

        nll = []
        
        if e % config['eval_epochs'] == 0:
            for _ in range(5):
                total_loss, total_neg_logpx_z, total_kl, total_is_estimate, total_eq_loss, num_batches = eval_epoch(best_model, test_loader, log, savepath, e, 
                                                                                                                n_is_samples=config['n_is_samples'],
                                                                                                                plot_maxact=False, 
                                                                                                                plot_class_selectivity=False,
                                                                                                                plot_cov=False,
                                                                                                                wandb_on=config['wandb_on'])
                log("Val Avg Loss", total_loss / num_batches)
                log("Val Avg -LogP(x|z)", total_neg_logpx_z / num_batches)
                log("Val Avg KL", total_kl / num_batches)
                log("Val IS Estiamte", total_is_estimate / num_batches)
                log("Val EQ Loss", total_eq_loss / num_batches)
                nll.append((total_is_estimate / num_batches).item())

            log("mean IS Estimate", np.mean(nll))
            log("std IS Estimate", np.std(nll))

if __name__ == '__main__':
    main()