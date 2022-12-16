import os
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.nn import functional as F
from copy import deepcopy

from tvae.data.mnist import Preprocessor
from tvae.containers.tvae import VAE
from tvae.models.mlp import Encoder_Chairs, Decoder_Chairs
from tvae.containers.encoder import Gaussian_Encoder
from tvae.containers.decoder import Gaussian_Decoder
from tvae.containers.grouper import NonTopographic_Capsules1d
from tvae.utils.logging import configure_logging, get_dirs
from tvae.utils.train_loops import train_epoch, eval_epoch, validate_epoch


class DynBinarizedMNIST(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        #x = (x > torch.distributions.Uniform(0, 1).sample(x.shape).to(x.device)).float()
        return x, 0

def create_model(n_caps, cap_dim, n_transforms):
    s_dim = n_caps * cap_dim
    group_kernel = (1,1,1)
    z_encoder = Gaussian_Encoder(Encoder_Chairs(s_dim=s_dim, n_cin=3, n_hw=64),
                                 loc=0.0, scale=1.0)

    u_encoder = None

    decoder = Gaussian_Decoder(Decoder_Chairs(s_dim=s_dim, n_cout=3, n_hw=64))

    grouper = NonTopographic_Capsules1d(
                    None,#nn.ConvTranspose3d(in_channels=1, out_channels=1,
                        #                  kernel_size=group_kernel, 
                        #                  padding=(0,0,0),
                        #                  stride=(1,1,1), padding_mode='zeros', bias=False),
                    None,#lambda x: F.pad(x, (0,0,0,0,0,0), mode='circular'),
                    n_caps=n_caps, cap_dim=cap_dim, n_transforms=n_transforms)
    
    return VAE(z_encoder, u_encoder, decoder, grouper)


def main():
    config = {
        'wandb_on': True,
        'lr': 1e-3,
        #'momentum': 0.9,
        'batch_size': 64,
        'max_epochs': 200,
        'eval_epochs': 20,
        #'dataset': 'MNIST',
        #'train_angle_set': '0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340',
        #'test_angle_set': '0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340', 
        #'train_color_set': '0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340',
        #'test_color_set': '0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340',
        #'train_scale_set': '0.60 0.64 0.68 0.72 0.76 0.79 0.83 0.87 0.91 0.95 0.99 1.03 1.07 1.11 1.14 1.18 1.22 1.26',
        #'test_scale_set': '0.60 0.64 0.68 0.72 0.76 0.79 0.83 0.87 0.91 0.95 0.99 1.03 1.07 1.11 1.14 1.18 1.22 1.26',
        #'pct_val': 0.2,
        #'random_crop': 28,
        'seed': 1,
        'n_caps': 8,
        'cap_dim': 2,
        'n_transforms': 8,
        'train_eq_loss': False,
        'n_is_samples': 100
        }

    name = 'NonT-VAE_MNIST_L=0'

    config['savedir'], config['data_dir'], config['wandb_dir'] = get_dirs()

    savepath = os.path.join(config['savedir'], name)
    #preprocessor = Preprocessor(config)
    #train_loader, val_loader, test_loader = preprocessor.get_dataloaders(batch_size=config['batch_size'])

    data_train = DynBinarizedMNIST(torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/sprites/Sprites_train_torch_131.pt'), map_location="cpu")['data'][:-1000])#[:10000]
    data_val = DynBinarizedMNIST(torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/sprites/Sprites_train_torch_131.pt'), map_location="cpu")['data'][-1000:])#[:5000]
    data_test = DynBinarizedMNIST(torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/sprites/Sprites_test_torch_131.pt'), map_location="cpu")['data'])

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


    model = create_model(n_caps=config['n_caps'], cap_dim=config['cap_dim'], n_transforms=config['n_transforms'])
    model.to('cuda')

    print(model, config)

    log, checkpoint_path = configure_logging(config, name, model)
    # model.load_state_dict(torch.load(load_checkpoint_path))

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

        if e % config['eval_epochs'] == 0:
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


if __name__ == '__main__':
    main()