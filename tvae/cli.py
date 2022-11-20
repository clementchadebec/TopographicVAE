import sys
import argparse
from tvae.experiments import (
    tvae_2d_mnist, nontvae_mnist, tvae_L0_mnist, tvae_Lhalf_mnist, 
    tvae_Lshort_mnist, tvae_Lpartial_mnist, bubbles_mnist, 
    tvae_Lpartial_mnist_generalization, 
    tvae_Lpartial_rotcolor_mnist, tvae_Lpartial_perspective_mnist, 
    
    tvae_Lhalf_dsprites, tvae_Lpartial_dsprites, tvae_L0_dsprites,
    tvae_Lshort_dsprites, nontvae_dsprites, bubbles_dsprites, tvae_Lhalf_colormnist,
    tvae_Lhalf_sprites, tvae_Lpartial_sprites, nontvae_sprites, tvae_L0_sprites, tvae_Lshort_sprites,
    tvae_L0_chairs, tvae_Lhalf_chairs, tvae_Lpartial_chairs, tvae_Lshort_chairs
)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str, help='experiment name')

def main():
    args = parser.parse_args()
    module_name = 'tvae.experiments.{}'.format(args.name)
    experiment = sys.modules[module_name]
    experiment.main()

if __name__ == "__main__":
    main()