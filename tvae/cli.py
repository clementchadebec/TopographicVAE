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
    tvae_L0_chairs, tvae_Lhalf_chairs, tvae_Lpartial_chairs, tvae_Lshort_chairs, bubbles_sprites, bubbles_starmen,
    bubbles_colormnist, bubbles_chairs, tvae_L0_colormnist, tvae_L0_starmen, tvae_Lhalf_starmen, 
    tvae_Lpartial_colormnist, tvae_Lpartial_starmen, tvae_Lshort_colormnist, tvae_Lshort_starmen,
    tvae_L0_rotmnist, tvae_Lhalf_rotmnist, tvae_Lpartial_rotmnist, tvae_Lshort_rotmnist,
    bubbles_rotmnist,

    tvae_L0_rotmnist_missing, tvae_Lhalf_rotmnist_missing, tvae_Lpartial_rotmnist_missing, tvae_Lshort_rotmnist_missing,
    tvae_L0_sprites_missing, tvae_Lhalf_sprites_missing, tvae_Lpartial_sprites_missing, tvae_Lshort_sprites_missing,
    tvae_L0_starmen_missing, tvae_Lpartial_starmen_missing, tvae_Lhalf_starmen_missing, tvae_Lshort_starmen_missing
)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str, help='experiment name')
parser.add_argument(
    "--prob_missing_data",
    type=float,
    default=0.,
    help='The probability of missing data (sequences will have at least 2 data)'
)
parser.add_argument(
    "--prob_missing_pixels",
    type=float,
    default=0.,
    help='The probability of missing pixels in the images'
)

def main():
    args = parser.parse_args()
    module_name = 'tvae.experiments.{}'.format(args.name)
    experiment = sys.modules[module_name]
    experiment.main(args)

if __name__ == "__main__":
    main()