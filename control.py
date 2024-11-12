import argparse
from glob import glob
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--func', choices=['remove_results'])
parser.add_argument('--model')
parser.add_argument('--dataset')
parser.add_argument('--method')
parser.add_argument('--shots')

ICL_DIR = '/data1/ay0119/icl/'

def remove_results(_model, _dataset, _method, shot):
    result_dir = f'{ICL_DIR}/results/{_model}/{_dataset}'

    target_files = glob(f'{result_dir}/{_method}*_{shot}_shot.json')

    for target_file in target_files:
        subprocess.run(['rm', target_file])
        print(f'Removed {target_file}')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.func == 'remove_results':
        remove_results(args.model, args.dataset, args.method, args.shots)
