import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', '-dr',
        help='data root directory')
    parser.add_argument('--data-config', '-df',
        help='training data configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    with open(args.data_config, 'r') as file:
        datasets = file.readlines()
        datasets = [dataset.strip() for dataset in datasets]
        datasets = list(filter(lambda d: len(d) > 0, datasets))
        datasets = list(filter(lambda d: 'background' in d, datasets))
        print('background datasets:\n {}'.format(datasets))
    
    for dataset in datasets:
        with open(dataset, 'r') as file:
            image_paths = file.readlines()
            image_paths = [path.strip() for path in image_paths]
            image_paths = list(filter(lambda d: len(d) > 0, image_paths))
            image_paths = [os.path.join(args.data_root, path) for path in image_paths]
        for image_path in image_paths:
            label_path = image_path.replace('.jpg', '.txt').replace('images', 'labels_with_ids')
            with open(label_path, 'w'):
                print('make {}'.format(label_path))