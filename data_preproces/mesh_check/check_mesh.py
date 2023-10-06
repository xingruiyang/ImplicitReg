import argparse
import concurrent.futures
import glob
import json
import os
import random
import subprocess


def find_mesh_in_directory(shape_dir):
    print(shape_dir)
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise ValueError('no mesh file')
    elif len(mesh_filenames) > 1:
        raise ValueError('multiple mesh file')
    return mesh_filenames[0]


def process_mesh(mesh_filepath):
    print('process mesh {}'.format(mesh_filepath))
    command = './bin/PreprocessMesh' + " -m " + mesh_filepath
    try:
        _ = subprocess.check_output(command, shell=True)
    except subprocess.CalledProcessError as _:
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('out_path')
    parser.add_argument('--num_samples', type=int, default=2)
    parser.add_argument('--num_threads', type=int, default=2)
    args = parser.parse_args()

    # split_filenames = [
    #     'sv2_chairs_train.json',
    #     'sv2_lamps_train.json',
    #     'sv2_planes_train.json',
    #     'sv2_sofas_train.json',
    #     'sv2_tables_train.json'
    # ]
    split_filenames = [
        'sv2_chairs_test.json',
        'sv2_lamps_test.json',
        'sv2_planes_test.json',
        'sv2_sofas_test.json',
        'sv2_tables_test.json'
    ]
    for split_file in split_filenames:
        split = json.load(
            open(os.path.join('../splits', split_file), 'r'))
        sample_out = []
        for key, value in split['ShapeNetV2'].items():
            random.shuffle(value)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=int(args.num_threads)
            ) as executor:

                for filename in value:
                    try:
                        filepath = os.path.join(args.data_path, key, filename)
                        mesh_file = find_mesh_in_directory(filepath)

                        if process_mesh(mesh_file):
                            sample_out.append(filename)
                            if len(sample_out) >= args.num_samples:
                                break
                    except ValueError as e:
                        print(str(e))

                executor.shutdown()

        out = dict()
        out[key] = sample_out
        json.dump(out, open(os.path.join(
            args.out_path, split_file), 'w'))
