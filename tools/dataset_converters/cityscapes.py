# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add class stats computation

import argparse
import json
import os, types, multiprocessing
import os.path as osp
from tqdm import tqdm

import mmcv
import numpy as np
from cityscapesscripts.preparation.json2labelImg import json2labelImg
from PIL import Image

def scandir_os(path, suffix='_polygons.json'):
    """
    지정한 폴더 `path`와 그 하위 폴더를 재귀적으로 탐색해
    파일명이 `suffix`로 끝나는 모든 파일의 전체 경로를 yield한다.
    예) '/a/b/c/test_polygons.json'
    """
    for root, _, files in os.walk(path):
        for fname in files:
            if fname.endswith(suffix):
                yield os.path.join(root, fname)

def track_progress(func, tasks):
    """mmcv.track_progress 대체 – 단일 프로세스"""
    return [func(t) for t in tqdm(tasks)]

def track_parallel_progress(func, tasks, nproc):
    """mmcv.track_parallel_progress 대체 – 멀티프로세스"""
    with multiprocessing.Pool(nproc) as pool:
        return list(tqdm(pool.imap(func, tasks), total=len(tasks)))

mmcv.scandir                 = scandir_os
mmcv.track_progress           = track_progress
mmcv.track_parallel_progress  = track_parallel_progress

def convert_json_to_label(json_file):
    label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
    json2labelImg(json_file, label_file, 'trainIds')

    if 'train/' in json_file:
        pil_label = Image.open(label_file)
        label = np.asarray(pil_label)
        sample_class_stats = {}
        for c in range(19):
            n = int(np.sum(label == c))
            if n > 0:
                sample_class_stats[int(c)] = n
        sample_class_stats['file'] = label_file
        return sample_class_stats
    else:
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    os.makedirs(out_dir, exist_ok=True)

    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    poly_files = []
    for poly in mmcv.scandir(gt_dir, '_polygons.json'):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    print(len(poly_files))

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_json_to_label, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_json_to_label,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)

    split_names = ['train', 'val', 'test']

    for split in split_names:
        filenames = []
        for poly in mmcv.scandir(
                osp.join(gt_dir, split), '_polygons.json'):
            filenames.append(poly.replace('_gtFine_polygons.json', ''))
        with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames)


if __name__ == '__main__':
    main()