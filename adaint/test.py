import os
import argparse

import mmcv
import torch
from mmcv.parallel import collate, scatter

from mmedit.apis import init_model
from mmedit.core import tensor2img
from mmedit.datasets.pipelines import Compose
import numpy as np

def enhancement_inference(model, img):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # remove gt from test_pipeline
    keys_to_remove = ['gt', 'gt_path']
    for key in keys_to_remove:
        for pipeline in list(cfg.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                cfg.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    cfg.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)

    test_pipeline = Compose(cfg.test_pipeline)

    data = dict(lq_path=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # inference
    with torch.no_grad():
        result = model(test_mode=True, **data)

    return result['output']


def parse_args():
    parser = argparse.ArgumentParser(description='Enhancement demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_path', help='path to input **image folder**')
    parser.add_argument('save_path', help='path to **output folder**')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.img_path):
        raise ValueError('img_path must be a folder')

    os.makedirs(args.save_path, exist_ok=True)

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    img_list = sorted([
        f for f in os.listdir(args.img_path)
        if os.path.splitext(f)[1].lower() in exts
    ])

    if len(img_list) == 0:
        raise ValueError("No images!")

    print(f"Find {len(img_list)} images,begin to process...")

    for fname in img_list:
        in_path = os.path.join(args.img_path, fname)
        out_path = os.path.join(args.save_path, fname)

        output = enhancement_inference(model, in_path)
        output = tensor2img(output, out_type=np.uint16)
        output = output[:, :, ::-1]

        mmcv.imwrite(output, out_path)
        print(f"Save in: {out_path}")

    print("All finished")


if __name__ == '__main__':
    main()
