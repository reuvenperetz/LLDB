import argparse
import os
import sys

import options as option

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from lightning.train_lightning import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, required=True, help="Path to option YAML file.")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices (GPUs).")
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)

    run(opt, num_devices=args.devices)


if __name__ == "__main__":
    main()
