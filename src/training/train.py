# This script is used to train a YOLO model using the configuration specified in a YAML file.
# It uses the `ultralytics` library to load the model and train it with the provided parameters.
# The configuration file should include the model path, dataset paths, and hyperparameters.

import yaml, argparse
import ultralytics as ultics

def main(cfg):
    cfgd = yaml.safe_load(open(cfg, 'r'))
    ultics.YOLO(cfgd['model']).train(**cfgd)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument('--cfg', required = True)
    main(**vars(p.parse_args()))


