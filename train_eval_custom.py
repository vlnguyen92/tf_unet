from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from pathlib import Path

import matplotlib.pyplot as plt
import argparse


def build_generator():
    nx = 572
    ny = 572

    generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=20)

    return generator


def train_mode(model, generator, dir):

    trainer = unet.Trainer(model, optimizer='momentum', opt_kwargs=dict(momentum=0.2))
    _ = trainer.train(generator, dir, training_iters=10, epochs=2, display_step=2)


def eval_mode(model, generator, dir):
    x_test, y_test = generator(1)
    #_ = model.predict(dir + '/model.ckpt', x_test)

    print(model.evaluate(dir + '/model.ckpt', x_test, y_test))


def main(args):
    curr_mode = args.mode

    generator = build_generator()

    net = unet.Unet(channels=generator.channels, n_class=generator.n_class, layers=3, features_root=16)

    if curr_mode == 'train':
        train_mode(net, generator, args.dir)
    elif curr_mode == 'eval':
        eval_mode(net, generator, args.dir)
    else:
        print('Mode not supported')
        return -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', default='train', help='eval or train')
    parser.add_argument('-d', '--dir', default='trained_models', help='Folder to save/load models')

    args = parser.parse_args()

    main(args)
