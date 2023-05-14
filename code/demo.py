import argparse
import tkinter as tk
import torch
import options.options as option
from isegm.utils import exp
# from isegm.inference import utils
from interactive_demo.app import InteractiveDemoApp
from data import create_dataset, create_dataloader
from models import create_model

def main():
    args, opt = parse_args()

    torch.backends.cudnn.deterministic = True
    model = create_model(opt)

    root = tk.Tk()
    root.minsize(960, 480)
    app = InteractiveDemoApp(root, args, model)
    root.deiconify()
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use only CPU for inference.')

    parser.add_argument('--limit-longest-size', type=int, default=800,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')

    parser.add_argument('--opt', type=str, default="./options/test/inter/c_s3_joint.yml",
                        help='The path to the config file.')

    args = parser.parse_args()
    if args.cpu:
        args.device =torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')

    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    return args, opt




if __name__ == '__main__':
    main()
