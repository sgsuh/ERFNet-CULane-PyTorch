"""
Create: 2021.10.12
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import torch
import onnx
import argparse 

from models.erfnet import LaneNet

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder_path",
                        default="")
    parser.add_argument("--est_decoder_path",
                        default="")
    parser.add_argument("--seg_decoder_path",
                        default="")
    parser.add_argument("--onnx_path",
                        default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    num_class = [5, 4]
    encoder_path = args.encoder_path
    est_decoder_path = args.est_decoder_path
    seg_decoder_path = args.seg_decoder_path

    model = LaneNet(num_class)

    model.encoder.load_state_dict(torch.load(encoder_path)['state_dict'])
    model.est_decoder.load_state_dict(torch.load(est_decoder_path)['state_dict'])
    model.seg_decoder.load_state_dict(torch.load(seg_decoder_path)['state_dict'])

    model.cuda()

    model.requires_grad_(False)
    model.eval()

    dummy_input = torch.randn(2, 272, 960, 3).cuda()

    onnx_path = args.onnx_path

    torch.onnx.export(model, dummy_input, onnx_path, export_params = True, opset_version = 9, input_names = ['erfnet_input'], output_names = ['estimation', 'segmentation'], verbose = True)

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)