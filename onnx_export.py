"""
ONNX export script
Export PyTorch models as ONNX graphs.
This export script originally started as an adaptation of code snippets found at
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

The default parameters work with PyTorch 2.0.1 and ONNX 1.13 and produce an optimal ONNX graph
for hosting in the ONNX runtime (see onnx_validate.py). To export an ONNX model compatible
"""

import argparse
import torch
import numpy as np
import onnx
import models
from timm.models import create_model



## python onnx_export.py --model vit_base_patch16_224 ./vit_base_patch16_224.onnx

parser = argparse.ArgumentParser(description='PyTorch ONNX Deployment')
parser.add_argument('--output', metavar='ONNX_FILE', default=None, type=str,
                    help='output model filename')

# Model & datasets params
parser.add_argument('--model', default='repnext_m1', type=str, metavar='MODEL',
                        choices=['repnext_m1', 'repnext_m2', 'repnext_m3', 'repnext_m4', 'repnext_m5'],
                    help='model architecture (default: repnext_m1)')
parser.add_argument('--checkpoint', default='./output/repnext_m1_best_checkpoint.pth', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')
parser.add_argument('--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--nb-classes', type=int, default=5,
                    help='Number classes in datasets')

parser.add_argument('--opset', type=int, default=10,
                    help='ONNX opset to use (default: 10)')
parser.add_argument('--keep-init', action='store_true', default=False,
                    help='Keep initializers as input. Needed for Caffe2 compatible export in newer PyTorch/ONNX.')
parser.add_argument('--aten-fallback', action='store_true', default=False,
                    help='Fallback to ATEN ops. Helps fix AdaptiveAvgPool issue with Caffe2 in newer PyTorch/ONNX.')
parser.add_argument('--dynamic-size', action='store_true', default=False,
                    help='Export model width dynamic width/height. Not recommended for "tf" models with SAME padding.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of datasets')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of datasets')




def main():
    args = parser.parse_args()

    # args.pretrained = True
    # if args.checkpoint:
    #     args.pretrained = False

    if args.output == None:
        args.output = f'./{args.model}.onnx'

    print("==> Creating PyTorch {} model".format(args.model))
    # NOTE exportable=True flag disables autofn/jit scripted activations and uses Conv2dSameExport layers
    # for models using SAME padding
    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        # exportable=True
    )

    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.eval()

    example_input = torch.randn((args.batch_size, 3, args.img_size or 224, args.img_size or 224), requires_grad=True)

    # Run model once before export trace, sets padding for models with Conv2dSameExport. This means
    # that the padding for models with Conv2dSameExport (most models with tf_ prefix) is fixed for
    # the input img_size specified in this script.
    # Opset >= 11 should allow for dynamic padding, however I cannot get it to work due to
    # issues in the tracing of the dynamic padding or errors attempting to export the model after jit
    # scripting it (an approach that should work). Perhaps in a future PyTorch or ONNX versions...
    model(example_input)

    print("==> Exporting model to ONNX format at '{}'".format(args.output))
    input_names = ["input0"]
    output_names = ["output0"]
    dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}
    if args.dynamic_size:
        dynamic_axes['input0'][2] = 'height'
        dynamic_axes['input0'][3] = 'width'
    if args.aten_fallback:
        export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    else:
        export_type = torch.onnx.OperatorExportTypes.ONNX

    torch_out = torch.onnx._export(
        model, example_input, args.output, export_params=True, verbose=True, input_names=input_names,
        output_names=output_names, keep_initializers_as_inputs=args.keep_init, dynamic_axes=dynamic_axes,
        opset_version=args.opset, operator_export_type=export_type)

    print("==> Loading and checking exported model from '{}'".format(args.output))
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)  # assuming throw on error
    print("==> Passed")

    if args.keep_init and args.aten_fallback:
        import caffe2.python.onnx.backend as onnx_caffe2
        # Caffe2 loading only works properly in newer PyTorch/ONNX combos when
        # keep_initializers_as_inputs and aten_fallback are set to True.
        print("==> Loading model into Caffe2 backend and comparing forward pass.".format(args.output))
        caffe2_backend = onnx_caffe2.prepare(onnx_model)
        B = {onnx_model.graph.input[0].name: x.data.numpy()}
        c2_out = caffe2_backend.run(B)[0]
        np.testing.assert_almost_equal(torch_out.data.numpy(), c2_out, decimal=5)
        print("==> Passed")


if __name__ == '__main__':
    main()