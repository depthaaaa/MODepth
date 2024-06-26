# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import json
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

import networks
# from ..layers import transformation_from_parameters

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for ManyDepth models.')

    parser.add_argument('--target_image_path', type=str,
                        default="",
                        help='path to a test image to predict for', required=False)
    parser.add_argument('--source_image_path', type=str,
                        default="",
                        help='path to a previous image in the video sequence', required=False)
    parser.add_argument('--intrinsics_json_path', type=str,
                        default="../assets/test_sequence_intrinsics.json",
                        help='path to a json file containing a normalised 3x3 intrinsics matrix',
                        required=False)
    parser.add_argument('--model_path', type=str,
                        default="",
                        help='path to a folder of weights to load', required=False)
    parser.add_argument('--mode', type=str, default='multi', choices=('multi', 'mono'),
                        help='"multi" or "mono". If set to "mono" then the network is run without '
                             'the source image, e.g. as described in Table 5 of the paper.',
                        required=False)
    return parser.parse_args()


def load_and_preprocess_image(image_path, resize_width, resize_height):
    image = pil.open(image_path).convert('RGB')
    original_width, original_height = image.size
    image = image.resize((resize_width, resize_height), pil.LANCZOS)
    image = transforms.ToTensor()(image).unsqueeze(0)
    if torch.cuda.is_available():
        return image.cuda(), (original_height, original_width)
    return image, (original_height, original_width)


def load_and_preprocess_intrinsics(intrinsics_path, resize_width, resize_height):
    K = np.eye(4)
    with open(intrinsics_path, 'r') as f:
        K[:3, :3] = np.array(json.load(f))

    # Convert normalised intrinsics to 1/4 size unnormalised intrinsics.
    # (The cost volume construction expects the intrinsics corresponding to 1/4 size images)
    K[0, :] *= resize_width // 4
    K[1, :] *= resize_height // 4

    invK = torch.Tensor(np.linalg.pinv(K)).unsqueeze(0)
    K = torch.Tensor(K).unsqueeze(0)

    if torch.cuda.is_available():
        return K.cuda(), invK.cuda()
    return K, invK


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_path is not None, \
        "You must specify the --model_path parameter"

    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    print("-> Loading model from ", args.model_path)

    # Loading pretrained model
    print("   Loading pretrained encoder")
    

    encoder_dict = torch.load(os.path.join(args.model_path, "encoder.pth"), map_location=device)
    encoder = networks.ResnetEncoderMatching(18, False,
                                             input_width=encoder_dict['width'],
                                             input_height=encoder_dict['height'],
                                             adaptive_bins=True,
                                             min_depth_bin=encoder_dict['min_depth_bin'],
                                             max_depth_bin=encoder_dict['max_depth_bin'],
                                             depth_binning='linear',
                                             num_depth_bins=96)

    filtered_dict_enc = {k: v for k, v in encoder_dict.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    print("   Loading pretrained decoder")
    # depth_decoder = networks.DepthDecoderUnet([0, 1, 2, 3])
    depth_decoder = networks.DepthDecoderUnet([0, 1, 2], in_channels=[64, 128, 192, 256], bins=64)
    
    # depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(os.path.join(args.model_path, "depth.pth"), map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    bins = networks.AdaPlanes(in_channels=64, embedding_dim=64 // 2, min_val=0.001, max_val=80.0)
    bins_path = torch.load(os.path.join(args.model_path, "bins.pth"), map_location=device)
    bins.load_state_dict(bins_path)

    print("   Loading pose network")
    pose_enc_dict = torch.load(os.path.join(args.model_path, "pose_encoder.pth"),
                               map_location=device)
    pose_dec_dict = torch.load(os.path.join(args.model_path, "pose.pth"), map_location=device)

    pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
    pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                    num_frames_to_predict_for=2)

    pose_enc.load_state_dict(pose_enc_dict, strict=True)
    pose_dec.load_state_dict(pose_dec_dict, strict=True)

    # Setting states of networks
    encoder.eval()
    depth_decoder.eval()
    bins.eval()
    pose_enc.eval()
    pose_dec.eval()
    if torch.cuda.is_available():
        encoder.cuda()
        depth_decoder.cuda()
        bins.cuda()
        pose_enc.cuda()
        pose_dec.cuda()

    # Load input data
    input_image, original_size = load_and_preprocess_image(args.target_image_path,
                                                           resize_width=encoder_dict['width'],
                                                           resize_height=encoder_dict['height'])

    source_image, _ = load_and_preprocess_image(args.source_image_path,
                                                resize_width=encoder_dict['width'],
                                                resize_height=encoder_dict['height'])

    K, invK = load_and_preprocess_intrinsics(args.intrinsics_json_path,
                                             resize_width=encoder_dict['width'],
                                             resize_height=encoder_dict['height'])

    with torch.no_grad():

        # Estimate poses
        pose_inputs = [source_image, input_image]
        pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
        axisangle, translation = pose_dec(pose_inputs)
        pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)

        if args.mode == 'mono':
            pose *= 0  # zero poses are a signal to the encoder not to construct a cost volume
            source_image *= 0

        # Estimate depth
        output, lowest_cost, _ = encoder(current_image=input_image,
                                         lookup_images=source_image.unsqueeze(1),
                                         poses=pose.unsqueeze(1),
                                         K=K,
                                         invK=invK,
                                         min_depth_bin=encoder_dict['min_depth_bin'],
                                         max_depth_bin=encoder_dict['max_depth_bin'])
        
                                         
        depth_decoder.eval()

        output = depth_decoder(output)

        bins.eval()

        output = bins(output)

        sigmoid_output = output[("disp", 0)]
        sigmoid_output_resized = torch.nn.functional.interpolate(
            sigmoid_output, original_size, mode="bilinear", align_corners=False)
        sigmoid_output_resized = sigmoid_output_resized.cpu().numpy()[:, 0]

        # Saving numpy file
        # directory, filename = os.path.split(args.target_image_path)
        directory, filename = "/media/lab280/F/zzt/depthplane/assets/", "depth"
        output_name = os.path.splitext(filename)[0]
        # name_dest_npy = os.path.join(directory, "{}_disp_{}.npy".format(output_name, args.mode))
        # np.save(name_dest_npy, sigmoid_output.cpu().numpy())

        # Saving colormapped depth image and cost volume argmin
        for plot_name, toplot in (('costvol_min', lowest_cost), ('disp', sigmoid_output_resized)):
            toplot = toplot.squeeze()
            normalizer = mpl.colors.Normalize(vmin=toplot.min(), vmax=np.percentile(toplot, 95))
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(toplot)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(directory,
                                        "{}_{}_{}.jpeg".format(output_name, plot_name, args.mode))
            im.save(name_dest_im)

            print("-> Saved output image to {}".format(name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    args.target_image_path = ""
    args.source_image_path = ""
    args.mode = "mono"  # "mono"
    test_simple(args)
