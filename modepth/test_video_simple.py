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
import cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from manydepth import networks
from layers import transformation_from_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for ManyDepth models.')

    parser.add_argument('--target_image_path', type=str,
                        help='path to a test image to predict for', default="../assets/test_sequence_target.jpg")
    parser.add_argument('--source_image_path', type=str,
                        help='path to a previous image in the video sequence',
                        default="../assets/test_sequence_source.jpg")
    parser.add_argument('--intrinsics_json_path', type=str,
                        help='path to a json file containing a normalised 3x3 intrinsics matrix',
                        default="../assets/test_sequence_intrinsics.json")
    parser.add_argument('--video_path', type=str,
                        help='path to a video file',
                        default="")
    parser.add_argument('--model_path', type=str,
                        help='path to a folder of weights to load',
                        default="")
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

def load_and_preprocess_image_video(image0, image1, resize_width, resize_height):
    original_width, original_height = image0.size
    image0 = image0.resize((resize_width, resize_height), pil.LANCZOS)
    image1 = image1.resize((resize_width, resize_height), pil.LANCZOS)
    image0 = transforms.ToTensor()(image0).unsqueeze(0)
    image1 = transforms.ToTensor()(image1).unsqueeze(0)
    if torch.cuda.is_available():
        return image0.cuda(), image1.cuda(), (original_height, original_width)
    return image0, image1, (original_height, original_width)


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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    # depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    depth_decoder = networks.DepthDecoderUnet()
    loaded_dict = torch.load(os.path.join(args.model_path, "depth.pth"), map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

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
    pose_enc.eval()
    pose_dec.eval()
    if torch.cuda.is_available():
        encoder.cuda()
        depth_decoder.cuda()
        pose_enc.cuda()
        pose_dec.cuda()

    cap = cv2.VideoCapture(args.video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    ret0, frame0 = cap.read()

    # Load input data
    # input_image, original_size = load_and_preprocess_image(args.target_image_path,
    #                                                        resize_width=encoder_dict['width'],
    #                                                        resize_height=encoder_dict['height'])
    #
    # source_image, _ = load_and_preprocess_image(args.source_image_path,
    #                                             resize_width=encoder_dict['width'],
    #                                             resize_height=encoder_dict['height'])
    K, invK = load_and_preprocess_intrinsics(args.intrinsics_json_path,
                                             resize_width=encoder_dict['width'],
                                             resize_height=encoder_dict['height'])

    while True:

        ret1, frame1 = cap.read()
        if ret1:
            
            image0 = Image.fromarray(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))  
            image1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

            # Load input data
            input_image, source_image, original_size = load_and_preprocess_image_video(image0, image1,
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

                output = depth_decoder(output)

                sigmoid_output = output[("disp", 0)]
                sigmoid_output_resized = torch.nn.functional.interpolate(
                    sigmoid_output, original_size, mode="bilinear", align_corners=False)
                sigmoid_output_resized = sigmoid_output_resized.cpu().numpy()[:, 0]

                # Saving numpy file
                # directory, filename = os.path.split(args.target_image_path)
                # output_name = os.path.splitext(filename)[0]
                # name_dest_npy = os.path.join(directory, "{}_disp_{}.npy".format(output_name, args.mode))
                # np.save(name_dest_npy, sigmoid_output.cpu().numpy())

                # # Saving depth file
                # depth_path = os.path.join(directory, "{}_{}_{}.jpeg".format(output_name, 'depth', args.mode))
                # # depth = 1 / (sigmoid_output_resized.squeeze().cpu().numpy() * (max_disp - min_disp) + min_disp) * SCALE
                # depth = 1 / sigmoid_output_resized
                # depth = pil.fromarray(np.uint8(depth.squeeze()))
                # depth.save(depth_path)

                # Saving colormapped depth image and cost volume argmin
                toplot = sigmoid_output_resized.squeeze()
                normalizer = mpl.colors.Normalize(vmin=toplot.min(), vmax=np.percentile(toplot, 95))
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(toplot)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)
                # arr = np.array(im)
                # mat = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)  
                # name_dest_im = os.path.join(directory,
                #                             "{}_{}_{}.jpeg".format(output_name, plot_name, args.mode))
                # im.save(name_dest_im)
                writer.write(img)
                # print("-> Saved output image to {}".format(name_dest_im))
            frame0 = frame1
            cv2.imshow('demo', img)

            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # VideoCapture
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
