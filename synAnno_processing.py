'''
                                                *********   Project: SynAnno (Harvard VCG)   ********

Temporary- directory structure. (Deleted after JSON is created)
        .
        |__Images
        |__Syn_Mid ;        Img_Mid ;       Before ;         After
            |__ GT-Images      |__EM-Images   |__Syn ; Img      |__Syn ; Img
                (Synapse_Idx)  (Synapse_Idx)      |_____|           |_____|
                (Mid_Slice)     (Mid_Slice)          |                 |
                                                  Synapse_Idx       Synapse_Idx
                                                  ___|___           ___|___
                                                  |      |          |      |
                                                  GT     EM         GT     EM
                                               (Previos Slices)  (Subsequent Slices)           #Both with reference to Middle Slice.

** While creating the figures, the images in Before/Syn and After/Syn are substituted by figures.

---------------------------------------------------------------------------------------------------------

Ouput JSON structure.

{
    "Data": [
        {
            "Name": "1.png",       # synpase_idx.png
            "EM": base64(EM = 7) ,     # Middle Slice (EM/Img) suppose ; mid_slice_idx = 7 & z-axis range = 0-14
            "GT": base64(GT = 7) ,     # Middle Slice (GT/Syn) suppose ; mid_slice_idx = 7 & z-axis range = 0-14
            "Before": [base64(6),base64(5),base64(4),base64(3),base64(2),base64(1),base64(0)],
            "After": [base64(7),base64(8),base64(9),base64(10),base64(11),base64(12),base64(13),base64(14)],
            "Label": "Correct"
        },
        {
            "Name": "10.png",
            "EM": ,
            "GT": ,
            "Before": [],
            "After": [],
            "Label": "Correct"
        },

        *** Similarly for all other synapses ***

            ]
}
'''

# Importing the required libraries
import itertools
import numpy as np
from skimage.morphology import binary_dilation, remove_small_objects
from skimage.measure import label as label_cc
from pytorch_connectomics.connectomics.data.utils import readvol
from matplotlib import pyplot as plt
import cv2
import os, sys
import io
from util import bfly, rotateIm
import PIL
from PIL import Image
import json
import util
import base64
import h5py
from matplotlib.pyplot import imread, imsave
import shutil
from scipy.stats import linregress


# Processing the synpases using binary dilation as well as by removing small objects.
def process_syn(gt, small_thres=16):
    seg = binary_dilation(gt.copy() != 0)
    seg = label_cc(seg).astype(int)
    seg = seg * (gt.copy() != 0).astype(int)
    seg = remove_small_objects(seg, small_thres)

    c2 = (gt.copy() == 2).astype(int)
    c1 = (gt.copy() == 1).astype(int)

    syn_pos = np.clip((seg * 2 - 1), a_min=0, a_max=None) * c1
    syn_neg = (seg * 2) * c2
    syn = np.maximum(syn_pos, syn_neg)

    return syn, seg


# Calculating the bounding boxes for every synpase in N-dimensions (3D as well as 2D)
def bbox2_ND(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


def bbox_adjust(low, high, sz):
    assert high >= low
    bbox_sz = high - low
    diff = abs(sz - bbox_sz) // 2
    if bbox_sz >= sz:
        return low + diff, low + diff + sz

    return low - diff, low - diff + sz


# Centering the synapse.
def crop_pad_data(data, z, bbox_2d, pad_val=0, mask=None):
    sz = data.shape[1:]
    y1o, y2o, x1o, x2o = bbox_2d  # region to crop
    y1m, y2m, x1m, x2m = 0, sz[0], 0, sz[1]
    y1, x1 = max(y1o, y1m), max(x1o, x1m)
    y2, x2 = min(y2o, y2m), min(x2o, x2m)
    cropped = data[z, y1:y2, x1:x2]

    if mask is not None:
        mask_2d = mask[z, y1:y2, x1:x2]
        cropped = cropped * (mask_2d != 0).astype(cropped.dtype)

    pad = ((y1 - y1o, y2o - y2), (x1 - x1o, x2o - x2))
    if not all(v == 0 for v in pad):
        cropped = np.pad(cropped, pad, mode='constant',
                         constant_values=pad_val)
    return cropped


# Converting to RGB
def syn2rgb(label):
    tmp = [None] * 3
    tmp[0] = np.logical_and((label % 2) == 1, label > 0)
    tmp[1] = np.logical_and((label % 2) == 0, label > 0)
    tmp[2] = (label > 0)
    out = np.stack(tmp, 0).astype(np.uint8)
    return (out * 255).transpose(1, 2, 0)


# Creating the directories and returning their paths.
def dir_creator(parent_dir_path, dir_name):
    if os.path.exists(os.path.join(parent_dir_path, dir_name)):
        pass
    else:
        os.mkdir(os.path.join(parent_dir_path, dir_name))
    return os.path.join(parent_dir_path, dir_name)


def visualize(syn, seg, img, sz=142, rgb=False):
    item_list = []

    final_file = dict()
    seg_idx = np.unique(seg)[1:]  # ignore background

    # Creating the temporary-directory structure for storing images.
    idx_dir = dir_creator('.', 'Images')
    syn_mid, img_mid = dir_creator(idx_dir, 'Syn_Mid'), dir_creator(idx_dir, 'Img_Mid')
    before, after = dir_creator(idx_dir, 'Before'), dir_creator(idx_dir, 'After')
    syn_before, img_before = dir_creator(before, 'Syn'), dir_creator(before, 'Img')
    syn_after, img_after = dir_creator(after, 'Syn'), dir_creator(after, 'Img')

    # Processing and iterating over the synapses, subsequently saving the middle slices and before/after slices for 3D navigation.
    for idx in seg_idx:

        # Creating directories for every synapse in Before/After directories.
        syn_nav_before, img_nav_before = dir_creator(syn_before, str(idx)), dir_creator(img_before, str(idx))
        syn_nav_after, img_nav_after = dir_creator(syn_after, str(idx)), dir_creator(img_after, str(idx))

        item = dict()
        temp = (seg == idx)
        bbox = bbox2_ND(temp)

        z_mid = (bbox[0] + bbox[1]) // 2
        temp_2d = temp[z_mid]
        bbox_2d = bbox2_ND(temp_2d)
        y1, y2 = bbox_adjust(bbox_2d[0], bbox_2d[1], sz)
        x1, x2 = bbox_adjust(bbox_2d[2], bbox_2d[3], sz)
        crop_2d = [y1, y2, x1, x2]
        cropped_syn = crop_pad_data(syn, z_mid, crop_2d, mask=temp)
        cropped_img = crop_pad_data(img, z_mid, crop_2d, pad_val=128)

        if rgb:
            cropped_syn = syn2rgb(cropped_syn)

        assert cropped_syn.shape == (sz, sz, 3) or cropped_syn.shape == (sz, sz)
        plt.imsave(os.path.join(syn_mid, str(idx) + '.png'), cropped_syn, cmap='gray')
        plt.imsave(os.path.join(img_mid, str(idx) + '.png'), cropped_img, cmap='gray')

        # Saving before and after slices for 3D navigation.
        before = [x for x in range(bbox[0], z_mid)]
        after = [x for x in range(z_mid, bbox[1] + 1)]
        before_processed_img = []
        after_processed_img = []

        # Before
        for navimg in before:

            temp_2d = temp[navimg]
            bbox_2d = bbox2_ND(temp_2d)
            y1, y2 = bbox_adjust(bbox_2d[0], bbox_2d[1], sz)
            x1, x2 = bbox_adjust(bbox_2d[2], bbox_2d[3], sz)
            crop_2d = [y1, y2, x1, x2]
            cropped_img = crop_pad_data(img, navimg, crop_2d, pad_val=128)
            cropped_syn = crop_pad_data(syn, navimg, crop_2d, mask=temp)

            if rgb:
                cropped_syn = syn2rgb(cropped_syn)
                param = 0.79
                cropped_im_dark = np.stack((cropped_img * param, cropped_img * param, cropped_img * param), axis=2)
                cropped_img = np.stack((cropped_img, cropped_img, cropped_img), axis=2)
                cropped_im_dark = cropped_im_dark.astype(np.uint8)
                cropped_syn = np.maximum(cropped_im_dark, cropped_syn)

            assert cropped_syn.shape == (sz, sz, 3) or cropped_syn.shape == (sz, sz)
            plt.imsave(os.path.join(syn_nav_before, str(navimg) + '.png'), cropped_syn, cmap='gray')
            plt.imsave(os.path.join(img_nav_before, str(navimg) + '.png'), cropped_img, cmap='gray')

        # After
        for navimg in after:
            temp_2d = temp[navimg]
            bbox_2d = bbox2_ND(temp_2d)
            y1, y2 = bbox_adjust(bbox_2d[0], bbox_2d[1], sz)
            x1, x2 = bbox_adjust(bbox_2d[2], bbox_2d[3], sz)
            crop_2d = [y1, y2, x1, x2]
            cropped_img = crop_pad_data(img, navimg, crop_2d, pad_val=128)
            cropped_syn = crop_pad_data(syn, navimg, crop_2d, mask=temp)

            if rgb:
                cropped_syn = syn2rgb(cropped_syn)
                param = 0.79
                cropped_im_dark = np.stack((cropped_img * param, cropped_img * param, cropped_img * param), axis=2)
                cropped_img = np.stack((cropped_img, cropped_img, cropped_img), axis=2)
                cropped_im_dark = cropped_im_dark.astype(np.uint8)
                cropped_syn = np.maximum(cropped_im_dark, cropped_syn)

            assert cropped_syn.shape == (sz, sz, 3) or cropped_syn.shape == (sz, sz)
            plt.imsave(os.path.join(syn_nav_after, str(navimg) + '.png'), cropped_syn, cmap='gray')
            plt.imsave(os.path.join(img_nav_after, str(navimg) + '.png'), cropped_img, cmap='gray')

        # Rotating the images based on given rotation parameters (Used to rotate before/after slices using the mid slice rotation parameters)


def rotate(syn, syn_path, syn_name, angle, pt_m):
    rot_syn = rotateIm(syn, angle, pt_m)
    imsave(syn_path + str(syn_name), rot_syn)


# Calculating the rotation angle, rotating and saving the middle slices. (Also, calling rotate() to rotate corresponding before/after slices.)
def rot(syn, syn_path, im, img_path, img_name, thres_dilation=5, a=0.79):
    # 3. compute rotation
    # compute rotatation by cleft
    # cleft: overlap for the dilated pre-/post-partner
    dilation_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thres_dilation, thres_dilation))
    cleft = cv2.dilate((syn > 0).astype(np.uint8), dilation_mask)
    if cleft.max() == 0:
        cleft = syn > 0
    pt = np.where(cleft > 0)
    if pt[0].min() == pt[0].max():
        w = 100;
        w2 = 0
        angle = 90
    else:
        if pt[1].min() == pt[1].max():
            w = 0
            angle = 0
        else:
            # angle concensus
            # pt[0]: x
            # pt[1]: y
            w, _, _, _, _ = linregress(pt[0], pt[1])
            angle = np.arctan(w) / np.pi * 180
            w2, _, _, _, _ = linregress(pt[1], pt[0])
            angle2 = np.arctan(w2) / np.pi * 180
            # if abs((angle+angle2)-90)>20:
            # trust the small one
            if abs(angle2) < abs(angle):
                angle = np.sign(angle2) * (90 - abs(angle2))
                w = 1 / w2

    # pre-post direction
    r1 = np.where(syn == 128)
    r2 = np.where(syn == 255)
    if len(r1[0]) == 0:
        r1 = r2
    if len(r2[0]) == 0:
        r2 = r1

    if abs(w) < 0.2:  # vertical bar, use w
        if abs(w) > 1e-4:
            diff = (r2[1] - w * r2[0]).mean() - (r1[1] - w * r1[0]).mean()
        else:  # almost 0
            diff = r2[1].mean() - r1[1].mean()
    else:  # horizontal bar, use w2
        diff = -w2 * ((r2[0] - w2 * r2[1]).mean() - (r1[0] - w2 * r1[1]).mean())
    # print bid,w,diff
    if diff < 0:
        angle = angle - 180
    pt_m = np.array([pt[1].mean(), pt[0].mean()])

    # re-center
    rot_im = rotateIm(im, -angle, tuple(pt_m))
    imsave(img_path + str(img_name), rot_im)

    rot_im = rot_im * a
    rot_syn = rotateIm(syn, -angle, tuple(pt_m))

    composite_image = np.maximum(rot_im, rot_syn)
    imsave(syn_path + str(img_name), composite_image)

    # Before
    dir_idx = str(img_name.strip('.png'))
    syn_path_before = './Images/Before/Syn/' + dir_idx + '/'
    img_path_before = './Images/Before/Img/' + dir_idx + '/'

    for idx, img_name in enumerate(os.listdir(syn_path_before)):
        syn = plt.imread(syn_path_before + str(img_name))
        rotate(syn, syn_path_before, img_name, -angle, tuple(pt_m))

    for idx, img_name in enumerate(os.listdir(img_path_before)):
        img = plt.imread(img_path_before + str(img_name))
        rotate(img, img_path_before, img_name, -angle, tuple(pt_m))

    # After
    syn_path_after = './Images/After/Syn/' + dir_idx + '/'
    img_path_after = './Images/After/Img/' + dir_idx + '/'

    for idx, img_name in enumerate(os.listdir(syn_path_after)):
        syn = plt.imread(syn_path_after + str(img_name))
        rotate(syn, syn_path_after, img_name, -angle, tuple(pt_m))

    for idx, img_name in enumerate(os.listdir(img_path_after)):
        img = plt.imread(img_path_after + str(img_name))
        rotate(img, img_path_after, img_name, -angle, tuple(pt_m))


# Creating a plot with the EM and GT images together (Used for before/after slices)
def fig_creator(syn, img, save_path, fig_name):
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(syn)
    plt.axis("off")
    plt.savefig(save_path + fig_name)
    plt.close()

# Encoding images using base 64 encoding.
def b64_encoder(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

    # Creating the JSON file and deleteing the Images (temporary-directory) at the end.


def json_creator(syn_path, img_path):
    imgs = os.listdir(syn_path)

    final_file = dict()
    item_list = []

    for img in imgs:

        before_list = []
        after_list = []

        item = dict()
        item["Name"] = img
        item["EM"] = b64_encoder(os.path.join(img_path, img))
        item["GT"] = b64_encoder(os.path.join(syn_path, img))

        # Before
        dir_idx = str(img.strip('.png'))
        syn_path_before = './Images/Before/Syn/' + dir_idx + '/'

        for fig_name in os.listdir(syn_path_before):
            before_list.append(b64_encoder(os.path.join(syn_path_before, fig_name)))
        before_list.reverse()

        item["Before"] = before_list

        # After
        syn_path_after = './Images/After/Syn/' + dir_idx + '/'
        for fig_name in os.listdir(syn_path_after):
            after_list.append(b64_encoder(os.path.join(syn_path_after, fig_name)))
        after_list.reverse()

        item["After"] = after_list

        item["Label"] = "Correct"
        item_list.append(item)

    final_file["Data"] = item_list

    json_obj = json.dumps(final_file, indent=4)

    #with open("synAnno.json", "w") as outfile:
        #outfile.write(json_obj)

    shutil.rmtree('./Images')
    return json_obj


def loading_3d_file(im_file, gt_file):
    # Loading the 3D data. Ensure this matches the user input.
    gt = readvol(gt_file)  # The labelled file (Ground Truth: GT)
    im = readvol(im_file)  # The original Image (EM)

    # Processing the 3D volume to get 2D patches.
    syn, seg = process_syn(gt)
    visualize(syn, seg, im, rgb=True)
    syn_path = './Images/Syn_Mid/'
    img_path = './Images/Img_Mid/'

    # Aligning the patches from previous step by rotating them.
    for idx, img_name in enumerate(os.listdir(syn_path)):
        syn = plt.imread(syn_path + str(img_name))
        im = plt.imread(img_path + str(img_name))
        rot(syn, syn_path, im, img_path, img_name, a=0.7)

    # Creating plot combining EM and GT images for Before and After slices.

    for idx, img_name in enumerate(os.listdir(syn_path)):
        dir_idx = str(img_name.strip('.png'))
        img_path_before = './Images/Before/Img/' + dir_idx + '/'
        syn_path_before = './Images/Before/Syn/' + dir_idx + '/'
        for idx, img_name in enumerate(os.listdir(syn_path_before)):
            syn = plt.imread(syn_path_before + str(img_name))
            im = plt.imread(img_path_before + str(img_name))
            fig_creator(syn, im, syn_path_before, img_name)

        img_path_after = './Images/After/Img/' + dir_idx + '/'
        syn_path_after = './Images/After/Syn/' + dir_idx + '/'
        for idx, img_name in enumerate(os.listdir(syn_path_after)):
            syn = plt.imread(syn_path_after + str(img_name))
            im = plt.imread(img_path_after + str(img_name))
            fig_creator(syn, im, syn_path_after, img_name)

    # Creating and exporting the JSON file.
    return json_creator(syn_path, img_path)