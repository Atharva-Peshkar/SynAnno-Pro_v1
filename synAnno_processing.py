# Importing the required libraries

import itertools
import numpy as np
from skimage.morphology import binary_dilation, remove_small_objects
from skimage.measure import label
import cv2
import os
import json
import cv2
import base64
import io
from pytorch_connectomics.connectomics.data.utils.data_io import readvol
from matplotlib import pyplot as plt
import sys
import h5py
from matplotlib.pyplot import imread,imsave
from scipy.stats import linregress
from util import bfly,rotateIm

# Creating the directories and returning their paths.
def dir_creator(parent_dir_path,dir_name):
    if os.path.exists(os.path.join(parent_dir_path,dir_name)):
        pass
    else:
        os.mkdir(os.path.join(parent_dir_path,dir_name))
    return os.path.join(parent_dir_path,dir_name)

# Encoding the images using Base64 encoding.
def b64_encoder(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8') 

# Saving the JSON
def json_creator(syn_path,img_path):
    
    imgs = os.listdir(syn_path)
    final_file = dict()
    item_list = []
    
    for img in imgs:
        item = dict()
        item["Name"] = img
        item["EM"] = b64_encoder(os.path.join(img_path,img))
        item["GT"] = b64_encoder(os.path.join(syn_path,img))
        item["Label"] = "Correct"
        item_list.append(item)

    final_file["Data"] = item_list

    json_obj = json.dumps(final_file, indent=4)

    #with open("synAnno.json", "w") as outfile:
        #outfile.write(json_obj)
    return json_obj

# Processing the synpases using binary dilation as well as by removing small objects.
def process_syn(gt, small_thres=16):
    seg = binary_dilation(gt.copy() != 0)
    seg = label(seg).astype(int)
    seg = seg * (gt.copy() != 0).astype(int)
    seg = remove_small_objects(seg, small_thres)

    c2 = (gt.copy() == 2).astype(int)
    c1 = (gt.copy() == 1).astype(int)

    syn_pos = np.clip((seg*2 - 1), a_min=0, a_max=None) * c1
    syn_neg = (seg*2) * c2
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
    y1o, y2o, x1o, x2o = bbox_2d # region to crop
    y1m, y2m, x1m, x2m = 0, sz[0], 0, sz[1]
    y1, x1 = max(y1o, y1m), max(x1o, x1m)
    y2, x2 = min(y2o, y2m), min(x2o, x2m)
    cropped = data[z, y1:y2, x1:x2]
    
    if mask is not None:
        mask_2d = mask[z, y1:y2, x1:x2]
        cropped = cropped*(mask_2d!=0).astype(cropped.dtype)
    
    pad = ((y1-y1o, y2o-y2), (x1-x1o, x2o-x2))
    if not all(v == 0 for v in pad):
        cropped = np.pad(cropped, pad, mode='constant', 
                         constant_values=pad_val)
    
    return cropped

# Converting to RGB
def syn2rgb(label):
    tmp = [None]*3
    tmp[0] = np.logical_and((label % 2) == 1, label > 0)
    tmp[1] = np.logical_and((label % 2) == 0, label > 0)
    tmp[2] = (label > 0)
    out = np.stack(tmp, 0).astype(np.uint8)
    return (out*255).transpose(1,2,0)

def visualize(syn, seg, img, syn_path, img_path, sz=100, rgb=False):
    seg_idx = np.unique(seg)[1:] # ignore background
    for idx in seg_idx:
        temp = (seg == idx)
        bbox = bbox2_ND(temp)
        z_mid = (bbox[0] + bbox[1]) // 2
        temp_2d = temp[z_mid]
        bbox_2d = bbox2_ND(temp_2d)
        y1, y2 = bbox_adjust(bbox_2d[0], bbox_2d[1], sz)
        x1, x2 = bbox_adjust(bbox_2d[2], bbox_2d[3], sz)
        crop_2d = [y1, y2, x1, x2]
        cropped_img = crop_pad_data(img, z_mid, crop_2d, pad_val=128)
        cropped_syn = crop_pad_data(syn, z_mid, crop_2d, mask=temp)
        
        if rgb:
            cropped_syn = syn2rgb(cropped_syn)

        # Saving the 2D processed patches.
        plt.imsave(os.path.join(syn_path,str(idx)+'.png'),cropped_syn,cmap='gray')
        plt.imsave(os.path.join(img_path,str(idx)+'.png'),cropped_img,cmap='gray')

def rot(syn,syn_path,im,img_path,img_name,thres_dilation=5,a=0.79):

    # compute rotation
    # compute rotatation by cleft
    # cleft: overlap for the dilated pre-/post-partner
    dilation_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thres_dilation, thres_dilation))
    cleft = np.logical_and(cv2.dilate((syn==255).astype(np.uint8), dilation_mask), \
                        cv2.dilate((syn==128).astype(np.uint8), dilation_mask))
    if cleft.max()==0:  #if there's no overlap (cleft) use the union of pre-post region
        cleft = syn>0
    pt = np.where(cleft>0)
    if pt[0].min()==pt[0].max():        #Using the pre-defined parameters (w, w2 and angle) or calculating them.
        w=100; w2=0
        angle = 90
    else:
        if pt[1].min()==pt[1].max():
            w=0
            angle = 0
        else:
            # angle concensus
            # pt[0]: x
            # pt[1]: y
            w,_,_,_,_ = linregress(pt[0],pt[1])
            angle = np.arctan(w)/np.pi*180
            w2,_,_,_,_ = linregress(pt[1],pt[0])
            angle2 = np.arctan(w2)/np.pi*180
            #if abs((angle+angle2)-90)>20:
            # trust the small one
            if abs(angle2) < abs(angle):
                angle = np.sign(angle2)*(90-abs(angle2))
                w = 1/w2

    # pre-post direction
    r1 = np.where(syn==128)
    r2 = np.where(syn==255)
    if len(r1[0])==0:
        r1 = r2
    if len(r2[0])==0:
        r2 = r1

    if abs(w)<0.2: # vertical bar, use w
        if abs(w)>1e-4:
            diff = (r2[1]-w*r2[0]).mean()-(r1[1]-w*r1[0]).mean()
        else: # almost 0
            diff = r2[1].mean()-r1[1].mean()
    else: # horizontal bar, use w2
        diff = -w2*((r2[0]-w2*r2[1]).mean()-(r1[0]-w2*r1[1]).mean())
    #print bid,w,diff
    if diff < 0:
        angle = angle-180
    pt_m = np.array([pt[1].mean(),pt[0].mean()])

    # re-center
    # Rotating the images.
    rot_im =  rotateIm(im, -angle, tuple(pt_m))
    imsave(os.path.join(img_path, img_name), rot_im)

    rot_im = rot_im*a
    rot_syn = rotateIm(syn, -angle, tuple(pt_m))

    # Creating and saving the composite image (GT overlaid over the EM (made a bit darker- done using the 'a' parameter from line 161))
    composite_image = np.maximum(rot_im,rot_syn)
    imsave(os.path.join(syn_path,img_name),composite_image)
    #print(rot_im.shape,rot_syn.shape)


def loading_3d_file(im_file, gt_file):
    # Loading the 3D data. Ensure this matches the user input.
    gt = readvol(gt_file)    #The labelled file (Ground Truth: GT)
    im = readvol(im_file)    #The original Image (EM)

#Creating the directory structure to store output data.
    parent_dir = dir_creator('./','Data')
    img_path = dir_creator('./Data','EM')
    syn_path = dir_creator('./Data','GT')

#Processing the 3D volume to get 2D patches.
    syn, seg = process_syn(gt)
    visualize(syn, seg, im, syn_path, img_path, sz=100, rgb=True)

# Aligning the patches from previous step by rotating them.
    for idx,img_name in enumerate(os.listdir(syn_path)):
        syn = plt.imread(os.path.join(syn_path,img_name))
        im = plt.imread(os.path.join(img_path,img_name))
        rot(syn,syn_path,im,img_path,img_name,a=0.7)

# Creating and exporting the JSON file.
    return json_creator(syn_path, img_path)
