'''
Filename: preprocessing.py
Authors: Luke Rowe, Jing Zhu, Quinton Yong
Date: April 19, 2019

This file contains methods used to extract and preprocess the data used in the MIML network
'''

import cv2
from torchvision import transforms
from PIL import Image
import librosa as li
import numpy as np
from sklearn.decomposition import NMF
import torch
import torchvision.models

def get_spectrograms(ts):
    # STFT matrix from which we can extract magnitude and phase
    # following the paper, we have 2401 frequency bins, so (1 + n_fft/2) = 2401 -> n_fft = 4800
    # To encode 0.1 second window length/ half second overlap:
    # n_fft represents the length of a frame (4800 samples = 0.1 seconds)
    # hop_length = 2400 means there is 2400 samples of "overlap" (0.05 second overlap)
    D = li.core.stft(ts, n_fft=4800, hop_length=2400)

    # shape in paper has 202 columns. I think this is a technicality
    # of little importance
    if D.shape != (2401, 201):
        print(D.shape)
        print("Length of ts:", len(ts))
    assert(D.shape == (2401,201))

    # as specified in librosa's stft documentation
    magnitude_spec = np.abs(D)
    phase_spec = np.angle(D)

    return D, magnitude_spec, phase_spec


def extract_bases(ts):
    #extract magnitude and phase spectrogram
    D, magnitude_spec, _ = get_spectrograms(ts)

    # initialize to non-negative random weights, use multiplicative-update solver (mu)
    # KL beta loss used, and maximum of 500 iterations
    nmf = NMF(n_components=25, init='random', solver='mu', beta_loss='kullback-leibler',
              max_iter=500)

    W = nmf.fit_transform(magnitude_spec)

    assert (W.shape == (2401, 25))

    return W


def get_frames(cap):
    # get frames images of video

    skip = 0 # skip if failed
    all_image_tensors = torch.zeros(10, 3, 224, 224)
    for i in range(0, 10): # capture 10 frames each video 
        cap.set(cv2.CAP_PROP_POS_MSEC, i * 1000) # capture frame every 1 second
        success, image = cap.read()
        if success:
            cv2.imwrite("frame{}.jpg".format(i), image)  # save frame as JPEG file

            image = Image.open('frame{}.jpg'.format(i))

            # perform transformation on the image. resize to 256, center crop to 224 and normalize in order
            # convert it into tensor
            normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

            image = Image.open('./frame{}.jpg'.format(i))
            image_tensor = image_transform(image).unsqueeze(0) # get transformed tensor from the current frame
            all_image_tensors[i] = image_tensor

        else:
            print("Frame {} not successfully extracted.".format(i))
            skip = i
            continue
    # all_image_tensors.shape: (10, 3, 224, 224)
    return all_image_tensors, skip




def get_frame_labels(all_image_tensors):
    # get predicted labels by using pre-trained resnet152

    resnet = torchvision.models.resnet152(pretrained='imagenet')
    resnet.eval()
    # return the most possible labels back. We take the max as a good overall
    # approximation of the objects present in the 10 image frames per sample
    max_pool_labels, _ = torch.max(resnet(all_image_tensors), dim=0)
    return max_pool_labels
    
