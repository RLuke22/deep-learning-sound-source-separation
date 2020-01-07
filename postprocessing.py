'''
Filename: postprocessing.py
Authors: Luke Rowe, Jing Zhu, Quinton Yong
Date: April 19, 2019

Description: This file reconstructed tthe isolated source signals for each
detected object in the video. The separated audio sources are written to separate WAV files
'''

import numpy as np
import librosa, librosa.display
import cv2
from sklearn.decomposition import non_negative_factorization
from preprocessing import get_spectrograms, get_frames, get_frame_labels
import os
from basis_disentangle import disentangle


def main():
    # retrieve the audio basis vectors for each object 
    object_dict = disentangle()

    if os.path.exists('./audio.wav'):
        os.remove('./audio.wav')
    if os.path.exists('./video.mp4'):
        os.remove('./video.mp4')

    # test video
    video_url = 'https://www.youtube.com/watch?v=DOn33Ugbefw'
    if (video_url):
        os.system("ffmpeg -ss " + str(105) + " -i $(youtube-dl -i -f 37/22/18 -g \'" + video_url + "\') -t " + str(
            10) + " -c copy video.mp4 >/dev/null 2>&1")
        os.system("ffmpeg -i video.mp4 audio.wav >/dev/null 2>&1")

        # obtain cv2.VideoCapture obj from downloaded video if success
        cap = cv2.VideoCapture("video.mp4")
    else:
        print("Error in downloading youtube video")

    # load audio file
    ts, sr = librosa.core.load("./audio.wav", sr=48000)

    # skip if audio is shorted than 10 seconds
    if (len(ts) < 10 * sr):
        os.remove("./audio.wav")
        os.remove("./video.mp4")
        print("\n\n\nSample {} is too short to be processed.".format(1))
        print("Namely, the sample is {} seconds long.\n\n\n".format(len(ts) / sr))
        exit(1)

    # crop to 10 seconds if audio is longer
    ts = ts[0:sr * 10]

    all_image_tensors, skip = get_frames(cap) # get all the transformed frames

    # skip video if error in frame extraction process
    if skip:
        print("\n\n\nUnable to extract all frames from sample {}\n\n\n".format(1))
        if os.path.exists('./audio.wav'):
            os.remove('./audio.wav')
        if os.path.exists('./video.mp4'):
            os.remove('./video.mp4')
        for k in range(skip):
            if os.path.exists('frame{}.jpg'.format(k)):
                os.remove('frame{}.jpg'.format(k))
        exit(1)

    # get predicted labels for captured frames
    max_pool_labels = get_frame_labels(all_image_tensors)

    # reshape the labels into (1000,) and perform softmax on labels
    labels = max_pool_labels.detach().unsqueeze(0).numpy().astype(float).reshape(1000, )
    softmax_labels = np.exp(labels) / np.sum(np.exp(labels), axis=0)

    # we take the top 4 objects in the scene and intersect with the piano/violin/guitar/drum labels
    labels = set(softmax_labels.argsort()[-4:][::-1]).intersection(set([889, 579, 881, 402, 541]))

    # reindex the labels of drum/guitar/piano/violin for convenience
    labels_new = []
    start_index = 0
    # sep holds the start index for each concatenated W matrix
    sep = [start_index]

    # append audio basis vectors of each object in columns
    for i in labels:
        if i == 541:
            labels_new.append('drum')
            start_index += object_dict['drum'].shape[1]
            sep.append(start_index)
        elif i == 402:
            labels_new.append('guitar')
            start_index += object_dict['guitar'].shape[1]
            sep.append(start_index)
        elif (i == 579 or i == 881) and 'piano' not in labels_new:
            labels_new.append('piano')
            start_index += object_dict['piano'].shape[1]
            sep.append(start_index)
        elif (i == 889):
            labels_new.append('violin')
            start_index += object_dict['violin'].shape[1]
            sep.append(start_index)

    print("Objects in test video: ", labels_new)

    # the last index is the number of basis vectors in the concatenated W matrix
    num_basis_vectors = sep[-1]

    # W shape (num_of_frequency_bins, num_of_basis_vectors)
    W = np.zeros((2401, num_basis_vectors))

    # concatenate audio bases of each object into W in columns
    for index, object in enumerate(labels_new):
        W[:, sep[index]:sep[index + 1]] = object_dict[object]

    # get spectrograms of audio
    spec, magnitude_spec, phase_spec = get_spectrograms(ts)

    V = magnitude_spec

    # W_transpose is used as the fixed "H" in the NMF procedure
    W_transpose = W.T
    assert (W_transpose.shape == (num_basis_vectors, 2401))

    # Since sklearn can only solve V=WH while keeping the H fixed, we solve the factorization:
    # V^T = H^T*W^T, and take the transpose of the resultant matrix H_t to retreive H.
    H_t, _, _ = non_negative_factorization(X=V.T, H=W_transpose, n_components=num_basis_vectors, init='random',
                                           update_H=False, max_iter=1500, verbose=1)
    H = H_t.T

    V_dict = {}

    #append to dictionary of object spectrograms
    for i, object in enumerate(labels_new):
        V_dict[object] = np.matmul(object_dict[object], H[sep[i]:sep[i + 1]])
        assert (V_dict[object].shape == (2401, 201))

    #calculate sum of all object magnitude spectrograms
    V_sum = np.zeros((2401, 201))
    for V_obj in V_dict.values():
        V_sum = V_sum + V_obj

    #mask the spectrogram, compute istft and write to wav file
    sample_rate = 48000
    for i, object in enumerate(labels_new):

        # softmask the mixture spectrogram
        double_V_j = (V_dict[object] / (V_sum)) * spec 

        # use istft to reconstruct time domain signal from sprectrogram
        source_j = librosa.core.istft(double_V_j, hop_length=2400) 

        # write reconstructed signal into wav file for testing
        print("Writing to ./{}.wav...".format(object))
        librosa.output.write_wav('./{}.wav'.format(object), source_j, sample_rate)

    # remove generated audio, video and frame images
    if os.path.exists('./audio.wav'):
        os.remove('./audio.wav')
    if os.path.exists('./video.mp4'):
        os.remove('./video.mp4')
    for i in range(10):
        if os.path.exists('./frame{}'.format(i)):
            os.remove('./frame{}'.format(i))

if __name__ == '__main__':
    main()
