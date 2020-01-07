'''
Filename: extract_test_data.py
Authors: Luke Rowe, Jing Zhu, Quinton Yong
Date: April 19, 2019

Description: This file extracts and preprocesses the test data used to evaluate the
MIML network. The extracted test data is written to an h5 file.
'''

import csv
import os
import cv2
import time
import h5py
import librosa as li
import numpy as np
import torch
from tqdm import tqdm
from preprocessing import get_frame_labels, extract_bases, get_frames

def find_training_data():
    print("Extracting training labels...")
    training_data_filename = "balanced_train_segments.csv" #use balanced youtube videos for testing
    labels_data_filename = "class_labels_indices.csv" #labels dict for pre-trained imageNet

    # labels violin, piano, guitar, drum (in that order)
    instrument_dict = {'violin':'/m/07y_7', 'piano':'/m/05r5c', 'guitar':'/m/042v_gx', 'drum':'/m/026t6'}
    violin_data = []
    piano_data = []
    guitar_data = []
    drum_data = []

    #load the video info from csv file
    with open(training_data_filename, mode='r') as file:
        csv_reader = csv.reader(file)

        # skip 3 lines of the csv file
        next(csv_reader)
        next(csv_reader)
        next(csv_reader)

        for row in (csv_reader):

            # process the encoded labels
            curr_label_id_list = row[3:len(row)]
            #remove the " characters from the label ids
            curr_label_id_list[0] = curr_label_id_list[0].replace("\"", "").strip()
            curr_label_id_list[-1] = curr_label_id_list[-1].replace("\"", "")

            # get 50 videos each class for testing
            if instrument_dict['violin'] in curr_label_id_list:
                violin_data.append([row[0], row[1].strip()])

            elif instrument_dict['piano'] in curr_label_id_list:
                piano_data.append([row[0], row[1].strip()])

            elif instrument_dict['guitar'] in curr_label_id_list:
                guitar_data.append([row[0], row[1].strip()])

            elif instrument_dict['drum'] in curr_label_id_list:
                drum_data.append([row[0], row[1].strip()])

    print("Number of violin samples:", len(violin_data))
    print("Number of piano samples:", len(piano_data))
    print("Number of guitar samples:", len(guitar_data))
    print("Number of drum samples:", len(drum_data))

    return violin_data + piano_data + guitar_data + drum_data

def get_audio_image(tr_data):
    # we keep track of the number of videos we cannot process
    num_skipped_videos = 0
    #numpy array to hold all W matrices
    W_all = np.zeros((len(tr_data), 2401,25))

    for count in tqdm(range(len(tr_data))):
        sample = tr_data[count]
        url = 'https://www.youtube.com/watch?v=' + sample[0]
        video_start_time = sample[1]

        # Download from local video file
        if (url):
            os.system("ffmpeg -ss " + str(video_start_time) + " -i $(youtube-dl -i -f 37/22/18 -g \'" + url + "\') -t " + str(
                10) + " -c copy video.mp4 >/dev/null 2>&1")
            os.system("ffmpeg -i video.mp4 audio.wav >/dev/null 2>&1")

            # obtain cv2.VideoCapture obj from downloaded video if success
            cap = cv2.VideoCapture("video.mp4")
        else:
            print("Error in downloading youtube video")
        if not os.path.exists("./video.mp4"):
            num_skipped_videos += 1
            continue

        # load audio from file
        ts, sr = li.core.load("./audio.wav", sr=48000)

        # skip if audio is shorter than 10 seconds
        if (len(ts) < 10*sr): 
            os.remove("./audio.wav")
            os.remove("./video.mp4")
            print("\n\n\n Sample {} is too short to be processed.".format(sample[0]))
            print("Namely, the sample is {} seconds long.\n\n\n".format(len(ts)/sr))
            num_skipped_videos += 1
            continue
        s = ts[0:10*sr] # cut audio into exact 10 seconds if it's longer than that

        all_image_tensors, skip = get_frames(cap) # get all the transformed frames

        # skip the current video if error occured during the frame extraction process
        if skip:
            num_skipped_videos += 1
            print("\n\n\nUnable to extract all frames from sample {}\n\n\n".format(sample[0]))
            if os.path.exists('./audio.wav'):
                os.remove('./audio.wav')
            if os.path.exists('./video.mp4'):
                os.remove('./video.mp4')
            for k in range(skip):
                if os.path.exists('frame{}.jpg'.format(k)):
                    os.remove('frame{}.jpg'.format(k))
            continue


        max_pool_labels = get_frame_labels(all_image_tensors) # get predicted labels for captured frames

        # create the set of basis vectors and object labels for each audio sample
        if count == 0:
            # call the NMF algorithm
            W_all = np.expand_dims(extract_bases(ts),0) # extract audio into audio bases
            labels_all = max_pool_labels.detach().unsqueeze(0) # use predicted maxpool labels

        else:
            W = extract_bases(ts) # extract audio into audio bases
            W_all = np.concatenate((W_all,np.expand_dims(W,0))) # append audio bases into list
            labels_all = torch.cat((labels_all, max_pool_labels.detach().unsqueeze(0)),0)

        # remove all the captured images, downloaded video and audio
        for i in range(10):
            os.remove('./frame{}.jpg'.format(i))
        os.remove('./video.mp4')
        os.remove('./audio.wav')

        # write data to h5 file every 500 samples in case lose connection
        # write audio frequency bases and Resnet maxpool labels into h5 file
        if (count % 500 == 0):
            with h5py.File('./test_data.h5', 'w') as hdf5:
                hdf5.create_dataset('bases', data=W_all)
                hdf5.create_dataset('labels', data=labels_all)

    # dump audio frequency bases and Resnet maxpool labels into h5 file
    with h5py.File('./test_data.h5', 'w') as hdf5:
        hdf5.create_dataset('bases', data=W_all)
        hdf5.create_dataset('labels', data=labels_all)

    print("{} samples were skipped.".format(num_skipped_videos))

def main():
    #remove wav and mp4 files if they exist from previous run
    if os.path.exists('./audio.wav'):
        os.remove('./audio.wav')
        print("Removing ./audio.wav file from previous run...")
    if os.path.exists('./video.mp4'):
        os.remove('./video.mp4')
        print("Removing ./video.mp4 file from previous run...")

    tr_data = find_training_data()
    print("Number of samples:", len(tr_data))
    get_audio_image(tr_data)

if __name__ == '__main__':
    main()
    
