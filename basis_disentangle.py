'''
Filename: basis_disentangle.py
Authors: Luke Rowe, Jing Zhu, Quinton Yong
Date: April 19, 2019

Description: This file contains the methods to perform the per-object
basis disentanglement following the training of the MIML network.
'''

import torch
import numpy as np
import os
import h5py
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from data.MIML_dataset import subsetOfClasses, softmax, normalizeBases
from tqdm import tqdm

def get_sample_data(index, bases_h5, labels_h5, opt):


    bases = bases_h5[index]

    # perform softmax on the scores of the selected classes
    loaded_label = softmax(subsetOfClasses(labels_h5[index])) 

    # initialize labels array to -1's. -1 labels are ignored
    label = np.zeros(opt.L) - 1

    # keep labels with prob. higher than threshold 0.3
    # we take the maximum label in case no label is higher than the 0.3 threshold
    label_index = [np.argmax(loaded_label)]
    label_index = list(set(label_index) | set(np.where(loaded_label >= 0.3)[0]))
    for i in range(len(label_index)):
        label[i] = label_index[i]

    #perform basis normalization
    bases = normalizeBases(bases, opt.norm)

    return {'bases': bases, 'label': label}


def all_sample_data(bases_h5, labels_h5, opt):
    #get all the sample bases and labels

    print("Collecting all training data...")

    lab = np.zeros((labels_h5.shape[0], opt.L))
    bas = np.zeros(bases_h5.shape)

    # generate preprocessed labels and bases data to prepare for "half-forward" pass
    for index in tqdm(range(len(bases_h5))):
        data_dict = get_sample_data(index, bases_h5, labels_h5, opt)
        lab[index] = np.expand_dims(data_dict['label'], 0)
        bas[index] = np.expand_dims(data_dict['bases'], 0)

    lab = torch.from_numpy(lab)
    bas = torch.from_numpy(bas)

    return {'bases': bas, 'label': lab}

def disentangle():
    opt = TestOptions().parse()

    # retrieve the model from the latest checkpoint
    model = torch.load(os.path.join('.', opt.checkpoints_dir, opt.name, str(opt.which_epoch) + '.pth'))

    # we disentangle bases using the training set
    h5f = h5py.File('train.h5', 'r')
    bases_h5 = h5f['bases']
    labels_h5 = h5f['labels']

    # retrieve training bases
    all_data_dict = all_sample_data(bases_h5, labels_h5, opt)
    #bases bank contains the bases for all the training data
    bases_bank = all_data_dict['bases']

    #find relation map by pushing through network until first max-pooling layer of forward pass
    relation_map = model.half_forward(all_data_dict).double().detach()
    relation_map = relation_map.cpu().numpy()

    # a binary list that determines whether the matrix of basis vectors
    # has been created for that particular object
    obj_bases = {}
    start_obj = [0,0,0,0]
    cur_num_basis = [0,0,0,0]
    print("Disentangling bases...")
    for i in tqdm(range(len(relation_map))):
        #perform softmax on each basis vectors
        relation_map[i] = softmax(relation_map[i])

        # we have 4 objects classes - due to the order specified in subsetOfClasses, we have
        # 0,1,2,3 corresponds to drum, guitar, piano, violin
        for object in range(4):
            # only check inidividual basis vectors if at least 4 have softmax score > 0.1
            if np.sum((relation_map[i][:,object] > 0.1)) >= 4:
                # retrieve all bases with softmax score over 0.15
                indices = np.array( np.where(relation_map[i][:,object] > 0.15))
                indices = indices.reshape((indices.shape[1],))
                if start_obj[object] == 0:
                    obj_bases[object] = bases_bank[i][:,indices]
                    start_obj[object] = 1
                    cur_num_basis[object] += obj_bases[object].shape[1]
                else:
                    #cap number of basis vectors at 50 for each object
                    if cur_num_basis[object] < 50:
                        obj_bases[object] = np.concatenate((obj_bases[object], bases_bank[i][:,indices]),1)
                        cur_num_basis[object] += bases_bank[i][:,indices].shape[1]

    print( "drum: ", obj_bases[0].shape[1], "guitar: ", obj_bases[1].shape[1],
           "piano: ", obj_bases[2].shape[1], "violin: ", obj_bases[3].shape[1])
    #return a dictionary containing the basis vectors for each object
    object_bases = {'drum' : obj_bases[0], 'guitar': obj_bases[1],
                    'piano': obj_bases[2], 'violin': obj_bases[3]}
    return object_bases

