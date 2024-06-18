import tensorflow as tf
import numpy as np
from model import dataset_utils

def create_dataset_for_protein(protein, labels_dict, subset):
    fnames_gfp_val = labels_dict[protein][subset]
    filenames_channels_list_val = [fnames_gfp_val]
    fnames_mask_val = ['none']*len(fnames_gfp_val)
    i = list(labels_dict).index(protein)
    labels_val = [i]*len(fnames_gfp_val)
    
    num_classes = len(list(labels_dict))

    dataset_val = dataset_utils.get_dataset_from_lists(filenames_channels_list_val, fnames_mask_val, labels_val, \
                                 num_classes=num_classes, do_mask=False, standardize=True, train=False, \
                                 onehot=True, batch_size=400)
    return dataset_val


def get_all_images_from_dataset(dataset_protein):
    im_total = []
    for im, l in dataset_protein:
        im_total.append(im)
    im_total = np.vstack(im_total)
    return im_total



def get_features(images, model, average=True):
    features = np.array(model.get_features(images))
    if average:
        features = np.mean(features,0)
    return features


def get_features_from_protein(protein, labels_dict, model, average=True, subset='test'):
    dataset_protein = create_dataset_for_protein(protein, labels_dict, subset='test')
    images = get_all_images_from_dataset(dataset_protein)
    features = get_features(images, model, average)
    return features, images


def create_features_dict(labels_dict, model):
    features_dict = {}
    all_proteins = list(labels_dict)
    
    for protein in all_proteins:
        features, _ = get_features_from_protein(protein, labels_dict, model)
        features_dict[protein] = features
    
    return features_dict
