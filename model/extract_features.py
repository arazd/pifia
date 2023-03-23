import sys, os, time
import argparse
import keras_models, datasets_tf2, PR_scores_utils
import logging

import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_labels_harsha(labels_type):
    labels_dict = None

    if labels_type == 'toy_dataset':
        labels_dict = np.load('./data/protein_to_files_dict_toy_dataset.npy', allow_pickle=True)[()]

    elif labels_type == 'proteins_by_screen':
        labels_dict = np.load('./data/protein_to_files_dict_harsha_by_screen.npy',allow_pickle=True)[()]

    elif labels_type == 'proteins_mix':
        labels_dict = np.load('./data/protein_to_files_dict_harsha_mixed.npy',allow_pickle=True)[()]

    elif labels_type == 'loc_huh':
        labels_dict = np.load('./data/huh_harsha_loc_to_files_dict.npy',allow_pickle=True)[()]

    return labels_dict



def create_dataset_for_gene(gene, labels_dict, subset='val'):
    fnames_gfp_val = labels_dict[gene][subset]
    fnames_gfp_val = [x.replace('ssd001/', 'ssd002/') for x in fnames_gfp_val]
    filenames_channels_list_val = [fnames_gfp_val]
    fnames_mask_val = ['none']*len(fnames_gfp_val)
    i = list(labels_dict).index(gene)
    labels_val = [i]*len(fnames_gfp_val)

    num_classes = len(list(labels_dict))

    dataset_val = datasets_tf2.get_dataset_from_lists(filenames_channels_list_val, fnames_mask_val, labels_val, \
                                 num_classes=num_classes, do_mask=False, standardize=True, rescale=False, train=False, \
                                 shuffle=False, onehot=True, b_size=400)
    return dataset_val



def get_all_images_from_dataset(dataset_gene):
    im_total = []
    for im, l in dataset_gene:
        im_total.append(im)
    im_total = np.vstack(im_total)
    return im_total



def get_features(images, model, layer, deep_loc_preds=False):
    if deep_loc_preds:
        features = np.array(model.predict(images))
    else:
        features = np.array(model.get_features(images, layer=layer))
    features = np.mean(features,0)
    return features


def get_features_from_gene(gene, labels_dict, model,
                           subset='val', layer='d2', deep_loc_preds=False):
    dataset_gene = create_dataset_for_gene(gene, labels_dict, subset)
    images = get_all_images_from_dataset(dataset_gene)
    features = get_features(images, model, layer, deep_loc_preds)
    return features



def create_features_dict(labels_dict, model,
                         subset='val', layer='d2', deep_loc_preds=False):
    features_dict = {}
    all_genes = list(labels_dict)
    pbar = tqdm(total=len(all_genes))

    for gene in all_genes:
        pbar.update(1)
        features = get_features_from_gene(gene, labels_dict, model, subset, layer, deep_loc_preds)
        features_dict[gene] = features

    pbar.close()
    return features_dict



def get_weights_list(model_dir):
    l = sorted(os.listdir(model_dir))

    f_list = set([x.split('.')[0].split('_')[-1] for x in l if 'features_dict' in x])
    f_list = sorted([int(x) for x in f_list])

    l = set([x.split('.')[0] for x in l if 'weights' in x])
    l = sorted(list(l))

    epochs = [x.split('epoch')[1].split('_')[0] for x in l]
    epochs = [int(x) for x in epochs]
    epochs = [x for x in epochs if x not in f_list]

    return epochs



def get_model(num_classes, num_channels, args):
    if args.backbone != 'pifia_network':
        model = models.custom_network(num_classes=num_classes,
                                      num_channels=num_channels,
                                      dropout_rate=0.0,
                                      backbone=args.backbone,
                                      dense1_size=args.dense1_size,
                                      num_features=args.num_features,
                                      pool=args.avg_pool)
    else:
        model = models.pifia_network(num_classes,
                                    k=args.width_factor,
                                    dense1_size=args.dense1_size,
                                    num_features=args.num_features,
                                    last_block=(args.last_block==1))

    return model



def main(args):

    if args.log_file is not None:
        log_file = args.log_file
    else:
        log_file = os.path.join('log' + str(args.epoch))
    logging.basicConfig(filename=log_file, level=logging.DEBUG)

    labels_dict = load_labels_harsha(args.labels_type)
    num_classes = len(list(labels_dict))
    if args.deep_loc_preds: num_classes = 17
    num_channels = 1 + 1*args.use_rfp + 1*args.use_nuc
    model = get_model(num_classes, num_channels, args)

    if args.epoch == -1:
        epochs = get_weights_list(args.model_dir)
    else:
        epochs = [args.epoch]


    for epoch in epochs:
        logging.info("extracting features at epoch {}".format(epoch) )

        weights_path = os.path.join(args.model_dir, 'weights_epoch' + str(epoch) + '_i0')
        model.load_weights(weights_path)
        features_dict = create_features_dict(labels_dict, model,
                                             subset=args.subset, layer=args.layer,
                                             deep_loc_preds=args.deep_loc_preds)
        path = os.path.join(args.model_dir, args.layer + '_features_dict_' + str(epoch))
        np.save(path, features_dict)

    logging.info("Finished!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='Script for extracting features from pre-trained models.'
    )

    parser.add_argument(
        '--backbone',
        type=str,
        help='name of the backbone architecture to be used',
        default='dense_net_121'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='dataset name (chong or harsha)',
        default='harsha'
    )

    parser.add_argument(
        '--labels_type',
        type=str,
        help='type of labels that will be used while training (i.e. individual proteins, localizations etc.)',
        default='proteins_mix'
    )

    parser.add_argument(
        '--layer',
        type=str,
        help='which layer to use to extract features',
        default='d2'
    )


    parser.add_argument(
        '--do_mask',
        type=bool,
        help='using masked cells? Alternatively raw data',
        default=False
    )

    parser.add_argument(
        '--deep_loc_preds',
        type=bool,
        help='get probability predictions from DeepLoc',
        default=False
    )

    parser.add_argument(
        '--use_rfp',
        type=bool,
        help='using RFP cytosolic channel',
        default=False
    )

    parser.add_argument(
        '--use_nuc',
        type=bool,
        help='using nuclear channel',
        default=False
    )

    parser.add_argument(
        '--subset',
        type=str,
        help='subset of data to be used to extract features, default: validation',
        default='val'
    )

    parser.add_argument(
        '--dense1_size',
        type=int,
        help='number of neurons in 1st dense layer (before feature layer)',
        default=512
    )

    parser.add_argument(
        '--num_features',
        type=int,
        help='number of neurons in 2nd dense layer (feature layer)',
        default=512
    )

    parser.add_argument(
        '--avg_pool',
        type=bool,
        help='do global avg pooling before flatten',
        default=False
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        help='path to look for the model weights at different epochs',
        default=None
    )

    parser.add_argument(
        '--epoch',
        type=int,
        help='epoch to load model from',
        default=-1
    )

    parser.add_argument(
        '--width_factor',
        type=float,
        help='coefficient to make wider deep loc',
        default=1
    )

    parser.add_argument(
        '--last_block',
        type=int,
        help='use last block (for DeepLock backbone)',
        default=1
    )

    parser.add_argument(
        '--log_file',
        type=str,
        help='specify the location of the output directory, default stdout',
        default=None
    )

    main(parser.parse_args())
