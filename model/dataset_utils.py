import numpy as np
import tensorflow as tf
import os, random
from PIL import Image
import matplotlib.pyplot as plt


def decode_img_py_func(file_paths_tuple, file_path_mask, do_mask=True):
    file_path_mask = file_path_mask.numpy().decode('UTF-8')

    if do_mask:
        mask = Image.open(file_path_mask)
        mask = np.array(mask).reshape(64,64,1) // 255 # mask file has pixels from set {0, 255}

    img_final = np.empty(((64,64,0)))
    # if we have rfp channel or nuclear channel,
    # then file_paths_tuple will have 2 or 3 items
    # we loop through all files and concatenate them as separate channels
    for file_path1 in file_paths_tuple:
        file_path1 = file_path1.numpy().decode('UTF-8')
        im = Image.open(file_path1)
        im = np.array(im).reshape(64,64,1)
        im = (im-np.min(im))/(np.max(im)-np.min(im))
        if do_mask: im = im*mask
        img_final = np.concatenate((img_final, im),2)

    return img_final


def process_path_py_func(file_paths_tuple, file_path_mask, label, num_classes, do_mask=True, onehot=True):
    # label = get_label(file_path)
    # load the raw data from the file as a string
    # img = tf.io.read_file(file_path)
    # decode_img(file_path)
    img = tf.py_function(func=decode_img_py_func, inp=[file_paths_tuple, file_path_mask, do_mask], Tout=tf.float32)
    if onehot:
        label = tf.one_hot(label, num_classes)
    return img, label # 4093



def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label


def standardize_im(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label


def replace_in_list(my_list, a, b):
    return [x.replace(a,b) for x in my_list]



def convert_gfp_filenames(gfp_filenames, dataset, convert_to):
    prefixes_dict = {}
    prefixes_dict['chong'] = {'rfp': ['_gfp.','_rfp.'], 'mask': ['_gfp.','_mask.']}
    prefixes_dict['harsha'] = {'rfp': ['_ch1_','_ch3_'], 'nuc': ['_ch1_','_ch2_'], 'mask': ['_ch1_','_mask_']}

    ch_gfp, ch_new = prefixes_dict[dataset][convert_to]
    fnames_converted = replace_in_list(gfp_filenames, ch_gfp, ch_new)

    return fnames_converted


# [a, b, c] -> [a, b, c, a, b] (extend to length 5)
def extend_to_length(my_list, target_len):
    l = len(my_list)
    k = (target_len // l) +1
    extended_list = np.array(my_list*k)[:target_len]
    return list(extended_list)



def get_random_idx(N, seed):
    idx = list(np.arange(N))
    random.seed(seed)
    random.shuffle(idx)
    return idx



def shuffle_list_with_seed(my_list, seed):
    idx = get_random_idx(len(my_list), seed)
    my_list = list(np.array(my_list)[idx])
    return my_list




def get_labels_dict(labels_type, dataset):
    #base_path = '/scratch/ssd001/home/anastasia/datasets_ssd/'
    base_path = './data/'
    labels_type_to_filepath = {}
    labels_type_to_filepath['harsha'] = {
        'toy_dataset': base_path+'protein_to_files_dict_toy_dataset.npy',
        'proteins_by_screen': base_path+'protein_to_files_dict_harsha_by_screen.npy',
        'proteins_mix': base_path+'protein_to_files_dict_harsha_mixed.npy',
        'loc_huh': base_path+'huh_harsha_loc_to_files_dict.npy',
        'loc_mix': base_path+'loc_to_files_dict_harsha_mixed.npy',
        'loc_by_screen': base_path+'loc_to_files_dict_harsha_by_screen.npy',
    }
    labels_type_to_filepath['chong'] = {
        'proteins': base_path+'protein_to_files_dict_chong.npy',
        'loc_chong': base_path+'chong_loc_to_files_dict.npy',
        'loc_huh': base_path+'huh_loc_to_files_dict.npy',
    }
    labels_dict = np.load(labels_type_to_filepath[dataset][labels_type], allow_pickle=True)[()]
    return labels_dict



def get_num_steps(labels_type, dataset, batch_size, subset='train'):
    labels_dict = get_labels_dict(labels_type, dataset)
    total_len = 0
    for gene in list(labels_dict):
        total_len += len(labels_dict[gene][subset])
    num_steps = total_len // batch_size + 1
    return num_steps



def get_files_and_labels_from_dict(protein_to_files_dict, subset, \
                                   uniform=False, window=100, shuffle_seed=-1, \
                                   override_gene_list=[], start_idx=0):
    keys = list(protein_to_files_dict)
    if len(override_gene_list)!=0:
        keys = [x for x in keys if x in override_gene_list]

    fnames_gfp, labels = [], []

    for i in range(len(keys)):
        curr_dict = protein_to_files_dict[keys[i]]
        gfp_gene = curr_dict[subset] # can be train / test / val files

        if uniform: # if True, each protein will have exactly N images
            gfp_gene = extend_to_length(gfp_gene, window)

        fnames_gfp += gfp_gene
        labels += [i]*len(gfp_gene)

    if shuffle_seed != -1:
        fnames_gfp = shuffle_list_with_seed(fnames_gfp, shuffle_seed)
        labels = shuffle_list_with_seed(labels, shuffle_seed)

    fnames_gfp = fnames_gfp[start_idx:]
    labels = labels[start_idx:]
    return fnames_gfp, labels




def get_dataset_from_lists(filenames_channels_list, filenames_mask, labels, \
                           num_classes, do_mask, batch_size, \
                           standardize=False, onehot=True, train=True):
    list_of_ds = []
    # we loop through the list of filenames lists
    # this can be [filenames_gfp, filenames_rfp, filenames_mask]
    for filenames_gfp in filenames_channels_list:
        ds_gfp = tf.data.Dataset.from_tensor_slices(filenames_gfp)
        list_of_ds.append(ds_gfp)

    ds_mask = tf.data.Dataset.from_tensor_slices(filenames_mask)
    list_of_ds.append(ds_mask)

    ds_labels = tf.data.Dataset.from_tensor_slices(labels)
    list_of_ds.append(ds_labels)
    # tuple is always constructed as following:
    # [ds_channel_1, ..., ds_channel_N, ds_mask, ds_label]
    # so that all channels are stacked together, mask is applied (if flag set to True) and
    # multi-channel masked image + label are returned at the end
    tuple_of_ds = tuple(list_of_ds)
    dataset = tf.data.Dataset.zip(tuple_of_ds)

    num_ch = len(filenames_channels_list)
    if num_ch==1:
        dataset = dataset.map(lambda f1,f_mask,l: process_path_py_func([f1],f_mask,l, num_classes,do_mask,onehot), \
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if num_ch==2:
        dataset = dataset.map(lambda f1,f2,f_mask,l: process_path_py_func([f1,f2],f_mask,l,num_classes,do_mask,onehot), \
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if num_ch==3:
        dataset = dataset.map(lambda f1,f2,f3,f_mask,l: process_path_py_func([f1,f2,f3],f_mask,l,num_classes,do_mask,onehot), \
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if standardize:
        dataset = dataset.map(standardize_im, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if train:
        dataset = dataset.map(train_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)

    return dataset






# Harsha data = ~10 million single-cell images
# consists of gfp, nucleus, rfp (=cyto) channels and masks
def get_dataset(labels_type, dataset, subset, batch_size, \
                do_mask=False, use_rfp=False, use_nuc=False, uniform_sampling=False, \
                override_gene_list=[], standardize=True, window=100, \
                start_idx=0, shuffle_seed=-1):

    labels_dict = get_labels_dict(labels_type, dataset)
    num_classes = len(list(labels_dict))

    fnames_gfp, labels = get_files_and_labels_from_dict(labels_dict, subset, uniform=uniform_sampling, \
                                                       window=window, shuffle_seed=shuffle_seed, \
                                                       override_gene_list=override_gene_list, start_idx=start_idx)

    fnames_channels_list = [fnames_gfp]
    fnames_mask = ['none']*len(fnames_gfp)

    if do_mask:
        standardize = False # we are normalizing instead (done by default) to keep background at 0
        fnames_mask = convert_gfp_filenames(fnames_gfp, dataset=dataset, convert_to='mask')

    if use_rfp:
        fnames_rfp = convert_gfp_filenames(fnames_gfp, dataset=dataset, convert_to='rfp')
        fnames_channels_list.append(fnames_rfp)

    if use_nuc:
        if dataset=='chong':
            raise Exception("No nuclear channel in Chong dataset")
        fnames_nuc = convert_harsha_gfp_filenames(fnames_gfp, convert_to='nuc')
        fnames_channels_list.append(fnames_nuc)

    dataset = get_dataset_from_lists(fnames_channels_list, fnames_mask, labels, \
                                     num_classes, do_mask, batch_size, \
                                     standardize=standardize, onehot=True, train=(subset=='train'))

    return dataset
