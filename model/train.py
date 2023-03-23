import sys, os, time
import argparse, logging
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np

import models, dataset_utils 

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
HDD_MODELS_DIR = os.environ['HDD_MODELS_DIR']

# create datasets
def create_dataset(batch_size, dataset_name, labels_type, \
                   do_mask, use_rfp, use_nuc, uniform_sampling, \
                   start_idx, shuffle_seed=-1):

    dataset_train = dataset_utils.get_dataset(labels_type, dataset_name, 'train', batch_size, \
                                              do_mask=do_mask, use_rfp=use_rfp, use_nuc=use_nuc, \
                                              uniform_sampling=uniform_sampling, standardize=True, window=1000, \
                                              start_idx=start_idx, shuffle_seed=shuffle_seed)

    dataset_val = dataset_utils.get_dataset(labels_type, dataset_name, 'val', batch_size, \
                                            do_mask=do_mask, use_rfp=use_rfp, use_nuc=use_nuc, \
                                            uniform_sampling=uniform_sampling, standardize=True, window=50)

    labels_dict = dataset_utils.get_labels_dict(labels_type, dataset_name)
    num_classes = len(list(labels_dict))
    return dataset_train, dataset_val, num_classes


# create model
def create_model(backbone, num_classes, num_channels, \
                 dropout_rate, dense1_size, num_features, avg_pool, width_factor):
    # Create an instance of the model
    if backbone=='pifia_network':
        model = models.pifia_network(num_classes,
                                      k=width_factor,
                                      num_features=num_features,
                                      dense1_size=dense1_size,
                                      dropout_rate=dropout_rate)

    else:
        model = models.custom_network(num_classes=num_classes,
                                            num_channels=num_channels,
                                            dropout_rate=dropout_rate,
                                            backbone=backbone,
                                            dense1_size=dense1_size,
                                            num_features=num_features,
                                            pool=avg_pool)
    return model



@tf.function
def train_step(images, labels, model, optimizer, train_loss, train_accuracy, loss_object):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)



@tf.function
def val_step(images, labels, model, val_loss, val_accuracy, loss_object):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  val_loss(t_loss)
  val_accuracy(labels, predictions)



def get_folder_name(args):
    save_name = args.dataset + '_dense'+ str(args.dense1_size) + '_features'+ str(args.num_features) +\
                args.labels_type + '_mask'*args.do_mask + \
                '_rfp'*args.use_rfp + '_nuc'*args.use_nuc + \
                ('_dropout_'+str(args.dropout_rate)) *(args.dropout_rate>0) + \
                '_' + args.backbone +  \
                '_uniform'*args.uniform_sampling + args.save_prefix +\
                '_batch_' + str(args.batch_size) + '_lr' + str(args.learning_rate) + '_cos_decay'*args.cosine_decay +\
                '_avg_pool' * args.avg_pool + ('_width' + str(args.width_factor)) * (args.backbone == 'pifia_network' )

    return save_name



def create_weights_folder(args):
    save_name = get_folder_name(args)
    # create a folder on HDD where weights are saved
    save_weights_dir = os.path.join(HDD_MODELS_DIR, save_name)
    if os.path.exists(save_weights_dir)==False:
        os.mkdir(save_weights_dir)

    # # create a symlink on SSD pointing to the saved weights folder
    # dst = os.path.join(LINK_MODELS_DIR, save_name)
    # if os.path.islink(save_path)==False:
    #     os.symlink(weights_save_path, dst)
    return save_weights_dir


def create_logfile_symlink(args):
    # create symlink to train.log on HDD folder
    save_name = get_folder_name(args)
    dst = os.path.join(HDD_MODELS_DIR, save_name, 'train.log')
    if os.path.islink(dst)==False:
         os.symlink(args.log_file, dst)


def get_learning_rate(args, initial_learning_rate, cos_decay):
    if cos_decay:
        decay_steps = args.num_epoch * \
                      dataset_utils.get_num_steps(args.labels_type, args.dataset, args.batch_size)
        learning_rate = tf.keras.experimental.CosineDecay(initial_learning_rate, decay_steps)
    else:
        learning_rate = initial_learning_rate
    return learning_rate



def create_or_restore_training_state(args, dataset, dataset_val, batch_size, checkpoint_dir,
                                     backbone, num_classes, num_channels, dropout_rate,
                                     initial_learning_rate, cos_decay):

    model = create_model(backbone, num_classes, num_channels, dropout_rate, \
                         args.dense1_size, args.num_features, args.avg_pool, \
                         args.width_factor)
    learning_rate = get_learning_rate(args, initial_learning_rate, cos_decay)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    epoch = tf.Variable(0)
    step  = tf.Variable(0)
    global_rng_state = tf.random.experimental.get_global_generator().state

    # create the checkpoint variable
    checkpoint = tf.train.Checkpoint(epoch=epoch, step=step,
                                     optimizer=optimizer,
                                     model=model, batch_size=tf.Variable(batch_size),
                                     train_loss=train_loss, train_accuracy=train_accuracy,
                                     global_rng_state=global_rng_state)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    checkpoint_dir,
                                                    max_to_keep=3)

    # now, try to recover from the saved checkpoint, if successful, it should
    # re-populate the fields of the checkpointed variables.
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    checkpoint.restore(latest_checkpoint).expect_partial()
    if latest_checkpoint:
        tf.random.experimental.set_global_generator(
            tf.random.experimental.Generator(
                state=global_rng_state, alg=1))
        logging.info("training state restored at epoch {}".
              format(int(epoch.numpy()) ))

        dataset, dataset_val, \
        num_classes = create_dataset(batch_size, args.dataset,
                                     args.labels_type, args.do_mask,
                                     args.use_rfp, args.use_nuc,
                                     args.uniform_sampling,
                                     start_idx = batch_size * int(step.numpy()),
                                     shuffle_seed = int(epoch.numpy()) )
    else:
        logging.info("No checkpoint detected, starting from initial state")

    return model, optimizer, dataset, dataset_val, train_loss, train_accuracy, val_loss, val_accuracy, \
           loss_object, epoch, step, checkpoint_manager, checkpoint_dir




def train(args, model, optimizer, dataset_train, dataset_val, \
          train_loss, train_accuracy, val_loss, val_accuracy, loss_object, \
          epoch, step, max_epoch, checkpoint_interval, checkpoint_manager, checkpoint_dir, save_weights_dir):

    start_time = time.time()
    while epoch < max_epoch:
        count = 0
        for images, labels in dataset_train:
            train_step(images, labels, model, optimizer, train_loss, train_accuracy, loss_object)

            if count == 0:
                w_name = "weights_epoch{}_i{}".format( int(epoch.numpy()), int(step.numpy()) )
                model.save_weights(os.path.join(save_weights_dir,w_name))
                logging.info("Saved weights ({})".format(int(step.numpy())))

            cur_time = time.time()
            if cur_time - start_time > checkpoint_interval:
                path = checkpoint_manager.save()
                logging.info("Epoch {}, Training state saved at {}".format(
                    int(epoch.numpy()), path))
                start_time = time.time()
            count += 1
            step.assign_add(1)

        for val_images, val_labels in dataset_val:
            val_step(val_images, val_labels, model, val_loss, val_accuracy, loss_object)

        template = 'Epoch {}, ({} steps), Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}, LR: {}'
        logging.info(template.format(
                    int(epoch.numpy()),
                    int(step.numpy()),
                    train_loss.result(),
                    train_accuracy.result() * 100,
                    val_loss.result(),
                    val_accuracy.result() * 100,
                    optimizer._decayed_lr('float32').numpy()))
        #data_iter = iter(dataset)

        # this attribute is available without the "_" in tf 2.2
        # Subtle! if you rebind the variable name to a different entity,
        # in this case, data_iter = iter(dataset), you need to update the
        # reference in the checkpoint as well otherwise the checkpoint will not save the
        # correct iterator.
        # Alternatively, all your checkpoint variables are directly read/written
        # through "checkpoint_manager._checkpoint", but the code looks clumsy.
        #
        # Notice that all other checkpointed variables are modified in-place
        # so they do not need an update to the checkpoint.
        #checkpoint_manager._checkpoint.data_iter = data_iter
        epoch.assign_add(1)
        step.assign(0)

        dataset_train, dataset_val, \
        num_classes = create_dataset(args.batch_size, args.dataset,
                                     args.labels_type, args.do_mask,
                                     args.use_rfp, args.use_nuc,
                                     args.uniform_sampling,
                                     start_idx=args.batch_size*int(step.numpy()),
                                     shuffle_seed=int(epoch.numpy()))

        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()



def main(args):

    save_weights_dir = create_weights_folder(args)

    checkpoint_dir = args.checkpoint_dir
    logging.basicConfig(filename=args.log_file,level=logging.DEBUG)
    num_channels = 1 + 1*args.use_rfp + 1*args.use_nuc

    logging.info("starting training script")
    logging.info("GPU available: {}".format(tf.test.is_gpu_available()))
    create_logfile_symlink(args)

    rand_seed = np.random.randint(1000)
    tf.random.set_seed(rand_seed)
    logging.info("random seed: {}".format(rand_seed))

    dataset_train, dataset_val, \
    num_classes = create_dataset(args.batch_size, args.dataset,
                                 args.labels_type, args.do_mask,
                                 args.use_rfp, args.use_nuc,
                                 args.uniform_sampling,
                                 start_idx=0, shuffle_seed=0)


    #for im, l in dataset.take(3).as_numpy_iterator():
    #    logging.info("{} {}".format(len(l), im.shape))

    model, optimizer, dataset_train, dataset_val, \
    train_loss, train_accuracy, val_loss, val_accuracy, loss_object, \
    epoch, step, checkpoint_manager, checkpoint_dir = \
        create_or_restore_training_state(args, dataset_train, dataset_val, args.batch_size, checkpoint_dir,
                                         args.backbone, num_classes, num_channels,
                                         args.dropout_rate, args.learning_rate, args.cosine_decay)

    #for im, l in dataset.take(3).as_numpy_iterator():
    #    logging.info("{} {}".format(len(l), im.shape))

    model = train(args, model, optimizer, dataset_train, dataset_val, \
                  train_loss, train_accuracy, val_loss, val_accuracy, \
                  loss_object, epoch, step, args.num_epoch, \
                  args.checkpoint_interval, checkpoint_manager, \
                  checkpoint_dir, save_weights_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='InverseBio training script that performs checkpointing for TensorFlow'
    )

    parser.add_argument(
        '--backbone',
        type=str,
        help='name of the backbone architecture to be used',
        default='dense_net_121'
    )

    parser.add_argument(
        '--width_factor',
        type=float,
        help='coefficient to make wider deep loc',
        default=1.0
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
        '--save_prefix',
        type=str,
        help='prefix for folder name where model is saved',
        default=''
    )

    parser.add_argument(
        '--dropout_rate',
        type=float,
        help='dropout rate to be used in the last layer',
        default=0.0
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        help='learning rate',
        default=0.001
    )

    parser.add_argument(
        '--cosine_decay',
        type=bool,
        help='use cosine decay schedule',
        default=False
    )

    parser.add_argument(
        '--do_mask',
        type=bool,
        help='using masked cells? Alternatively raw data',
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
        '--uniform_sampling',
        type=bool,
        help='sample equal number of proteins from each class',
        default=False
    )

    # parser.add_argument(
    #     '--load_last',
    #     type=bool,
    #     help='Should we load the last model (if training has been interrupted)',
    #     default=False
    # )

    parser.add_argument(
        '--num_epoch',
        type=int,
        help='number of epochs to run',
        required=True
    )

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='path to save and look for the checkpoint file',
        default=None
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        help='batch size per iteration',
        default=128
    )

    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        help='period to take checkpoints in seconds',
        default=1800
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
        '--log_file',
        type=str,
        help='specify the location of the output directory, default stdout',
        default=None
    )

    main(parser.parse_args())
