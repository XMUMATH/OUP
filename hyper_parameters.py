import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

## The following flags are related to save paths, tensorboard outputs and screen outputs

tf.app.flags.DEFINE_string('version', 'test_110_L2_UP_val5k_0.9cut_2', '''A version number defining the directory to save 
logs and checkpoints''')#test_50_noBN_Xreg_05
tf.app.flags.DEFINE_integer('report_freq', 400, '''Steps takes to output errors on the screen
and write summaries''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's
moving average shown on tensorboard''')


## The following flags define hyper-parameters regards training

tf.app.flags.DEFINE_integer('train_steps', 20000, '''Total steps that you want to train''')
tf.app.flags.DEFINE_boolean('is_full_validation', False , '''Validation w/ full validation set or
a random batch''')
tf.app.flags.DEFINE_integer('train_batch_size', 128, '''Train batch size''')
tf.app.flags.DEFINE_integer('validation_batch_size', 5000, '''Validation batch size, better to be
a divisor of 10000 for this task''')
tf.app.flags.DEFINE_integer('test_batch_size', 128, '''Test batch size''')

tf.app.flags.DEFINE_float('init_lr', 1e-3, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.1, '''How much to decay the learning rate each
time''')
tf.app.flags.DEFINE_integer('decay_step0', 32000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 48000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('fine_tune_point', 90000, '''At which step to fune tune''')
tf.app.flags.DEFINE_boolean('is_use_fine_tune', False, '''Whether to use fine tune''')
## The following flags define hyper-parameters modifying the training network

tf.app.flags.DEFINE_integer('num_residual_blocks', 18, '''How many residual blocks do you want''')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, '''scale for l2 regularization''')


## The following flags are related to data-augmentation

tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on
each side of the image''')


## If you want to load a checkpoint and continue training

tf.app.flags.DEFINE_string('ckpt_path', './logs_test_110_Fc_L2_val5k_0.9cut/model.ckpt-79999', '''Checkpoint
directory to restore''')
tf.app.flags.DEFINE_boolean('is_use_ckpt', True, '''Whether to load a checkpoint and continue
training''')

#tf.app.flags.DEFINE_string('test_ckpt_path', './logs_test_my_50_baseline/model.ckpt-79999', '''Checkpoint
#directory to restore''')


train_dir = 'logs_' + FLAGS.version + '/'
