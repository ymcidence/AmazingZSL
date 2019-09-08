import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
SESS = tf.Session(config=config)

awa_1 = {'sess': SESS,
         'batch_size': 512,
         'set_name': 'AWA1',
         'gan': False,
         'task_name': 'bicls',
         'max_iter': 50000,
         'temp': .5,
         'lamb': 10}

SETTINGS = {
    'awa_1': awa_1
}
