import tensorflow as tf
from util.flow import inn_module

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
SESS = tf.Session(config=config)

awa_1 = {'sess': SESS,
         'batch_size': 512,
         'set_name': 'AWA1',
         'gan': False,
         'task_name': 'hehe',
         'max_iter': 50000,
         'temp': .5,
         'lamb': 10,
         'cls_from': 's',
         'lr': 1e-3}

awa_1_gan = {'sess': SESS,
             'batch_size': 512,
             'set_name': 'AWA1',
             'task_name': 'simple',
             'max_iter': 50000,
             'temp': .5,
             'lamb': 10,
             'cls_from': 's',
             'lr': 2e-4,
             'inn_module': inn_module.SimpleINN,
             'cali': .5}

awa_1_igan = {'sess': SESS,
              'batch_size': 512,
              'set_name': 'AWA1',
              'task_name': 'iwg_l10',
              'max_iter': 50000,
              'temp': .5,
              'lamb': 3,
              'lamb2': 10,
              'cls_from': 's',
              'lr': 2e-4,
              'cali': .6}

awa_1_bi = {'sess': SESS,
            'batch_size': 512,
            'set_name': 'AWA1',
            'task_name': 'bi_k4',
            'max_iter': 50000,
            'temp': .5,
            'lamb': 10,
            'cls_from': 's',
            'lr': 2e-4,
            'inn_module': inn_module.SimpleINN,
            'cali': .5}
awa_1_comp = {'sess': SESS,
              'batch_size': 512,
              'set_name': 'AWA1',
              'task_name': 'complex_ns_t03',
              'max_iter': 50000,
              'temp': .3,
              'lamb': 10,
              'cls_from': 's',
              'lr': 2e-4,
              'inn_module': inn_module.SimpleINN,
              'cali': .5}

SETTINGS = {'awa_1': awa_1,
            'awa_1_gan': awa_1_gan,
            'awa_1_igan': awa_1_igan,
            'awa_1_bi': awa_1_bi,
            'awa_1_comp': awa_1_comp
            }
