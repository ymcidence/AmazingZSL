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
         'task_name': 'basic_correct',
         'max_iter': 30000,
         'temp': .7,
         'lamb': 10,
         'cls_from': 's',
         'lr': 5e-4,
         'inn_module': inn_module.SimpleINN,
         'cali': .5,
         'depth': 1,
         'norm': False,
         'coupling': 1
         }

awa_1_gan = {'sess': SESS,
             'batch_size': 512,
             'set_name': 'AWA1',
             'task_name': 'correct_v',
             'max_iter': 50000,
             'temp': .5,
             'lamb': 10,
             'cls_from': 's',
             'lr': 2e-4,
             'inn_module': inn_module.SimpleINN,
             'cali': .5,
             'depth': 1,
             'norm': False}

awa_1_igan = {'sess': SESS,
              'batch_size': 512,
              'set_name': 'AWA1',
              'task_name': 'iwg_correct',
              'max_iter': 50000,
              'temp': .5,
              'lamb': 3,
              'lamb2': 10,
              'cls_from': 's',
              'lr': 2e-4,
              'inn_module': inn_module.SimpleINN,
              'cali': .5,
              'depth': 1,
              'norm': False
              }

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
