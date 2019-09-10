from model.basic_model import ClsModel as Model
from absl import flags
from absl import app
from config import basic_config as config

flags.DEFINE_string("c", 'awa_1', "config")
FLAGS = flags.FLAGS


def main(_):
    settings = config.SETTINGS.get(FLAGS.c)
    model = Model(**settings)
    if settings.get('restore_file') is not None:
        restore_file = settings.get('restore_file')
        model.train(restore_file=restore_file, task=settings.get('task_name'), max_iter=settings.get('max_iter'))
    else:
        model.train(task=settings.get('task_name'), max_iter=settings.get('max_iter'))
    return


if __name__ == '__main__':
    app.run(main)
