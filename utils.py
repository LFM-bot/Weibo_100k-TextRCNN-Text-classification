import os
import logging


def set_logger(args):
    """
    Write logs to checkpoint and console
    """
    save_path = args.log_save

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    log_file = os.path.join(save_path, '%s_ratio%.1f_%d.log' % (args.model_name,
                                                                args.data_ratio,
                                                                args.run_time))
    for i in range(10):
        if not os.path.isfile(log_file):
            break
        log_file = os.path.join(save_path, '%s_ratio%.1f_%d.log' % (args.model_name,
                                                                    args.data_ratio,
                                                                    args.run_time + i))
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info('log save at : {}'.format(log_file))
    logging.info(args)


