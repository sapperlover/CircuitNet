import os
from multiprocessing import Process
import argparse
from src.util import divide_n
from src.read import Paraser


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('expected a boolean value')


class ArgParaser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_root', default='./data', help='the parent dir of log dirs')
        self.parser.add_argument('--save_path', default='./out', help='save path')
        self.parser.add_argument('--process_capacity', type=int, default=2, help='number of process for multi process')
        self.parser.add_argument('--plot', type=str2bool, default=True, help='plot the results in $save_path/visual')
        self.parser.add_argument('--debug', type=str2bool, default=False, help='disable multi process to use pdb')
        self.parser.add_argument('--final_test', type=str2bool, default=False, help='prevent using static_ir to mimic the final environment')
        self.parser.add_argument('--max_cases', type=int, default=None, help='optional limit on number of cases to process')


def read(read_list, arg):

    for path in read_list:
        save_name = path
        path = os.path.join(arg.data_root, path)
        process_log = Paraser(path, arg, save_name)
        print(save_name)
        process_log.get_IR_drop_features()     


if __name__ == '__main__':
    argp = ArgParaser()
    arg = argp.parser.parse_args()
    if not os.path.exists(arg.save_path):
        os.makedirs(arg.save_path)


    read_list = [
        name for name in sorted(os.listdir(arg.data_root))
        if os.path.isdir(os.path.join(arg.data_root, name))
    ]
    if arg.max_cases is not None:
        read_list = read_list[:arg.max_cases]
    nlist = divide_n(read_list, arg.process_capacity) 

    if arg.debug:
        read(read_list, arg)
    else:
        process = []
        for divided_list in nlist:
            p = Process(target=read, args=(divided_list, arg))
            process.append(p)
        for p in process:
            p.start()
        for p in process:
            p.join()
