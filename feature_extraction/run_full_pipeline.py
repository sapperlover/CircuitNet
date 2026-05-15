import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile


ARCHIVE_SUFFIXES = (
    '.tar',
    '.tar.gz',
    '.tgz',
    '.tar.bz2',
    '.tbz2',
    '.zip',
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('expected a boolean value')


def split_csv(value):
    if value is None:
        return None
    if isinstance(value, list):
        items = []
        for item in value:
            items.extend(split_csv(item))
        return items
    return [item.strip() for item in value.split(',') if item.strip()]


def is_archive(path):
    return os.path.isfile(path) and path.endswith(ARCHIVE_SUFFIXES)


def list_techs(data_root):
    techs = []
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if os.path.isdir(path):
            techs.append(name)
    return techs


def list_archives(path):
    if is_archive(path):
        return [path]

    archives = []
    for root, _, files in os.walk(path):
        for filename in sorted(files):
            file_path = os.path.join(root, filename)
            if is_archive(file_path):
                archives.append(file_path)
    return sorted(archives)


def has_any(path, names):
    return any(os.path.exists(os.path.join(path, name)) for name in names)


def is_case_dir(path, final_test=False):
    has_power = has_any(
        path,
        (
            'pulpino_top.inst.power.rpt.gz',
            'pulpino_top.inst.power.rpt',
            'NV_nvdla.inst.power.rpt.gz',
            'NV_nvdla.inst.power.rpt',
            'Vortex.inst.power.rpt.gz',
            'Vortex.inst.power.rpt',
        ),
    )
    has_eff_res = has_any(path, ('eff_res.rpt.gz', 'eff_res.rpt'))
    has_static_ir = final_test or has_any(path, ('static_ir.gz', 'static_ir'))
    return has_power and has_eff_res and has_static_ir


def discover_case_dirs(root, final_test=False):
    case_dirs = []
    for dirpath, dirnames, _ in os.walk(root):
        if is_case_dir(dirpath, final_test=final_test):
            case_dirs.append(dirpath)
            dirnames[:] = []
    return sorted(case_dirs)


def link_cases(case_dirs, case_root, used_names=None):
    os.makedirs(case_root, exist_ok=True)
    linked = []
    if used_names is None:
        used_names = set()
    for case_dir in case_dirs:
        base_name = os.path.basename(case_dir.rstrip(os.sep))
        link_name = base_name
        suffix = 1
        while link_name in used_names or os.path.exists(os.path.join(case_root, link_name)):
            link_name = '{}_{}'.format(base_name, suffix)
            suffix += 1
        used_names.add(link_name)
        dst = os.path.join(case_root, link_name)
        os.symlink(os.path.abspath(case_dir), dst)
        linked.append(dst)
    return linked


def run_command(command, cwd):
    print('\n$ {}'.format(' '.join(command)))
    subprocess.run(command, cwd=cwd, check=True)


def extract_archive(archive, extract_root):
    os.makedirs(extract_root, exist_ok=True)
    archive_out = os.path.join(extract_root, os.path.splitext(os.path.basename(archive))[0])
    os.makedirs(archive_out, exist_ok=True)
    print('extracting {} -> {}'.format(archive, archive_out))
    shutil.unpack_archive(archive, archive_out)
    return archive_out


def process_one_tech(args, script_dir, tech, temp_root):
    tech_source = os.path.join(args.data_root, tech)
    archives = list_archives(tech_source)
    if not archives:
        raise FileNotFoundError('no archive found for tech {} under {}'.format(tech, tech_source))

    tech_temp = os.path.join(temp_root, tech)
    used_case_names = set()
    processed_cases = 0

    for archive_index, archive in enumerate(archives):
        if args.max_cases is not None and processed_cases >= args.max_cases:
            break

        archive_work = os.path.join(tech_temp, 'archive_{:05d}'.format(archive_index))
        extract_root = os.path.join(archive_work, 'extracted')
        case_root = os.path.join(archive_work, 'cases')

        if os.path.exists(archive_work):
            shutil.rmtree(archive_work)

        try:
            archive_out = extract_archive(archive, extract_root)
            case_dirs = discover_case_dirs(archive_out, final_test=args.final_test)
            if args.max_cases is not None:
                remaining_cases = args.max_cases - processed_cases
                case_dirs = case_dirs[:remaining_cases]
            if not case_dirs:
                print('warning: no valid case directory found in {}'.format(archive))
                continue

            link_cases(case_dirs, case_root, used_names=used_case_names)
            print(
                'tech {}: archive {}/{} has {} cases, {} total'.format(
                    tech,
                    archive_index + 1,
                    len(archives),
                    len(case_dirs),
                    processed_cases + len(case_dirs),
                )
            )

            run_command(
                [
                    sys.executable,
                    'process_data.py',
                    '--data_root',
                    case_root,
                    '--save_path',
                    os.path.join(args.out_root, tech),
                    '--process_capacity',
                    str(args.process_capacity),
                    '--plot',
                    str(args.plot).lower(),
                    '--debug',
                    str(args.debug).lower(),
                    '--final_test',
                    str(args.final_test).lower(),
                ],
                cwd=script_dir,
            )
            processed_cases += len(case_dirs)
        finally:
            if not args.keep_temp and os.path.exists(archive_work):
                print('removing extracted data {}'.format(archive_work))
                shutil.rmtree(archive_work)

    if processed_cases == 0:
        raise RuntimeError('no valid case directory found after extracting {}'.format(tech))

    run_command(
        [
            sys.executable,
            'generate_training_set.py',
            '--data_path',
            os.path.join(args.out_root, tech),
            '--save_path',
            args.training_set_root,
            '--prefix',
            tech,
            '--process_capacity',
            str(args.process_capacity),
        ],
        cwd=script_dir,
    )


def collect_samples(script_dir, out_root, training_set_root, techs):
    samples = []
    for tech in techs:
        feature_dir = os.path.join(script_dir, training_set_root, tech, 'feature')
        label_dir = os.path.join(script_dir, training_set_root, tech, 'label')
        if not os.path.isdir(feature_dir):
            raise FileNotFoundError('missing feature dir for tech {}: {}'.format(tech, feature_dir))
        if not os.path.isdir(label_dir):
            raise FileNotFoundError('missing label dir for tech {}: {}'.format(tech, label_dir))

        for filename in sorted(os.listdir(feature_dir)):
            feature_path = os.path.join(feature_dir, filename)
            label_path = os.path.join(label_dir, filename)
            if not os.path.isfile(feature_path) or not os.path.isfile(label_path):
                continue

            instance_name = os.path.splitext(filename)[0] + '.npz'
            samples.append(
                (
                    tech,
                    os.path.join(training_set_root, tech, 'feature', filename),
                    os.path.join(training_set_root, tech, 'label', filename),
                    os.path.join(out_root, tech, 'features', 'instance_count', filename),
                    os.path.join(out_root, tech, 'features', 'instance_IR_drop', filename),
                    os.path.join(out_root, tech, 'features', 'instance_name', instance_name),
                )
            )
    return samples


def write_csv(args, script_dir, csv_techs):
    train_techs = split_csv(args.train_techs) if args.train_techs is not None else None
    test_techs = split_csv(args.test_techs) if args.test_techs is not None else None

    if train_techs is not None or test_techs is not None:
        if train_techs is None or test_techs is None:
            raise ValueError('--train_techs and --test_techs must be used together')
        train_rows = collect_samples(script_dir, args.out_root, args.training_set_root, train_techs)
        test_rows = collect_samples(script_dir, args.out_root, args.training_set_root, test_techs)
        if not train_rows:
            raise RuntimeError('no train samples found for techs: {}'.format(', '.join(train_techs)))
        if not test_rows:
            raise RuntimeError('no test samples found for techs: {}'.format(', '.join(test_techs)))
    else:
        samples = collect_samples(script_dir, args.out_root, args.training_set_root, csv_techs)
        if not samples:
            raise RuntimeError('no samples found for csv techs: {}'.format(', '.join(csv_techs)))

        random.seed(args.seed)
        random.shuffle(samples)

        train_rows = []
        test_rows = []
        if len(samples) == 1:
            train_rows.append(samples[0])
            test_rows.append(samples[0])
        else:
            for sample in samples:
                if random.random() <= args.split_ratio:
                    train_rows.append(sample)
                else:
                    test_rows.append(sample)
            if not train_rows:
                train_rows.append(test_rows.pop())
            if not test_rows:
                test_rows.append(train_rows[-1])

    train_csv = os.path.join(script_dir, args.train_csv)
    test_csv = os.path.join(script_dir, args.test_csv)

    with open(train_csv, 'w', newline='') as f_train:
        writer = csv.writer(f_train)
        for _, feature_path, label_path, _, _, _ in train_rows:
            writer.writerow([feature_path, label_path])

    with open(test_csv, 'w', newline='') as f_test:
        writer = csv.writer(f_test)
        for _, feature_path, label_path, instance_count, instance_ir_drop, instance_name in test_rows:
            writer.writerow([feature_path, label_path, instance_count, instance_ir_drop, instance_name])

    print('wrote {} train rows to {}'.format(len(train_rows), train_csv))
    print('wrote {} test rows to {}'.format(len(test_rows), test_csv))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run archive extraction, feature extraction, training-set packing, and CSV generation.'
    )
    parser.add_argument('--data_root', default='./data', help='directory containing per-tech archive folders')
    parser.add_argument('--techs', nargs='+', default=None, help='tech names to extract, e.g. riscy riscy_fpu')
    parser.add_argument('--csv_techs', nargs='+', default=None, help='tech names used to generate train/test csv')
    parser.add_argument('--train_techs', nargs='+', default=None, help='tech names used only for train csv rows')
    parser.add_argument('--test_techs', nargs='+', default=None, help='tech names used only for test csv rows')
    parser.add_argument('--out_root', default='./out', help='feature extraction output root')
    parser.add_argument('--training_set_root', default='./training_set', help='packed training-set output root')
    parser.add_argument('--train_csv', default='./train.csv', help='output train csv path')
    parser.add_argument('--test_csv', default='./test.csv', help='output test csv path')
    parser.add_argument('--process_capacity', type=int, default=2, help='number of worker processes')
    parser.add_argument('--plot', type=str2bool, default=False, help='save feature heatmaps during extraction')
    parser.add_argument('--debug', type=str2bool, default=False, help='run process_data in debug mode')
    parser.add_argument('--final_test', type=str2bool, default=False, help='process data without static_ir labels')
    parser.add_argument('--max_cases', type=int, default=None, help='optional case limit per tech')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='train split ratio')
    parser.add_argument('--seed', type=int, default=230907, help='csv split random seed')
    parser.add_argument('--temp_root', default=None, help='temporary extraction root')
    parser.add_argument('--keep_temp', action='store_true', help='keep temporary extracted data')
    parser.add_argument('--csv_only', action='store_true', help='only generate train/test csv from existing outputs')
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = os.path.abspath(os.path.join(script_dir, args.data_root))

    csv_techs = split_csv(args.csv_techs) if args.csv_techs is not None else None
    train_techs = split_csv(args.train_techs) if args.train_techs is not None else None
    test_techs = split_csv(args.test_techs) if args.test_techs is not None else None
    if args.techs is None:
        if args.csv_only:
            if train_techs is not None or test_techs is not None:
                techs = sorted(set((train_techs or []) + (test_techs or [])))
            elif csv_techs is not None:
                techs = csv_techs
            else:
                techs = list_techs(args.data_root)
        else:
            techs = list_techs(args.data_root)
    else:
        techs = split_csv(args.techs)
    if not techs:
        raise RuntimeError('no tech selected')

    csv_techs = csv_techs if csv_techs is not None else techs

    temp_root = args.temp_root
    created_temp = False
    if temp_root is None:
        temp_root = tempfile.mkdtemp(prefix='circuitnet_extract_')
        created_temp = True
    else:
        temp_root = os.path.abspath(os.path.join(script_dir, temp_root))
        os.makedirs(temp_root, exist_ok=True)

    try:
        if not args.csv_only:
            for tech in techs:
                process_one_tech(args, script_dir, tech, temp_root)
        write_csv(args, script_dir, csv_techs)
    finally:
        if not args.keep_temp and created_temp and os.path.exists(temp_root):
            print('removing temporary directory {}'.format(temp_root))
            shutil.rmtree(temp_root)
        elif not args.keep_temp:
            print('temporary directory kept because it was user-provided: {}'.format(temp_root))


if __name__ == '__main__':
    main()
