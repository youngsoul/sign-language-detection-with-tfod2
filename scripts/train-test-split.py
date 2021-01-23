import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import model_config as mc

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images-dir", required=True, help="path to the input images root with subfolders")
    ap.add_argument("-o", "--output-dir", required=True, help="path to the output directory where test and train directories will go")
    ap.add_argument("--test-size", type=float, required=False, default=0.2, help="fraction for test size dataset.  default: 0.2")
    ap.add_argument("--holdout-size", type=float, required=False, default=0, help="fraction for holdout size dataset.  default: 0")

    args = vars(ap.parse_args())

    path = Path(args['images_dir'])

    output_path = Path(args['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    train_dir = output_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir = output_path / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir = output_path / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    # get all of the subdirectories and collect the images from all of the subdirectories
    dirs = [e for e in path.iterdir() if e.is_dir()]
    for dir in dirs:
        print(dir.name)
        files = [f for f in dir.iterdir() if f.is_file()]
        # print(files)
        (train, test) = train_test_split(files, test_size=args["test_size"])

        holdout = None
        if args['holdout_size'] > 0:
            (train, holdout) = train_test_split(train, test_size=args['holdout_size'])
            print(f"train/test/holdout: {len(train)}, {len(test)}, {len(holdout)}")
        else:
            print(f"train/test/holdout: {len(train)}, {len(test)}, 0")

        for train_file in train:
            train_file.rename(train_dir / train_file.name)

        for test_file in test:
            test_file.rename(test_dir / test_file.name)

        if args['holdout_size'] > 0 and holdout is not None:
            for holdout_file in holdout:
                holdout_file.rename(holdout_dir / holdout_file.name)



