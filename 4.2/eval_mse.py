import argparse
import csv
import numpy as np


def read_submission(path):
    ids = []
    vals = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            ids.append(row[0])
            vals.append(float(row[1]))
    return ids, np.array(vals, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="预测 submission 文件路径")
    parser.add_argument("--gt", required=True, help="GT submission 风格文件路径")
    args = parser.parse_args()

    ids_pred, vals_pred = read_submission(args.pred)
    ids_gt, vals_gt = read_submission(args.gt)

    if ids_pred != ids_gt:
        raise ValueError("预测与 GT 的 ImageID 序列不一致，请检查生成顺序。")

    mse = float(np.mean((vals_pred - vals_gt) ** 2))
    print(f"MSE = {mse:.6f}")


if __name__ == "__main__":
    main()




