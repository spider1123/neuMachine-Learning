"""根据训练集 GT 生成与 sample_submission 相同结构的“伪提交”文件，用于本地评估。"""

import csv
import pandas as pd

from config import CSV_GT


def main(out_path="train_gt_submission.csv"):
    df = pd.read_csv(CSV_GT)
    rows = [("ImageID", "value")]
    for _, row in df.iterrows():
        idx = int(row["data"])
        x = float(row["Fovea_X"])
        y = float(row["Fovea_Y"])
        rows.append((f"{idx}_Fovea_X", x))
        rows.append((f"{idx}_Fovea_Y", y))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved train GT submission-style file to {out_path}")


if __name__ == "__main__":
    main()


