# -*- coding: utf-8 -*-
"""
阶段 0：下载 MovieLens-1M 数据集

MovieLens 是推荐系统领域最经典的公开数据集，由美国 GroupLens 实验室维护。
1M 版本包含：6040 个用户、约 3900 部电影、1,000,209 条评分（1~5 星）。

运行方式：python src/download_data.py
"""
import io
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# 项目根目录（本文件在 src/ 下，所以取上一级）
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "ml-1m.zip"

    if (DATA_DIR / "ml-1m" / "ratings.dat").exists():
        print("数据已存在，跳过下载：", DATA_DIR / "ml-1m")
        return

    print(f"正在下载 {URL} ...（约 6 MB）")
    urlretrieve(URL, zip_path)
    print("下载完成，正在解压 ...")

    with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()  # 删除压缩包，只留解压后的文件

    print("完成！数据位于：", DATA_DIR / "ml-1m")
    for f in sorted((DATA_DIR / "ml-1m").iterdir()):
        print(f"  - {f.name}  ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
