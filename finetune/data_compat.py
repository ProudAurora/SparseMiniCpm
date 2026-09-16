"""
数据加载兼容层
支持新版 datasets 保存的 Arrow 文件 (pyarrow ipc stream format)
当 load_from_disk 失败时, 自动回退到 pyarrow 直接读取
"""
import os
import glob

def load_dataset_from_disk_compat(data_path: str):
    """兼容加载 Arrow 数据集, 优先用 datasets.load_from_disk, 失败则用 pyarrow 直接读

    Args:
        data_path: Arrow 数据目录
    Returns:
        datasets.Dataset
    """
    from datasets import load_from_disk, Dataset

    # Try standard method first
    try:
        ds = load_from_disk(data_path)
        return ds
    except Exception as e:
        print(f"load_from_disk failed ({e}), falling back to pyarrow direct read...")

    # Fallback: use pyarrow to read ipc stream files
    import pyarrow as pa

    # Find all arrow files
    arrow_files = sorted(glob.glob(os.path.join(data_path, "data-*.arrow")))
    if not arrow_files:
        raise FileNotFoundError(f"No arrow files found in {data_path}")

    print(f"Reading {len(arrow_files)} arrow files with pyarrow...")
    tables = []
    for f in arrow_files:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
        tables.append(table)
        print(f"  {os.path.basename(f)}: {len(table)} rows")

    # Concatenate all tables
    full_table = pa.concat_tables(tables)
    print(f"Total: {len(full_table)} rows")
    print(f"Columns: {full_table.column_names}")

    # Convert to datasets Dataset via from_dict (avoids metadata parsing issues)
    data_dict = {col: full_table.column(col).to_pylist() for col in full_table.column_names}
    ds = Dataset.from_dict(data_dict)
    print(f"Dataset created: {len(ds)} samples")
    print(f"Features: {ds.features}")
    return ds
