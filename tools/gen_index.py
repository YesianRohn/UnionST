import os
import numpy as np
from tqdm import tqdm
import h5py
import json

def build_font_index(paths):
    all_paths = []
    all_counts = []
    all_ids = []
    all_tables = []

    for path in paths:
        if not os.path.exists(path):
            continue
        font_files = [path]
        if os.path.isdir(path):
            font_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.ttf', '.otf'))]
        all_paths.append(font_files)
        all_counts.append(len(font_files))

        glyphset = set()
        for font_path in tqdm(font_files):
            txt_path = f"{os.path.splitext(font_path)[0]}.txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as fp:
                    glyphset.update(fp.read())
        glyph2id = {g: i for i, g in enumerate(glyphset)}
        all_ids.append(glyph2id)

        table = np.zeros((len(font_files), len(glyph2id)), dtype=bool)
        for idx, font_path in enumerate(font_files):
            txt_path = f"{os.path.splitext(font_path)[0]}.txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as fp:
                    glyphs = fp.read()
                ids = [glyph2id[g] for g in glyphs if g in glyph2id]
                table[idx, ids] = True
        all_tables.append(table)

    with h5py.File("resources/font_index.h5", "w") as f:
        for i, table in enumerate(all_tables):
            f.create_dataset(f"table_{i}", data=table, compression="gzip")
        f.attrs["paths"] = json.dumps(all_paths, ensure_ascii=False)
        f.attrs["counts"] = json.dumps(all_counts)
        for i, glyph2id in enumerate(all_ids):
            f.attrs[f"glyph2id_{i}"] = json.dumps(glyph2id, ensure_ascii=False)

paths = ["resources/filtered_fonts"]
build_font_index(paths)
