import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from fontTools.ttLib import TTFont
from pygame import freetype
from tqdm import tqdm
import time

INPUT_DIRS = ["resources/fonts"]
OUTPUT_DIR = "resources/filtered_fonts"
WORKERS = 8 

def search_files(roots, exts=None):
    paths = []
    for root in roots:
        for dir_path, _, file_names in os.walk(root):
            for file_name in file_names:
                file_path = os.path.join(dir_path, file_name)
                file_ext = os.path.splitext(file_name)[1]
                if exts is not None and file_ext.lower() not in exts:
                    continue
                paths.append(file_path)
    return paths

def get_cmap(path):
    cmap = set()
    font = TTFont(path)
    for table in font["cmap"].tables:
        for code, _ in table.cmap.items():
            try:
                char = chr(code)
            except:
                continue
            cmap.add(char)
    return cmap

def font_has_all_diff_glyphs(path):
    try:
        cmap = get_cmap(path)
        freetype.init()
        font = freetype.Font(path)
        font.antialiased = True
        font.pad = True
        font.size = 72

        for c in range(ord('A'), ord('Z')+1):
            upper = chr(c)
            lower = chr(c + 32)
            if upper not in cmap or lower not in cmap:
                continue
            try:
                glyph_upper, _ = font.render_raw(upper)
                glyph_lower, _ = font.render_raw(lower)
            except:
                continue
            if glyph_upper == glyph_lower:
                return False

        return True
    except Exception as e:
        return False

def main():
    start_time = time.time()
    paths = search_files(INPUT_DIRS, exts=[".ttf", ".otf"])
    total_fonts = len(paths)
    diff_fonts = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=WORKERS)
    futures = {}
    for path in paths:
        future = executor.submit(font_has_all_diff_glyphs, path)
        futures[future] = path

    pbar = tqdm(total=total_fonts, desc="Processing fonts", ncols=80)
    for future in as_completed(futures):
        path = futures[future]
        try:
            result = future.result()
        except Exception as e:
            pbar.update(1)
            continue
        if result:
            diff_fonts += 1
            try:
                shutil.copy2(path, os.path.join(OUTPUT_DIR, os.path.basename(path)))
            except Exception as e:
                print(f"Copy failed: {path} -> {OUTPUT_DIR}: {e}")
        pbar.update(1)
    pbar.close()
    executor.shutdown()
    print(f"\nTotal fonts: {total_fonts}")
    print(f"Fonts with ALL different glyphs (A/a-Z/z): {diff_fonts}")
    print(f"Fonts copied to: {OUTPUT_DIR}")
    print(f"{time.time() - start_time:.2f} seconds elapsed")

if __name__ == "__main__":
    main()