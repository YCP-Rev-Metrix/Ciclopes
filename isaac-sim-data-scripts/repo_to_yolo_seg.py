import os, re, argparse, shutil, json
from glob import glob
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# ---------- CONFIGURATION (edit if needed) ----------

# NOTE: Mappings will be loaded dynamically from JSON files in the dataset.
# This script now auto-detects the mapping files from frame 0.
# If you need to override, edit the TARGET_CLASSES_ORDERED below.

# Keep only these classes (normalized, lowercase, first token before comma)
TARGET_CLASSES_ORDERED = ["lane", "ball"]  # -> YOLO ids 0: lane, 1: ball

# Filename token patterns to auto-detect files in ONE folder
RGB_PATTERNS       = [r"rgb", r"_rgb"]
INSTANCE_PATTERNS  = [r"instance_segmentation", r"instance", r"_inst", r"_instance_seg"]
SEMANTIC_PATTERNS  = [r"semantic_segmentation", r"semantic", r"_sem", r"_semantic_seg"]
MAPPING_PATTERNS   = [r"instance_segmentation_mapping"]  # for instance->class mapping
SEMANTICS_PATTERNS = [r"instance_segmentation_semantics_mapping"]  # for instance->semantics
SEMANTIC_LABELS_PATTERNS = [r"semantic_segmentation_labels"]  # for semantic id->class

# Polygon simplification & filtering
EPSILON_RATIO      = 0.0025   # approxPolyDP epsilon = ratio * perimeter
MIN_CONTOUR_POINTS = 3        # must be at least a triangle
MIN_AREA_PX        = 10       # skip tiny specks

# ----------------------------------------------------

def normalize_class_name(raw):
    if not raw:
        return None
    # take the part before comma, lowercase, strip
    name = str(raw).split(",")[0].strip().lower()
    # Map synonyms
    if name == "sphere":
        name = "ball"
    return name

def rgba_tuple_from_str(s):
    """Convert string like '(240, 4, 111, 255)' to tuple (240, 4, 111, 255)"""
    s = s.strip("()")
    return tuple(map(int, s.split(",")))

def load_mappings_from_json(json_path):
    """Load and parse JSON mapping file. Returns dict with appropriate keys."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    parsed = {}
    for key, value in data.items():
        # Check if key is a tuple string like "(240, 4, 111, 255)"
        if key.startswith("(") and key.endswith(")"):
            parsed[rgba_tuple_from_str(key)] = value
        else:
            # Numeric ID as string
            parsed[int(key)] = value
    return parsed

CLASS_TO_YOLO = {c: i for i, c in enumerate(TARGET_CLASSES_ORDERED)}
TARGET_SET = set(TARGET_CLASSES_ORDERED)

def any_token_match(name, patterns):
    return any(re.search(p, name, flags=re.IGNORECASE) for p in patterns)

def stem_key(p):
    """Create a pairing stem by removing the last matched token like _rgb/_instance/_semantic and extension."""
    name = p.stem
    # remove well-known trailing tokens - check longer patterns first!
    all_patterns = (SEMANTICS_PATTERNS + MAPPING_PATTERNS + SEMANTIC_LABELS_PATTERNS + 
                    INSTANCE_PATTERNS + SEMANTIC_PATTERNS + RGB_PATTERNS)
    for pat in all_patterns:
        # Remove pattern at end or before underscore/dash/dot
        name = re.sub(f"{pat}$", "", name, flags=re.IGNORECASE)
        name = re.sub(f"{pat}(?=[._-])", "", name, flags=re.IGNORECASE)
    return name

def find_files_and_mappings(in_dir):
    """Find RGB, instance, semantic images and their mapping JSONs."""
    files = [Path(p) for p in glob(os.path.join(in_dir, "*")) if os.path.isfile(p)]
    
    # group by detected type
    rgbs, insts, sems = {}, {}, {}
    inst_mappings, inst_semantics, sem_labels = {}, {}, {}
    
    for f in files:
        base = f.stem
        if any_token_match(base, SEMANTICS_PATTERNS):
            # Must check this BEFORE instance patterns (more specific)
            inst_semantics.setdefault(stem_key(f), []).append(f)
        elif any_token_match(base, MAPPING_PATTERNS):
            inst_mappings.setdefault(stem_key(f), []).append(f)
        elif any_token_match(base, SEMANTIC_LABELS_PATTERNS):
            sem_labels.setdefault(stem_key(f), []).append(f)
        elif any_token_match(base, RGB_PATTERNS):
            rgbs.setdefault(stem_key(f), []).append(f)
        elif any_token_match(base, INSTANCE_PATTERNS):
            insts.setdefault(stem_key(f), []).append(f)
        elif any_token_match(base, SEMANTIC_PATTERNS):
            sems.setdefault(stem_key(f), []).append(f)
    
    # build sets by common stem
    stems = sorted(set(rgbs.keys()) | set(insts.keys()) | set(sems.keys()))
    bundles = []
    for s in stems:
        if s in rgbs and s in insts and s in sems:
            # Each bundle: (rgb, inst, sem, inst_semantics_json, semantic_labels_json, stem)
            inst_sem_json = inst_semantics[s][0] if s in inst_semantics else None
            sem_lbl_json = sem_labels[s][0] if s in sem_labels else None
            bundles.append((
                rgbs[s][0], 
                insts[s][0], 
                sems[s][0],
                inst_sem_json,
                sem_lbl_json,
                s
            ))
    
    return bundles

def load_rgba_instance_mask(path):
    """Load instance mask as RGBA array."""
    im = Image.open(path)
    arr = np.array(im)
    # Ensure it's RGBA (H, W, 4)
    if arr.ndim == 2:
        # Grayscale - expand to RGBA
        arr = np.stack([arr] * 4, axis=-1)
    elif arr.shape[2] == 3:
        # RGB - add alpha channel
        arr = np.dstack([arr, np.full((arr.shape[0], arr.shape[1]), 255, dtype=np.uint8)])
    return arr

def load_gray(path):
    """Load semantic mask as grayscale array."""
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 3:
        # Convert to single channel
        arr = arr[:, :, 0]
    return arr

def get_unique_rgba_colors(rgba_mask):
    """Get unique RGBA colors from the mask. Returns list of (r,g,b,a) tuples."""
    h, w, c = rgba_mask.shape
    reshaped = rgba_mask.reshape(-1, c)
    unique_colors = np.unique(reshaped, axis=0)
    # Convert to regular Python tuples with int values
    return [tuple(int(x) for x in color) for color in unique_colors]

def majority_semantic_for_rgba(inst_mask, sem_mask, rgba_color):
    """Find the majority semantic ID for pixels matching the given RGBA color."""
    # Create boolean mask for this RGBA color
    mask = np.all(inst_mask == rgba_color, axis=-1)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    vals = sem_mask[ys, xs]
    # ignore 0/1 (background/unlabelled) when voting if possible
    vals = vals[(vals != 0) & (vals != 1)]
    if vals.size == 0:
        return None
    return int(np.bincount(vals).argmax())

def contours_from_rgba_color(inst_mask, rgba_color):
    """Extract contours for pixels matching the given RGBA color."""
    # Create binary mask for this RGBA color
    mask = np.all(inst_mask == rgba_color, axis=-1).astype(np.uint8) * 255
    if mask.sum() < MIN_AREA_PX:
        return []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA_PX:
            continue
        peri = cv2.arcLength(c, True)
        eps = max(1.0, EPSILON_RATIO * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) >= MIN_CONTOUR_POINTS:
            out.append(approx.reshape(-1, 2))
    return out

def write_yolo_seg(txt_path, objects):
    # objects: list of (yolo_class_id, Nx2 polygon in normalized coords)
    with open(txt_path, "w") as f:
        for cls_id, poly in objects:
            flat = " ".join([f"{x:.6f} {y:.6f}" for x, y in poly])
            f.write(f"{cls_id} {flat}\n")

def process_single_frame_wrapper(bundle_data, images_dir, labels_dir):
    """Wrapper for multiprocessing - unpacks bundle and calls process_bundle"""
    rgb_p, inst_p, sem_p, inst_sem_json, sem_lbl_json, s = bundle_data
    try:
        process_bundle(rgb_p, inst_p, sem_p, inst_sem_json, sem_lbl_json, images_dir, labels_dir, s)
        return True, s
    except Exception as e:
        return False, f"{s}: {str(e)}"

def process_bundle(rgb_path, inst_path, sem_path, inst_sem_json_path, sem_lbl_json_path, 
                    out_img_dir, out_lbl_dir, stem):
    """Process one image bundle (rgb, instance, semantic + JSON mappings)."""
    # load
    img = Image.open(rgb_path).convert("RGB")
    W, H = img.size
    inst = load_rgba_instance_mask(inst_path)
    sem  = load_gray(sem_path)

    # Load mappings from JSON files if available
    inst_rgba_to_semantics = {}
    sem_id_to_class = {}
    
    if inst_sem_json_path and inst_sem_json_path.exists():
        inst_rgba_to_semantics = load_mappings_from_json(inst_sem_json_path)
    
    if sem_lbl_json_path and sem_lbl_json_path.exists():
        sem_id_to_class = load_mappings_from_json(sem_lbl_json_path)

    # copy image
    out_img = Path(out_img_dir) / f"{stem}.jpg"
    img.save(out_img, quality=95)

    # Get unique RGBA colors in instance mask
    unique_colors = get_unique_rgba_colors(inst)
    
    # Filter out background (0,0,0,0) and unlabelled (0,0,0,255)
    background_colors = [(0, 0, 0, 0), (0, 0, 0, 255)]
    unique_colors = [c for c in unique_colors if c not in background_colors]

    objects = []
    for rgba_color in unique_colors:
        # 1) Try to get class name from instance RGBA -> semantics mapping
        c_name = None
        if rgba_color in inst_rgba_to_semantics:
            semantics_info = inst_rgba_to_semantics[rgba_color]
            if isinstance(semantics_info, dict):
                c_name = semantics_info.get("class")
            else:
                c_name = semantics_info
        
        # 2) If not found, vote from semantic image
        if not c_name:
            sem_id = majority_semantic_for_rgba(inst, sem, rgba_color)
            if sem_id is not None and sem_id in sem_id_to_class:
                sem_info = sem_id_to_class[sem_id]
                if isinstance(sem_info, dict):
                    c_name = sem_info.get("class")
                else:
                    c_name = sem_info

        if not c_name:
            continue

        # normalize & filter to targets
        c_name = normalize_class_name(c_name)
        if c_name not in TARGET_SET:
            continue

        yolo_id = CLASS_TO_YOLO[c_name]
        for poly in contours_from_rgba_color(inst, rgba_color):
            # normalize to 0..1
            poly = poly.astype(np.float32)
            poly[:, 0] /= W
            poly[:, 1] /= H
            objects.append((yolo_id, poly))

    # write txt
    out_lbl = Path(out_lbl_dir) / f"{stem}.txt"
    if objects:
        write_yolo_seg(out_lbl, objects)
    else:
        # empty file is valid; or skip—Ultralytics can handle images without labels in training
        open(out_lbl, "w").close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Root folder containing all files together")
    ap.add_argument("--out_dir", required=True, help="Output dataset folder")
    ap.add_argument("--train_val_split", type=float, default=1.0,
                    help="Fraction to put in train (0<split<=1). If <1, remainder goes to val.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of parallel workers (default: CPU count - 1). Use 1 for single-threaded.")
    args = ap.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # folders
    images_dir = os.path.join(out_dir, "images")
    labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print("Finding image bundles...")
    bundles = find_files_and_mappings(in_dir)
    if not bundles:
        raise SystemExit("No (rgb, instance, semantic) bundles found. Adjust patterns at top of script.")

    print(f"Found {len(bundles)} frames")

    # Determine number of workers
    if args.workers is None:
        num_workers = max(1, cpu_count() - 1)
    else:
        num_workers = max(1, args.workers)
    
    # Process frames
    if num_workers == 1:
        # Single-threaded processing
        print("Processing frames (single-threaded)...")
        for bundle in tqdm(bundles, desc="Converting frames", unit="frame"):
            rgb_p, inst_p, sem_p, inst_sem_json, sem_lbl_json, s = bundle
            process_bundle(rgb_p, inst_p, sem_p, inst_sem_json, sem_lbl_json, images_dir, labels_dir, s)
    else:
        # Multi-threaded processing
        print(f"Processing frames with {num_workers} workers...")
        process_func = partial(process_single_frame_wrapper, images_dir=images_dir, labels_dir=labels_dir)
        
        successful = 0
        failed = 0
        failed_frames = []
        
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, bundles),
                total=len(bundles),
                desc="Converting frames",
                unit="frame"
            ))
        
        for success, info in results:
            if success:
                successful += 1
            else:
                failed += 1
                failed_frames.append(info)
        
        if failed > 0:
            print(f"\nWARNING: {failed} frames failed to process:")
            for frame in failed_frames[:10]:  # Show first 10
                print(f"  - {frame}")
            if len(failed_frames) > 10:
                print(f"  ... and {len(failed_frames) - 10} more")
        
        print(f"\nProcessed {successful}/{len(bundles)} frames successfully")

    # Check if split already done
    already_split = (Path(out_dir) / "images" / "train").exists()
    
    # optional split into images/train, images/val style
    if 0 < args.train_val_split < 1.0 and not already_split:
        print(f"\nApplying {args.train_val_split:.0%}/{(1-args.train_val_split):.0%} train/val split...")
        train_img = Path(out_dir) / "images" / "train"
        val_img   = Path(out_dir) / "images" / "val"
        train_lbl = Path(out_dir) / "labels" / "train"
        val_lbl   = Path(out_dir) / "labels" / "val"
        for p in (train_img, val_img, train_lbl, val_lbl):
            p.mkdir(parents=True, exist_ok=True)

        all_imgs = sorted((Path(images_dir)).glob("*.jpg"))
        n_train = int(len(all_imgs) * args.train_val_split)
        train_set = set(p.stem for p in all_imgs[:n_train])

        # move files
        for imgf in tqdm(all_imgs, desc="Organizing dataset", unit="file"):
            stem = imgf.stem
            lblf = Path(labels_dir) / f"{stem}.txt"
            if stem in train_set:
                shutil.move(str(imgf), train_img / imgf.name)
                if lblf.exists():
                    shutil.move(str(lblf), train_lbl / lblf.name)
            else:
                shutil.move(str(imgf), val_img / imgf.name)
                if lblf.exists():
                    shutil.move(str(lblf), val_lbl / lblf.name)

        # Remove original flat dirs if empty
        try:
            Path(images_dir).rmdir()
            Path(labels_dir).rmdir()
        except OSError:
            pass

        yaml_path = Path(out_dir) / "data.yaml"
        yaml = f"""path: {out_dir}
train: images/train
val: images/val
names:
  0: lane
  1: ball
"""
        yaml_path.write_text(yaml)
    elif already_split:
        print("\nTrain/val split already exists, skipping...")
        yaml_path = Path(out_dir) / "data.yaml"
        if not yaml_path.exists():
            yaml = f"""path: {out_dir}
train: images/train
val: images/val
names:
  0: lane
  1: ball
"""
            yaml_path.write_text(yaml)
            print(f"Created {yaml_path}")
    else:
        # No split requested
        yaml_path = Path(out_dir) / "data.yaml"
        yaml = f"""path: {out_dir}
train: images
val: images
names:
  0: lane
  1: ball
"""
        yaml_path.write_text(yaml)

    print(f"\nDone. Dataset at: {out_dir}")
    if yaml_path.exists():
        print(f"Config: {yaml_path}")
        # Print stats
        train_dir = Path(out_dir) / "images" / "train"
        val_dir = Path(out_dir) / "images" / "val"
        flat_dir = Path(out_dir) / "images"
        
        if train_dir.exists():
            train_count = len(list(train_dir.glob("*.jpg")))
            val_count = len(list(val_dir.glob("*.jpg")))
            print(f"  Train: {train_count} images")
            print(f"  Val: {val_count} images")
        elif flat_dir.exists():
            total_count = len(list(flat_dir.glob("*.jpg")))
            print(f"  Total: {total_count} images (no train/val split)")

if __name__ == "__main__":
    main()
