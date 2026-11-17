"""
messidor2_from_hf.py  -  Unpack the Hugging Face copy of Messidor-2 (backup if ADCIS never answers)
=================================================================================================

1. Download the .parquet files from
      https://huggingface.co/datasets/sngsfydy/Messidor2/tree/main/data
   into  F:\\drthesis\\MESSIDOR2\\hf\\
2. (Recommended) download Google's official grades CSV from Kaggle ("messidor2 dr grades",
   Google Brain) into  F:\\drthesis\\MESSIDOR2\\
3. Once:   pip install pyarrow
4. Run:    python code/messidor2_from_hf.py

It writes every image to  F:\\drthesis\\MESSIDOR2\\images\\  and then:
  - if the parquet keeps the original file names AND Google's CSV is there:
      Google's official grades are used, and it reports how often the copy's labels agree;
  - otherwise: it writes  hf_grades.csv  (labels of the copy) and compares the grade counts
      with the published adjudicated counts (1017 / 270 / 347 / 75 / 35).
After that, run  python code/06_detect_grading.py  again: it picks up Messidor-2 by itself.
"""
import io
import os
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("DRTHESIS_ROOT", "F:/drthesis"))
M = ROOT / "MESSIDOR2"
PUBLISHED = {0: 1017, 1: 270, 2: 347, 3: 75, 4: 35}   # adjudicated grades, 1,744 gradable images


def main():
    files = sorted((M / "hf").rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No .parquet files in {M / 'hf'}")
    out = M / "images"
    out.mkdir(parents=True, exist_ok=True)
    rows, k = [], 0
    for f in files:
        df = pd.read_parquet(f)
        for img, label in zip(df["image"], df["label"]):
            name = Path(str(img.get("path") or "")).name if isinstance(img, dict) else ""
            if not name:
                name = f"hf_{k:04d}.png"
            data = img["bytes"] if isinstance(img, dict) else img
            im = Image.open(io.BytesIO(data))
            target = out / (Path(name).stem + ".png")
            if not target.exists():
                im.convert("RGB").save(target)
            rows.append({"image_id": target.name, "hf_label": int(label), "original_name": name})
            k += 1
        print(f"  {f.name}: {len(df)} images", flush=True)
    hf = pd.DataFrame(rows)
    print(f"{len(hf)} images written to {out}")
    print("grade counts in this copy:", hf.hf_label.value_counts().sort_index().to_dict())
    print("published adjudicated counts:", PUBLISHED)

    official = [c for c in M.glob("*.csv") if c.name != "hf_grades.csv"
                and "adjudicated_dr_grade" in pd.read_csv(c, nrows=0).columns]
    named = not hf.original_name.str.startswith("hf_").all()
    if official and named:
        g = pd.read_csv(official[0])
        g["stem"] = [Path(str(i)).stem.lower() for i in g.iloc[:, 0]]
        hf["stem"] = [Path(n).stem.lower() for n in hf.original_name]
        j = hf.merge(g, on="stem")
        agree = (j.hf_label == j.adjudicated_dr_grade).mean() if len(j) else float("nan")
        print(f"matched with Google's CSV by file name: {len(j)} of {len(hf)} | labels agree: {agree:.1%}")
        print("-> 06 will use Google's official grades.")
        (M / "hf_grades.csv").unlink(missing_ok=True)
    else:
        why = "no original file names in the copy" if not named else "Google's CSV not found"
        print(f"Using the labels of the copy ({why}).")
        hf.rename(columns={"hf_label": "adjudicated_dr_grade"})[["image_id", "adjudicated_dr_grade"]] \
            .to_csv(M / "hf_grades.csv", index=False)
        print(f"-> wrote {M / 'hf_grades.csv'}; 06 will use it.")


if __name__ == "__main__":
    main()
