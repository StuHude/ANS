import argparse, os, json, pickle
from collections import defaultdict

def load_refs(pkl_path):
    refs = pickle.load(open(pkl_path, "rb"), encoding="latin1")
    by_ref_id = {}
    by_ann_id = defaultdict(list)
    for r in refs:
        rid = int(r["ref_id"])
        aid = int(r["ann_id"])
        sents = r.get("sentences", [])
        caps = []
        for s in sents:
            raw = s.get("raw") or s.get("sent") or ""
            raw = " ".join(raw.strip().split())
            if raw:
                caps.append(raw)
        if caps:
            by_ref_id[rid] = caps
            by_ann_id[aid].append(caps)   # ann_id 可能对应多个 ref（极少数），所以用 list
    return by_ref_id, by_ann_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_mask_json", required=True)
    ap.add_argument("--refs_pkl",       required=True)  # refs(umd).p（RefCOCOg 用 umd）
    ap.add_argument("--out_json",       required=True)
    args = ap.parse_args()

    wm = json.load(open(args.with_mask_json, "r"))
    images = wm["images"]
    anns   = wm["annotations"]

    by_ref_id, by_ann_id = load_refs(args.refs_pkl)

    imgid2caps = defaultdict(list)
    miss_ref = miss_ann = 0

    for a in anns:
        img_id = int(a["image_id"])
        ref_id = a.get("original_id", None)
        got = False

        # 1) original_id → ref_id（有的 with_mask 确实存的是 ref_id；你的这份并不是）
        if ref_id is not None:
            ref_id = int(ref_id)
            caps = by_ref_id.get(ref_id)
            if caps:
                imgid2caps[img_id].extend(caps)
                got = True
            else:
                miss_ref += 1

        # 2) original_id → ann_id（你的数据：original_id 就是 ann_id，命中 4896/4896）
        if not got:
            ann_id = int(a.get("original_id", -1))  # ★ 关键修正：用 original_id 当 ann_id
            cap_lists = by_ann_id.get(ann_id)
            if cap_lists:
                for cs in cap_lists:
                    imgid2caps[img_id].extend(cs)
                got = True
            else:
                miss_ann += 1

    # 去重并写出 COCO-Caption GT
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    out = {
        "type": "captions",
        "images": [{"id": int(im["id"])} for im in images],
        "annotations": []
    }
    aid = 1
    for img_id, caps in imgid2caps.items():
        seen = set()
        for c in caps:
            if not c:
                continue
            if c in seen:
                continue
            seen.add(c)
            out["annotations"].append({"id": aid, "image_id": int(img_id), "caption": c})
            aid += 1

    json.dump(out, open(args.out_json, "w"))
    print(f"[OK] Wrote {len(out['annotations'])} captions over {len(out['images'])} images.")
    print(f"[STATS] misses: ref_id={miss_ref}, ann_id={miss_ann}  "
          f"(按你的数据结构，ref_id 会大量 miss，但 ann_id 应该几乎 0)")

if __name__ == "__main__":
    main()
