"""Model A input check: whole image vs Model C crop.

Same detector file and accept rule as DogDetector.java, same padded crop as
VisionPipeline.cropPadded, same Model A file and 224x224 RGB 0-255 input as
PostureClassifier.java. Runs on Model A's held-out test split (GitHub A
model-a/test_predictions.json, 235 photos) and on the six project clips.

  python -m venv venv && venv/bin/pip install mediapipe ai-edge-litert pillow numpy
  venv/bin/python eval/model_a_crop_vs_full.py \
      android-app/app/src/main/assets/efficientdet_lite0.tflite \
      android-app/app/src/main/assets/model_a_posture.tflite \
      ../cs663-project-2-vision
"""
import sys, json, glob, os, collections
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
from ai_edge_litert.interpreter import Interpreter
detp, ap, vision_root = sys.argv[1], sys.argv[2], sys.argv[3]
LABELS = ["sitting", "standing", "lying_down", "walking"]
det = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
    base_options=mpt.BaseOptions(model_asset_path=detp), max_results=5, score_threshold=0.25,
    category_allowlist=["dog","cat","horse","sheep","cow","bear"]))
it = Interpreter(model_path=ap); it.allocate_tensors()
inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
def probs(im):
    x = np.asarray(im.convert("RGB").resize((224,224), Image.BILINEAR), dtype=np.float32)[None]
    it.set_tensor(inp, x); it.invoke(); return it.get_tensor(out)[0].copy()
def lab(p): return LABELS[int(p.argmax())], float(p.max())
def cls(im): return lab(probs(im))
def box(path):
    r = det.detect(mp.Image.create_from_file(path))
    best=None
    for d in r.detections:
        c=d.categories[0]
        if c.score >= (.25 if c.category_name=="dog" else .30) and (best is None or c.score>best[0]):
            b=d.bounding_box; best=(c.score,(b.origin_x,b.origin_y,b.width,b.height))
    return best
def crop(im, bb, pad):
    x,y,w,h = bb; cx,cy = x+w/2, y+h/2
    side=max(w,h)*(1+2*pad); cw=max(w*(1+2*pad), side*.75); ch=max(h*(1+2*pad), side*.75)
    l=max(0,round(cx-cw/2)); t=max(0,round(cy-ch/2)); r=min(im.width,round(cx+cw/2)); b=min(im.height,round(cy+ch/2))
    return im.crop((l,t,max(r,l+1),max(b,t+1)))
PADS=[0.15,0.35,0.6]
def evaluate(items):
    ok=collections.Counter(); conf_ok=collections.Counter(); confident=collections.Counter(); n=0; nd=0
    per=collections.defaultdict(lambda: collections.Counter())
    for path,true in items:
        im=Image.open(path); n+=1
        bb=box(path)
        pf=probs(im); preds={"full":lab(pf)}
        if bb:
            nd+=1
            for p in PADS: preds[f"crop{p}"]=cls(crop(im,bb[1],p))
            pc=probs(crop(im,bb[1],0.15)); preds["avg"]=lab((pf+pc)/2)
        else:
            for p in PADS: preds[f"crop{p}"]=None
            preds["avg"]=None
        for k,v in preds.items():
            v2 = v if v is not None else preds["full"]   # app falls back when no dog: here use full for fairness
            ok[k]+= v2[0]==true; per[true][k]+= v2[0]==true
            if v2[1]>=0.65: confident[k]+=1; conf_ok[k]+= v2[0]==true
        per[true]["n"]+=1
    return n,nd,ok,confident,conf_ok,per
def report(name, items):
    n,nd,ok,confident,conf_ok,per=evaluate(items)
    print(f"\n### {name}: {n} images, Model C found the dog in {nd} ({nd/n:.0%}); no-dog images use the full-frame answer in every column")
    keys=["full"]+[f"crop{p}" for p in PADS]+["avg"]
    print("| Model A input | Accuracy | Confident (>=0.65) | Accuracy when confident | " + " | ".join(sorted(per)) + " |")
    print("|---|---|---|---|" + "---|"*len(per))
    for k in keys:
        name_k = "whole image" if k=="full" else ("mean of whole image + crop 0.15" if k=="avg" else f"Model C crop, pad {k[4:]}")
        print(f"| {name_k} | {ok[k]/n:.1%} | {confident[k]/n:.0%} | {conf_ok[k]/max(1,confident[k]):.1%} | " + " | ".join(f"{per[c][k]}/{per[c]['n']}" for c in sorted(per)) + " |")
root=os.path.join(vision_root,"gemini-finetune")
test=json.load(open(os.path.join(vision_root,"model-a/test_predictions.json")))
items=[(os.path.join(root,r["dest"]), r["true"]) for r in test]
print("sanity (stored test acc):", sum(r["true"]==r["pred"] for r in test)/len(test))
report("Held-out test photos (Model A test split)", items)
CLIP={"seq01":"standing","seq02":"lying_down","seq03":"walking","seq04":"lying_down","seq05":"lying_down","seq06":"standing"}
vitems=[]
for d in sorted(glob.glob(os.path.join(vision_root,"video-sequence-layer/video-sequences/seq0*_frames"))):
    clip_lab=CLIP[os.path.basename(d)[:5]]
    vitems+=[(f,clip_lab) for f in sorted(glob.glob(d+"/*.jpg")+glob.glob(d+"/*.png"))]
report("Video clip frames (clip label applied to every frame)", vitems)
