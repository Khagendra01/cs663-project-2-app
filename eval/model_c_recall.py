"""Model C recall check: how often EfficientDet-Lite0 finds the dog.

Runs the same detector file and the same accept rule as the app
(DogDetector.java) on the six video clips in GitHub A. Every frame in
those clips contains a dog, so the number is recall per frame.

  python -m venv venv && venv/bin/pip install mediapipe
  venv/bin/python eval/model_c_recall.py \
      android-app/app/src/main/assets/efficientdet_lite0.tflite \
      ../cs663-project-2-vision/video-sequence-layer/video-sequences
"""
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
model, root = sys.argv[1], sys.argv[2]
def mk(allow): return vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
    base_options=mpt.BaseOptions(model_asset_path=model), max_results=5, score_threshold=0.25, category_allowlist=allow))
old, new = mk(["dog","cat"]), mk(["dog","cat","horse","sheep","cow","bear"])
def ok_old(r): return any((c.category_name=="dog" and c.score>=.25) or (c.category_name=="cat" and c.score>=.45) for d in r.detections for c in d.categories[:1])
def ok_new(r): return any(c.score >= (.25 if c.category_name=="dog" else .30) for d in r.detections for c in d.categories[:1])
T=[0,0,0]
print("| Clip | Frames | Old rule (dog/cat) | New rule (4-legged) |\n|---|---|---|---|")
for d in sorted(glob.glob(os.path.join(root,"seq0*_frames"))):
    fs=sorted(glob.glob(os.path.join(d,"*.jpg"))+glob.glob(os.path.join(d,"*.png")))
    a=b=0
    for f in fs:
        im=mp.Image.create_from_file(f); a+=ok_old(old.detect(im)); b+=ok_new(new.detect(im))
    T[0]+=len(fs);T[1]+=a;T[2]+=b
    print(f"| {os.path.basename(d).replace('_frames','')} | {len(fs)} | {a/len(fs):.0%} | {b/len(fs):.0%} |")
print(f"| **All** | {T[0]} | {T[1]/T[0]:.0%} | {T[2]/T[0]:.0%} |")
