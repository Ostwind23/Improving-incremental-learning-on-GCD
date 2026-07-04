"""Quick token boundary check."""
from transformers import BertTokenizer
ALL_CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
tok = BertTokenizer.from_pretrained("bert-base-uncased")
cap = ". ".join(ALL_CLASSES) + "."
enc = tok(cap, return_offsets_mapping=True, add_special_tokens=True)
offsets = enc["offset_mapping"]
ranges = {}
cursor = 0
for ci, cname in enumerate(ALL_CLASSES):
    idx = cap.find(cname, cursor)
    if idx < 0: continue
    c0, c1 = idx, idx + len(cname)
    toks = [ti for ti, (s, e) in enumerate(offsets) if s < c1 and e > c0]
    ranges[ci] = toks
    cursor = c1
old_end = max(ranges[69]) if 69 in ranges else -1
new_start = min(ranges[70]) if 70 in ranges else -1
print("Old class 69 (%s): tokens %s" % (ALL_CLASSES[69], ranges[69]))
print("New class 70 (%s): tokens %s" % (ALL_CLASSES[70], ranges[70]))
print("New class 79 (%s): tokens %s" % (ALL_CLASSES[79], ranges[79]))
print("Old ends at token: %d" % old_end)
print("New starts at token: %d" % new_start)
print("Gap: %d" % (new_start - old_end - 1))
print("NEW_TOKEN_START = %d" % new_start)
print()
# Verify: all new class tokens >= new_start
all_ok = True
for c in range(70, 80):
    for t in ranges[c]:
        if t < new_start:
            print("ERROR: class %d token %d < %d" % (c, t, new_start))
            all_ok = False
for c in range(0, 70):
    for t in ranges[c]:
        if t >= new_start:
            print("ERROR: class %d token %d >= %d" % (c, t, new_start))
            all_ok = False
if all_ok:
    print("ALL CHECKS PASSED: token %d is the exact new-class boundary" % new_start)
