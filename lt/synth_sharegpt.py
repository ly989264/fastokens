"""A ShareGPT-shaped sample ([[{role, content}, ...], ...]) built from this
repository's own markdown, for prefix_check.py where the real dataset is not
at hand. usage: python lt/synth_sharegpt.py REPO OUT.json"""
import glob, json, os, random, sys

repo, out = sys.argv[1], sys.argv[2]
paras = []
for f in sorted(glob.glob(os.path.join(repo, "**", "*.md"), recursive=True)):
    paras += [p.strip() for p in open(f, encoding="utf-8").read().split("\n\n") if len(p.strip()) > 40]
rng = random.Random(0)
convs = []
for _ in range(400):
    turns = rng.randint(2, 8)
    convs.append([{"role": "user" if t % 2 == 0 else "assistant", "content": rng.choice(paras)} for t in range(turns)])
json.dump(convs, open(out, "w", encoding="utf-8"))
print(f"{len(convs)} conversations from {len(paras)} paragraphs")
