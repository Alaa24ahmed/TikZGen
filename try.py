from operator import itemgetter

from detikzify.model import load
from detikzify.infer import DetikzifyPipeline
import torch

image = "https://w.wiki/A7Cc"
pipeline = DetikzifyPipeline(*load(
    base_model="nllg/detikzify-ds-7b",
    device_map="auto",
    torch_dtype=torch.bfloat16,
))

# generate a single TikZ program
fig = pipeline.sample(image=image)

# if it compiles, rasterize it and show it
if fig.is_rasterizable:
    fig.rasterize().show()

# run MCTS for 10 minutes and generate multiple TikZ programs
figs = set()
for score, fig in pipeline.simulate(image=image, timeout=600):
    figs.add((score, fig))

# save the best TikZ program
best = sorted(figs, key=itemgetter(0))[-1][1]
best.save("fig.tex")