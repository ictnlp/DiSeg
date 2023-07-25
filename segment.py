import argparse
from email.policy import default
import torch
import os
import pickle
import json
import soundfile as sf
import tqdm
import time

# from models import audio_encoder
import tqdm
import numpy as np
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
import pdb
from fairseq import checkpoint_utils, tasks, utils, options
import re

parser = options.get_generation_parser()
parser.add_argument("--ckpt", type=str)
parser.add_argument("--wav", type=str)
parser.add_argument("--save-root", type=str, default="seg_fig")
args = options.parse_args_and_arch(parser)
print(args)

samples, sr = sf.read(args.wav, dtype="float32")
time = np.arange(0, len(samples))
os.makedirs(args.save_root, exist_ok=True)

out_dir = os.path.join(args.save_root, args.wav.strip(".wav").split("/")[-1])
print("data save at: ", out_dir)

########################## setup model ##########################
task = tasks.setup_task(args)
tgt_dict = task.target_dictionary
models, cfg = checkpoint_utils.load_model_ensemble([args.ckpt], task=task, strict=False)
model = models[0]
model.cuda()
########################## setup model ##########################


def get_seg(samples):

    net_input = {}
    net_input["src_tokens"] = torch.Tensor(samples).unsqueeze(0)
    net_input["src_lengths"] = torch.Tensor([len(samples)]).long()
    net_input = utils.move_to_cuda(net_input)
    seg_speech = getattr(cfg.criterion, "seg_speech", False)
    if seg_speech:
        encoder_outs = model.encoder.forward(
            net_input["src_tokens"],
            net_input["src_lengths"],
            mode="st",
            seg_speech=True,
        )
    else:
        encoder_outs = model.encoder.forward(
            net_input["src_tokens"], net_input["src_lengths"], mode="st"
        )
    return encoder_outs["seg_prob"]


seg_prob = get_seg(samples)
seg_speech = getattr(cfg.criterion, "seg_speech", False)

new_seg_prob = []

if seg_speech:
    new_seg_prob.append(seg_prob[0][0])
    for i in range(1, seg_prob.size(1)):
        if seg_prob[0][i] >= 0.5 and seg_prob[0][i - 1] >= 0.5:
            new_seg_prob.append(0)
        else:
            new_seg_prob.append(seg_prob[0][i].item())

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.pyplot import MultipleLocator

fig = plt.figure(figsize=(15, 2))
plt.rcParams["savefig.dpi"] = 800
plt.rcParams["figure.dpi"] = 800


plt.plot(time, samples, lw=0.1, color="blue")

i = 400
seg_i = 0
while i < len(samples):
    if new_seg_prob[seg_i] >= 0.5:
        plt.axvline(x=i, ls="-", lw=new_seg_prob[seg_i], c="red")
    i += 320
    seg_i += 1

plt.ylim((-1.3 * np.max(np.abs(samples)), 1.3 * np.max(np.abs(samples))))
plt.xlim((-5, len(samples) + 5))
plt.yticks([])
plt.xticks(time, time // 16)

x_major_locator = MultipleLocator(8000)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.tick_top()


plt.savefig(out_dir + ".jpg", bbox_inches="tight")
plt.close()
