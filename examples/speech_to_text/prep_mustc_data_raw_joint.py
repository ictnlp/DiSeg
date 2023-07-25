#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_config_yaml_raw,
    get_zip_manifest,
    save_df_to_tsv,
    cal_gcmvn_stats,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pdb

from fairseq.data.audio.audio_utils import get_waveform


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = [
    "id",
    "audio",
    "n_frames",
    "src_text",
    "tgt_text",
    "speaker",
    "src_lang",
    "tgt_lang",
]


class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        # pdb.set_trace()
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            """
            if _lang == "en":
                filename = txt_root / f"{split}.{_lang}.clean"
            else:
                filename = txt_root / f"{split}.{_lang}"
            """

            filename = txt_root / f"{split}.{_lang}"
            with open(filename) as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, str, str]:
        return self.data[n]

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute()
    lang = args.tgt_lang
    cur_root = root / f"en-{lang}"
    if not cur_root.is_dir():
        print(f"{cur_root.as_posix()} does not exist. Skipped.")
    # Extract features

    for split in MUSTC.SPLITS:
        is_train_split = split.startswith("train")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = MUSTC(args.data_root, lang, split)
        for wav, offset, n_frames, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(
            dataset
        ):
            manifest["id"].append(utt_id)
            manifest["audio"].append(f"{wav}:{offset}:{n_frames}")
            manifest["n_frames"].append(n_frames)
            manifest["tgt_text"].append(tgt_utt)
            manifest["speaker"].append(speaker_id)
            manifest["tgt_lang"].append(lang)
            manifest["src_text"].append(src_utt)
            manifest["src_lang"].append("en")

            manifest["id"].append(utt_id)
            manifest["audio"].append(f"{wav}:{offset}:{n_frames}")
            manifest["n_frames"].append(n_frames)
            manifest["tgt_text"].append(src_utt)
            manifest["speaker"].append(speaker_id)
            manifest["tgt_lang"].append("en")
            manifest["src_text"].append(src_utt)
            manifest["src_lang"].append("en")
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(
            df, is_train_split=is_train_split, min_n_frames=1000, max_n_frames=480000
        )
        save_df_to_tsv(df, cur_root / f"{split}_joint.tsv")

    for split in ["tst-COMMON", "dev"]:
        is_train_split = split.startswith("train")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = MUSTC(args.data_root, lang, split)
        for wav, offset, n_frames, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(
            dataset
        ):
            manifest["id"].append(utt_id)
            manifest["audio"].append(f"{wav}:{offset}:{n_frames}")
            manifest["n_frames"].append(n_frames)
            manifest["tgt_text"].append(tgt_utt)
            manifest["speaker"].append(speaker_id)
            manifest["tgt_lang"].append(lang)
            manifest["src_text"].append(src_utt)
            manifest["src_lang"].append("en")
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(
            df, is_train_split=is_train_split, min_n_frames=1000, max_n_frames=480000
        )
        save_df_to_tsv(df, cur_root / f"{split}_st.tsv")
    # Generate config YAM
    gen_config_yaml_raw(
        cur_root,
        None,
        yaml_filename=f"config.yaml",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--cmvn-type",
        default="utterance",
        choices=["global", "utterance"],
        help="The type of cepstral mean and variance normalization",
    )
    parser.add_argument(
        "--gcmvn-max-num",
        default=150000,
        type=int,
        help=(
            "Maximum number of sentences to use to estimate" "global mean and variance"
        ),
    )
    parser.add_argument("--tgt-lang", help="target language")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
