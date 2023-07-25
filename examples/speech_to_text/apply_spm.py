import argparse
from argparse import Namespace
from fairseq.data import encoders
import tqdm
import re
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str, required=True)
parser.add_argument("--output-file", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--clean", action="store_true")

args = parser.parse_args()

bpe_tokenizer = encoders.build_bpe(
    Namespace(
        bpe="sentencepiece",
        sentencepiece_model=args.model,
    )
)

with open(args.input_file, "r") as input_file:
    input_lines = input_file.readlines()

with open(args.output_file, "w") as output_file:
    for line in tqdm.tqdm(input_lines):
        if args.clean:
            line = re.sub("[^a-z'() ]", "", line.strip().lower())
            line = re.sub(" +", " ", line.strip().lower())
        encoded_line = bpe_tokenizer.encode(line)
        output_file.write(encoded_line + "\n")
