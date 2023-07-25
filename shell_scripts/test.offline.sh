export CUDA_VISIBLE_DEVICES=0

MUSTC_ROOT=path_to_mustc_data
LANG=de

SAVE_DIR=path_to_save_checkpoints

python scripts/average_checkpoints.py \
    --inputs ${SAVE_DIR} \
    --num-update-checkpoints 5 \
    --output ${SAVE_DIR}/average-model.pt \
    --best True


python fairseq_cli/generate.py ${MUSTC_ROOT}/en-${LANG} --tgt-lang ${LANG} \
    --config-yaml config_raw.yaml \
    --gen-subset tst-COMMON_raw \
    --task speech_to_text_multitask \
    --path ${SAVE_DIR}/average-model.pt \
    --max-tokens 1000000 \
    --batch-size 250 \
    --beam 1 \
    --scoring sacrebleu \
    --prefix-size 1 \
    --max-source-positions 1000000 \
    --eval-task st