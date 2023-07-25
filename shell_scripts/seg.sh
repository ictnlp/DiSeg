export CUDA_VISIBLE_DEVICES=0

MUSTC_ROOT=path_to_mustc_data
LANG=de
SAVE_DIR=path_to_save_checkpoints
OUPUT_SEG=path_to_save_segment

WAV=path_to_wav_file

python segment.py ${MUSTC_ROOT}/en-${LANG} \
    --task speech_to_text_multitask  \
    --config-yaml config_raw.yaml \
    --ckpt ${SAVE_DIR}/average-model.pt \
    --save-root ${OUPUT_SEG} \
    --wav ${wav}