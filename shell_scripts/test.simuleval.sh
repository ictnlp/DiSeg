export CUDA_VISIBLE_DEVICES=0

MUSTC_ROOT=path_to_mustc_data
LANG=de
EVAL_ROOT=path_to_save_simuleval_data
SAVE_DIR=path_to_save_checkpoints
OUTPUT_DIR=path_to_save_simuleval_results

lagging_seg=5 # lagging segment in DiSeg

simuleval --agent diseg_agent.py \
    --source ${EVAL_ROOT}/tst-COMMON/tst-COMMON.wav_list \
    --target ${EVAL_ROOT}/tst-COMMON/tst-COMMON.${LANG} \
    --data-bin ${MUSTC_ROOT}/en-${LANG} \
    --config config_raw.yaml \
    --model-path ${SAVE_DIR}/average-model.pt \
    --output ${OUTPUT_DIR} \
    --lagging-segment ${lagging_seg}  \
    --lang ${LANG} \
    --scores --gpu --fp16 \
    --port 12345