export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MUSTC_ROOT=path_to_mustc_data
LANG=de

SAVE_DIR=path_to_save_checkpoints
W2V_MODEL=path_to_wav2vec_model

mean=0
var=3

# (optional) pre-train a mt encoder/decoder and load the pre-trained model with --load-pretrained-mt-encoder-decoder-from ${PATH_TO_PRETRAINED_MODEL}
python train.py ${MUSTC_ROOT}/en-${LANG}  --tgt-lang ${LANG} --ddp-backend=legacy_ddp \
  --config-yaml config_raw.yaml \
  --train-subset train_raw \
  --valid-subset dev_raw \
  --save-dir ${SAVE_DIR} \
  --max-tokens 1500000  --batch-size 32 --max-tokens-text 4096 \
  --update-freq 1 \
  --num-workers 8 \
  --task speech_to_text_multitask \
  --criterion speech_to_text_multitask_with_seg \
  --report-accuracy \
  --arch convtransformer_espnet_base_wav2vec_seg \
  --w2v2-model-path ${W2V_MODEL} \
  --optimizer adam \
  --lr 0.0001 \
  --lr-scheduler inverse_sqrt \
  --weight-decay 0.0001 \
  --label-smoothing 0.1 \
  --warmup-updates 4000 \
  --clip-norm 10.0 \
  --seed 1 \
  --seg-encoder-layers 6 \
  --noise-mean ${mean} --noise-var ${var} \
  --st-training --mt-training --asr-training \
  --seg-speech --add-speech-seg-text-ctr \
  --eval-task st \
  --eval-bleu \
  --eval-bleu-args '{"beam": 1,"prefix_size":1}' \
  --eval-bleu-print-samples \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --keep-best-checkpoints 20 \
  --save-interval-updates 1000 \
  --keep-interval-updates 30 \
  --max-source-positions 800000 \
  --skip-invalid-size-inputs-valid-test \
  --dropout 0.1 --activation-dropout 0.1 --attention-dropout 0.1 --layernorm-embedding \
  --empty-cache-freq 1000 \
  --ignore-prefix-size 1 \
  --fp16 