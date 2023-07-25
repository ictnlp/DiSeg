MUSTC_ROOT=path_to_mustc_data
LANG=de
tar -xzvf ${MUSTC_ROOT}/MUSTC_v1.0_en-${LANG}.tar.gz

# 1. prepare raw mustc data
python3 examples/speech_to_text/prep_mustc_data_raw.py \
    --data-root ${MUSTC_ROOT} --tgt-LANG ${LANG}

# 2. prepare vocabulary
python3 examples/speech_to_text/prep_vocab.py \
    --data-root ${MUSTC_ROOT} \
    --vocab-type unigram --vocab-size 10000 --joint \
    --tgt-LANG ${LANG}

# 3. prepare mustc mt data
MUSTC_TEXT_ROOT=${MUSTC_ROOT}/en-${LANG}-text
SPM_MODEL==${MUSTC_ROOT}/en-${LANG}/spm_unigram10000_raw.model
mkdir ${MUSTC_TEXT_ROOT}

for SPLIT in train dev tst-COMMON
do
    for L in en ${LANG}
    do
        python3 examples/speech_to_text/apply_spm.py \
            --input-file ${MUSTC_ROOT}/en-${LANG}/data/${SPLIT}/txt/${SPLIT}.${L} \
            --output-file ${MUSTC_TEXT_ROOT}/${SPLIT}.spm.${L}  \
            --model ${SPM_MODEL}
    done
done

fairseq-preprocess --source-lang en --target-lang ${LANG} \
    --trainpref ${MUSTC_TEXT_ROOT}/train.spm --validpref ${MUSTC_TEXT_ROOT}/dev.spm \
    --testpref ${MUSTC_TEXT_ROOT}/tst-COMMON.spm \
    --destdir ${MUSTC_ROOT}/data-bin/mustc_en_${LANG}_text \
    --tgtdict ${MUSTC_ROOT}/en-${LANG}/spm_unigram10000_raw.txt \
    --srcdict ${MUSTC_ROOT}/en-${LANG}/spm_unigram10000_raw.txt \
    --nwordssrc 10000 --nwordstgt 10000 \
    --workers 60

# 4. generate the wav list and reference file for SimulEval
EVAL_ROOT=path_to_save_simuleval_data # such as ${MUSTC_ROOT}/en-de-simuleval
for SPLIT in dev tst-COMMON
do
    python examples/speech_to_text/seg_mustc_data.py \
    --data-root ${MUSTC_ROOT} --LANG ${LANG} \
    --split ${SPLIT} --task st \
    --output ${EVAL_ROOT}/${SPLIT}
done
