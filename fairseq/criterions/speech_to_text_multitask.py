# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import pdb


@dataclass
class SpeechToTextMultitaskCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    mt_training: bool = field(
        default=False,
        metadata={"help": "add mt multi-task"},
    )
    asr_training: bool = field(
        default=False,
        metadata={"help": "add asr multi-task"},
    )


@register_criterion(
    "speech_to_text_multitask", dataclass=SpeechToTextMultitaskCriterionConfig
)
class SpeechToTextMultitaskCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        mt_training=False,
        asr_training=False,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.mt_training = mt_training
        self.asr_training = asr_training

    def forward_st(self, model, sample, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        st_output, st_encoder_out = model(**audio_input)
        loss, _ = self.compute_loss(model, st_output, sample, reduce=reduce)
        return loss, st_encoder_out

    def forward_mt(self, model, sample, reduce):
        text_input = {
            "src_tokens": sample["net_input"]["transcription"],
            "src_lengths": sample["net_input"]["transcription_lengths"],
            "mode": "mt",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        mt_output, _ = model(**text_input)
        loss, _ = self.compute_loss(model, mt_output, sample, reduce=reduce)
        return loss

    def forward_asr(self, model, sample, reduce, speech_encoder_out=None):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_transcription_tokens"],
        }
        asr_output, _ = model(**audio_input, speech_encoder_out=speech_encoder_out)
        loss, _ = self.compute_loss(
            model, asr_output, {"target": sample["transcription"]}, reduce=reduce
        )
        return loss

    def forward_ext_mt(self, model, sample, reduce):
        text_output, _ = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        st_loss, mt_loss, asr_loss, ext_mt_loss = (
            torch.Tensor([0]).cuda(),
            torch.Tensor([0]).cuda(),
            torch.Tensor([0]).cuda(),
            torch.Tensor([0]).cuda(),
        )
        st_size, mt_size, asr_size, ext_mt_size = 0, 0, 0, 0

        # pdb.set_trace()

        mode = sample["net_input"]["mode"]
        if mode == "st":
            st_loss, speech_encoder_out = self.forward_st(model, sample, reduce)
            st_size = sample_size = sample["ntokens"]

            if self.mt_training and self.training:
                mt_loss = self.forward_mt(model, sample, reduce)
                mt_size = sample["ntokens"]

            if self.asr_training and self.training:
                asr_loss = self.forward_asr(
                    model, sample, reduce, speech_encoder_out=speech_encoder_out
                )
                asr_size = sample["src_ntokens"]

            loss = st_loss + mt_loss + asr_loss

            """
            if self.mt_training and self.training:
                st_loss = self.forward_st(model, sample, reduce)
                mt_loss = self.forward_mt(model, sample, reduce)
                loss = st_loss + mt_loss
                st_size = mt_size = sample_size = sample["ntokens"]
            else:
                loss = st_loss = self.forward_st(model, sample, reduce)
                st_size = sample_size = sample["ntokens"]
            """
        elif mode == "ext_mt":
            loss = ext_mt_loss = self.forward_ext_mt(model, sample, reduce)
            ext_mt_size = sample_size = sample["ntokens"]

        logging_output = {
            "loss": loss.data,
            "st_loss": st_loss.data,
            "st_sample_size": st_size,
            "mt_loss": mt_loss.data,
            "mt_sample_size": mt_size,
            "asr_loss": asr_loss.data,
            "asr_sample_size": asr_size,
            "ext_mt_loss": ext_mt_loss.data,
            "ext_mt_sample_size": ext_mt_size,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        asr_loss_sum = sum(log.get("asr_loss", 0) for log in logging_outputs)
        ext_mt_loss_sum = sum(log.get("ext_mt_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        mt_sample_size = sum(log.get("mt_sample_size", 0) for log in logging_outputs)
        asr_sample_size = sum(log.get("asr_sample_size", 0) for log in logging_outputs)
        ext_mt_sample_size = sum(
            log.get("ext_mt_sample_size", 0) for log in logging_outputs
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "st_loss",
            st_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0,
            st_sample_size,
            round=3,
        )
        metrics.log_scalar(
            "mt_loss",
            mt_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0,
            mt_sample_size,
            round=3,
        )
        metrics.log_scalar(
            "asr_loss",
            asr_loss_sum / asr_sample_size / math.log(2) if asr_sample_size != 0 else 0,
            asr_sample_size,
            round=3,
        )
        metrics.log_scalar(
            "ext_mt_loss",
            ext_mt_loss_sum / ext_mt_sample_size / math.log(2)
            if ext_mt_sample_size != 0
            else 0,
            ext_mt_sample_size,
            round=3,
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
