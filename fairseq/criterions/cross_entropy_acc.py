# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.models.fairseq_encoder import EncoderOut


# @register_criterion("cross_entropy_acc")
# class CrossEntropyWithAccCriterion(FairseqCriterion):
#     def __init__(self, task, sentence_avg):
#         super().__init__(task)
#         self.sentence_avg = sentence_avg
#
#     def compute_loss(self, model, net_output, target, reduction, log_probs):
#         # N, T -> N * T
#         target = target.view(-1)
#         lprobs = model.get_normalized_probs(net_output, log_probs=log_probs)
#         if not hasattr(lprobs, "batch_first"):
#             logging.warning(
#                 "ERROR: we need to know whether "
#                 "batch first for the net output; "
#                 "you need to set batch_first attribute for the return value of "
#                 "model.get_normalized_probs. Now, we assume this is true, but "
#                 "in the future, we will raise exception instead. "
#             )
#         batch_first = getattr(lprobs, "batch_first", True)
#         if not batch_first:
#             lprobs = lprobs.transpose(0, 1)
#
#         # N, T, D -> N * T, D
#         lprobs = lprobs.view(-1, lprobs.size(-1))
#         loss = F.nll_loss(
#             lprobs, target.long(), ignore_index=self.padding_idx, reduction=reduction
#         )
#         return lprobs, loss
#
#     def get_logging_output(self, sample, target, lprobs, loss):
#         target = target.view(-1)
#         mask = target != self.padding_idx
#         correct = torch.sum(
#             lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)
#         )
#         total = torch.sum(mask)
#         sample_size = (
#             sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
#         )
#
#         logging_output = {
#             "loss": utils.item(loss.data),  # * sample['ntokens'],
#             "ntokens": sample["ntokens"],
#             "nsentences": sample["target"].size(0),
#             "sample_size": sample_size,
#             "correct": utils.item(correct.data),
#             "total": utils.item(total.data),
#         }
#
#         return sample_size, logging_output
#
#     def forward(self, model, sample, reduction="sum", log_probs=True):
#         """Computes the cross entropy with accuracy metric for the given sample.
#
#         This is similar to CrossEntropyCriterion in fairseq, but also
#         computes accuracy metrics as part of logging
#
#         Args:
#             logprobs (Torch.tensor) of shape N, T, D i.e.
#                 batchsize, timesteps, dimensions
#             targets (Torch.tensor) of shape N, T  i.e batchsize, timesteps
#
#         Returns:
#         tuple: With three elements:
#             1) the loss
#             2) the sample size, which is used as the denominator for the gradient
#             3) logging outputs to display while training
#
#         TODO:
#             * Currently this Criterion will only work with LSTMEncoderModels or
#             FairseqModels which have decoder, or Models which return TorchTensor
#             as net_output.
#             We need to make a change to support all FairseqEncoder models.
#         """
#         # decoder_out = model(**sample["net_input"])
#         encoder_output = model.get_encoder_output(sample["net_input"])
#         decoder_out = model.decoder(
#             prev_output_tokens=sample["net_input"]["prev_output_tokens"],
#             encoder_out=encoder_output
#         )
#         target = sample["target"]
#         lprobs, loss = self.compute_loss(
#             model, decoder_out, target, reduction, log_probs
#         )
#         sample_size, logging_output = self.get_logging_output(
#             sample, target, lprobs, loss
#         )
#
#         # if not model.training and loss/sample_size < 2.0:
#         #     import editdistance
#         #
#         #     c_err = 0
#         #     c_len = 0
#         #     with torch.no_grad():
#         #         decodeds = self.w2l_decoder.decode(encoder_output)
#         #         import pdb; pdb.set_trace()
#         #         for decoded, t, inp_l in zip(decodeds, sample["target"], input_lengths):
#         #             decoded = decoded[0]
#         #             if len(decoded) < 1:
#         #                 decoded = None
#         #             else:
#         #                 decoded = decoded[0]
#         #
#         #             p = (t != self.task.target_dictionary.pad()) & (
#         #                 t != self.task.target_dictionary.eos()
#         #             )
#         #             targ = t[p]
#         #             targ_units_arr = targ.tolist()
#         #             pred_units_arr = decoded.tolist()
#         #
#         #             c_err += editdistance.eval(pred_units_arr, targ_units_arr)
#         #             c_len += len(targ_units_arr)
#         #
#         #         logging_output["c_errors"] = c_err
#         #         logging_output["c_total"] = c_len
#
#         return loss, sample_size, logging_output
#
#     @staticmethod
#     def aggregate_logging_outputs(logging_outputs):
#         """Aggregate logging outputs from data parallel training."""
#         correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
#         total_sum = sum(log.get("total", 0) for log in logging_outputs)
#         loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
#         ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
#         nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
#         sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
#         c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
#         c_total = sum(log.get("c_total", 0) for log in logging_outputs)
#         agg_output = {
#             "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
#             # if args.sentence_avg, then sample_size is nsentences, then loss
#             # is per-sentence loss; else sample_size is ntokens, the loss
#             # becomes per-output token loss
#             "ntokens": ntokens,
#             "nsentences": nsentences,
#             "sample_size": sample_size,
#             "acc": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
#             "correct": correct_sum,
#             "total": total_sum,
#             # total is the number of validate tokens
#         }
#         if sample_size != ntokens:
#             agg_output["nll_loss"] = loss_sum / ntokens
#
#         if c_total > 0:
#             agg_output["uer"] = c_errors / c_total
#         # loss: per output token loss
#         # nll_loss: per sentence loss
#         return agg_output


@register_criterion("ce_acc")
class CrossEntropyWithAccV2Criterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(task)
        self.decoder = self.build_decoder(args, task)

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)

    def build_decoder(self, args, task):
        decoder = getattr(args, "decoder", None)

        if decoder == "seq2seq_decoder":
            from examples.speech_recognition.seq2seq_decoder import Seq2seqDecoder

            decoder = Seq2seqDecoder(args, task.target_dictionary, {})
        elif decoder == "seq2seq_lm_decoder":
            from examples.speech_recognition.seq2seq_decoder import Seq2seqDecoder

            decoder = Seq2seqDecoder(args, task.target_dictionary, ({}, {}))

        return decoder

    def compute_loss(self, model, net_output, target, reduction, log_probs):
        # N, T -> N * T
        target = target.view(-1)
        lprobs = model.get_normalized_probs(net_output, log_probs=log_probs)
        if not hasattr(lprobs, "batch_first"):
            logging.warning(
                "ERROR: we need to know whether "
                "batch first for the net output; "
                "you need to set batch_first attribute for the return value of "
                "model.get_normalized_probs. Now, we assume this is true, but "
                "in the future, we will raise exception instead. "
            )
        batch_first = getattr(lprobs, "batch_first", True)
        if not batch_first:
            lprobs = lprobs.transpose(0, 1)

        # N, T, D -> N * T, D
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = F.nll_loss(
            lprobs, target.long(), ignore_index=self.padding_idx, reduction=reduction
        )
        return lprobs, loss

    def get_logging_output(self, sample, target, lprobs, loss):
        target = target.view(-1)
        mask = target != self.padding_idx
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)
        )
        total = torch.sum(mask)
        sample_size = sample["ntokens"]

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
        }

        return sample_size, logging_output

    def forward(self, model, sample, reduction="sum", log_probs=True):
        encoder_output = model.get_encoder_output(sample["net_input"])

        decoder_out = model.decoder(
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            encoder_out=encoder_output,
        )

        target = sample["target"]
        lprobs, loss = self.compute_loss(
            model, decoder_out, target, reduction, log_probs
        )
        sample_size, logging_output = self.get_logging_output(
            sample, target, lprobs, loss
        )

        if not model.training:
            import editdistance

            c_err = 0
            c_len = 0
            self.decoder.step_forward_fn = model.decoder
            input_lengths = (~encoder_output.encoder_padding_mask).sum(-1)
            with torch.no_grad():
                decodeds = self.decoder.decode(encoder_output, 50)
                for decoded, t, inp_l in zip(decodeds, sample["target"], input_lengths):
                    decoded = decoded[0]["tokens"]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units_arr = targ.tolist()
                    pred_units_arr = decoded.tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        schedule_sampling = sum(
            log.get("schedule_sampling", 0.0) for log in logging_outputs
        )
        agg_output = {
            "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
            # if args.sentence_avg, then sample_size is nsentences, then loss
            # is per-sentence loss; else sample_size is ntokens, the loss
            # becomes per-output token loss
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "acc": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
            "correct": correct_sum,
            "total": total_sum,
            "ss": schedule_sampling / nsentences
            # total is the number of validate tokens
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        c_total = sum(log.get("c_total", 1) for log in logging_outputs)
        if c_total > 1:
            agg_output["uer"] = c_errors * 100.0 / c_total
        # loss: per output token loss
        # nll_loss: per sentence loss
        return agg_output


@register_criterion("ls_ce_acc")
class LabelSmoothedCrossEntropyWithAccCriterion(CrossEntropyWithAccV2Criterion):
    def compute_loss(self, model, net_output, target, reduction, log_probs):
        # N, T -> N * T
        target = target.view(-1)
        lprobs = model.get_normalized_probs(net_output, log_probs=log_probs)
        if not hasattr(lprobs, "batch_first"):
            logging.warning(
                "ERROR: we need to know whether "
                "batch first for the net output; "
                "you need to set batch_first attribute for the return value of "
                "model.get_normalized_probs. Now, we assume this is true, but "
                "in the future, we will raise exception instead. "
            )
        batch_first = getattr(lprobs, "batch_first", True)
        if not batch_first:
            lprobs = lprobs.transpose(0, 1)

        # N, T, D -> N * T, D
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss, _ = label_smoothed_nll_loss(
            lprobs,
            target.long(),
            0.1,
            ignore_index=self.padding_idx,
            reduce=reduction,
        )

        return lprobs, loss


@register_criterion("ctc_ls_ce_acc")
class CTCLabelSmoothedCrossEntropyWithAccCriterion(
    LabelSmoothedCrossEntropyWithAccCriterion
):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.blk_idx = task.target_dictionary.index("<ctc_blank>")
        self.pad_idx = task.target_dictionary.pad()
        self.bos_idx = task.target_dictionary.eos()
        self.eos_idx = task.target_dictionary.eos()

    def forward(self, model, sample, reduction="sum", log_probs=True):
        encoder_output = model.encoder(tbc=False, **sample["net_input"])
        ctc_logits = encoder_output["encoder_out"]
        len_ctc_logits = (~encoder_output["encoder_padding_mask"]).long().sum(-1)
        encoder_output = EncoderOut(
            encoder_out=encoder_output["encoded"].transpose(0, 1),  # T x B x C
            encoder_embedding=None,
            encoder_padding_mask=encoder_output["encoder_padding_mask"],  # B x T
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

        p = max((model.num_updates - model.teacher_forcing_updates) / 2000.0, 0.0)
        if model.num_updates <= model.teacher_forcing_updates:
            decoder_out = model.decoder(
                prev_output_tokens=sample["net_input"]["prev_output_tokens"],
                encoder_out=encoder_output,
            )
        else:
            with torch.no_grad():
                decoder_out = model.decoder(
                    prev_output_tokens=sample["net_input"]["prev_output_tokens"],
                    encoder_out=encoder_output,
                )
                decoded = decoder_out["logits"].argmax(-1).int()
                device = decoded.device
                prev_self_deocded = torch.cat(
                    [
                        torch.ones([decoded.size(0), 1]).int().to(device)
                        * self.task.target_dictionary.eos(),
                        decoded[:, :-1],
                    ],
                    1,
                )
                prev_output = torch.where(
                    (torch.rand(decoded.size()) > p).to(device),
                    sample["net_input"]["prev_output_tokens"],
                    prev_self_deocded,
                )
            decoder_out = model.decoder(
                prev_output_tokens=prev_output, encoder_out=encoder_output
            )

        target = sample["target"]
        target_lengths = sample["target_lengths"]
        lprobs, ctc_loss, ce_loss = self.compute_loss(
            model,
            ctc_logits,
            len_ctc_logits,
            decoder_out["logits"],
            target,
            target_lengths,
            reduction,
            log_probs,
        )
        sample_size, logging_output = self.get_logging_output(
            sample, target, lprobs, ctc_loss, ce_loss
        )
        loss = ctc_loss + ce_loss
        logging_output["schedule_sampling"] = p

        if not model.training:
            import editdistance

            c_err = 0
            c_len = 0
            self.decoder.step_forward_fn = model.decoder
            input_lengths = (~encoder_output.encoder_padding_mask).sum(-1)
            with torch.no_grad():
                decodeds = self.decoder.decode(encoder_output, 50)
                for decoded, t, inp_l in zip(decodeds, sample["target"], input_lengths):
                    decoded = decoded[0]["tokens"]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units_arr = targ.tolist()
                    pred_units_arr = decoded.tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    def compute_loss(
        self,
        model,
        ctc_logits,
        len_ctc_logits,
        logits,
        target,
        target_lengths,
        reduction,
        log_probs,
    ):
        # N, T -> N * T
        ctc_lprob, lprobs = model.get_normalized_probs(
            ctc_logits, logits, log_probs=log_probs
        )
        ctc_loss = self.cal_ctc_loss(
            ctc_lprob, len_ctc_logits, target, target_lengths - 1
        )

        if not hasattr(lprobs, "batch_first"):
            logging.warning(
                "ERROR: we need to know whether "
                "batch first for the net output; "
                "you need to set batch_first attribute for the return value of "
                "model.get_normalized_probs. Now, we assume this is true, but "
                "in the future, we will raise exception instead. "
            )
        batch_first = getattr(lprobs, "batch_first", True)
        if not batch_first:
            lprobs = lprobs.transpose(0, 1)

        # N, T, D -> N * T, D
        target = target.view(-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss, _ = label_smoothed_nll_loss(
            lprobs,
            target.long(),
            0.1,
            ignore_index=self.padding_idx,
            reduce=reduction,
        )

        return lprobs, ctc_loss, loss

    def cal_ctc_loss(self, lprobs, len_lprobs, target, target_lengths):
        """
        target: without sos eos
        """
        if getattr(lprobs, "batch_first", True):
            lprobs = lprobs.transpose(0, 1)  # T x B x V
        pad_mask = (
            (target != self.pad_idx)
            & (target != self.bos_idx)
            & (target != self.eos_idx)
        )
        targets_flat = target.masked_select(pad_mask)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                len_lprobs,
                target_lengths,
                blank=self.blk_idx,
                reduction="sum",
                zero_infinity=True,
            )

        return loss

    def get_logging_output(self, sample, target, lprobs, ctc_loss, ce_loss):
        target = target.view(-1)
        mask = target != self.padding_idx
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)
        )
        total = torch.sum(mask)
        sample_size = sample["ntokens"]

        logging_output = {
            "loss": utils.item((ctc_loss + ce_loss).data),  # * sample['ntokens'],
            "ctc_loss": utils.item(ctc_loss.data),
            "ce_loss": utils.item(ce_loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
        }

        return sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        schedule_sampling = sum(
            log.get("schedule_sampling", 0.0) for log in logging_outputs
        )
        agg_output = {
            "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
            "ctc_loss": ctc_loss_sum / sample_size if sample_size > 0 else 0.0,
            "ce_loss": ce_loss_sum / sample_size if sample_size > 0 else 0.0,
            # if args.sentence_avg, then sample_size is nsentences, then loss
            # is per-sentence loss; else sample_size is ntokens, the loss
            # becomes per-output token loss
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "acc": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
            "correct": correct_sum,
            "total": total_sum,
            "ss": schedule_sampling / nsentences
            # total is the number of validate tokens
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        c_total = sum(log.get("c_total", 1) for log in logging_outputs)
        if c_total > 1:
            agg_output["uer"] = c_errors * 100.0 / c_total
        # loss: per output token loss
        # nll_loss: per sentence loss
        return agg_output
