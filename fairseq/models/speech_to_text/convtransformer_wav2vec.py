#!/usr/bin/env python3

import logging
import math
from typing import Dict, List, Optional, OrderedDict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.convtransformer import base_architecture
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from fairseq.models.wav2vec import Wav2Vec2Model
import pdb

logger = logging.getLogger(__name__)


@register_model("convtransformer_wav2vec")
class ConvTransformerModelWac2Vec(FairseqEncoderDecoderModel):
    """
    Transformer-based Speech translation model from ESPNet-ST
    https://arxiv.org/abs/2004.10234
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--input-feat-per-channel",
            type=int,
            metavar="N",
            help="encoder input dimension per input channel",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--decoder-output-dim",
            type=int,
            metavar="N",
            help="decoder output dimension (extra linear layer if different from decoder embed dim)",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument(
            "--conv-out-channels",
            type=int,
            metavar="INT",
            help="the number of output channels of conv layer",
        )
        parser.add_argument(
            "--w2v2-model-path",
            default="/path/wav2vec_small.pt",
            type=str,
            help="path to wav2vec model",
        )
        parser.add_argument(
            "--uni-encoder", default=False, type=bool, help="unidirectional encoder"
        )
        # pretrain
        parser.add_argument(
            "--load-pretrained-mt-encoder-decoder-from",
            type=str,
            help="model to take mt encoder/decoder weight from (for initialization)",
        )

    @classmethod
    def build_encoder(cls, args, task, encoder_embed_tokens):
        encoder = ConvTransformerEncoder(args, task, encoder_embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = TransformerDecoderNoExtra(args, task.target_dictionary, embed_tokens)
        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder_embed_tokens = decoder_embed_tokens

        encoder = cls.build_encoder(args, task, encoder_embed_tokens)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)

        # load pretrained mt models
        mt_pretrained_path = getattr(
            args, "load_pretrained_mt_encoder_decoder_from", None
        )
        if mt_pretrained_path is not None and Path(mt_pretrained_path).exists():
            state_dict = checkpoint_utils.load_checkpoint_to_cpu(mt_pretrained_path)[
                "model"
            ]
            mt_encoder_state_dict = OrderedDict()
            mt_decoder_state_dict = OrderedDict()
            for key in state_dict.keys():
                if "hubert" in key or "subsampler" in key:
                    continue
                if key.startswith("encoder"):
                    subkey = key[len("encoder") + 1 :]
                    mt_encoder_state_dict[subkey] = state_dict[key]
                if key.startswith("decoder"):
                    subkey = key[len("decoder") + 1 :]
                    mt_decoder_state_dict[subkey] = state_dict[key]
            encoder.load_state_dict(mt_encoder_state_dict, strict=False)
            decoder.load_state_dict(mt_decoder_state_dict, strict=False)
            logger.info(
                f"load pretrained mt encoder/decoder from: {mt_pretrained_path}"
            )

        return cls(encoder, decoder)

    @staticmethod
    @torch.jit.unused
    def set_batch_first(lprobs):
        lprobs.batch_first = True

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        if self.training:
            self.set_batch_first(lprobs)
        return lprobs

    def output_layout(self):
        return "BTD"

    """
    The forward method inherited from the base class has a **kwargs argument in
    its input, which is not supported in torchscript. This method overrites the forward
    method definition without **kwargs.
    """

    def forward(
        self, src_tokens, src_lengths, mode, prev_output_tokens, speech_encoder_out=None
    ):

        if speech_encoder_out:
            encoder_out = speech_encoder_out
        else:
            encoder_out = self.encoder(
                src_tokens=src_tokens, src_lengths=src_lengths, mode=mode
            )
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        if mode == "st":
            return decoder_out, encoder_out
        else:
            return decoder_out, None


class ConvTransformerEncoder(FairseqEncoder):
    """Conv + Transformer encoder"""

    def __init__(self, args, dictionary=None, embed_tokens=None):
        """Construct an Encoder object."""
        super().__init__(None)

        # initialize wav2vec
        wav2vec_ckpt = torch.load(args.w2v2_model_path)
        self.w2v_args = wav2vec_ckpt["args"]

        self.wav2vec_model = Wav2Vec2Model.build_model(wav2vec_ckpt["args"], task=None)
        self.wav2vec_model.load_state_dict(wav2vec_ckpt["model"])

        self.dim_proj = nn.Linear(
            self.w2v_args.encoder_embed_dim, args.encoder_embed_dim
        )

        # use no conv
        self.dropout = args.dropout
        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = (
            1.0 if args.no_scale_embedding else math.sqrt(args.encoder_embed_dim)
        )
        self.padding_idx = 1
        self.in_channels = 1
        self.input_dim = args.input_feat_per_channel
        max_source_positions = args.max_source_positions
        if max_source_positions < 3200000:
            max_source_positions = 3200000
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions,
            args.encoder_embed_dim,
            self.padding_idx,
            learned=False,
        )

        self.embed_tokens = embed_tokens
        export = getattr(args, "export", False)
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(
                embed_tokens.embedding_dim, export=export
            )
        else:
            self.layernorm_embedding = None

        self.uni_encoder = getattr(args, "uni_encoder", False)

        self.transformer_layers = nn.ModuleList([])
        self.transformer_layers.extend(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self._future_mask = torch.empty(0)

    def _get_w2v_feature(self, src_tokens, src_lengths):
        """
        :param src_tokens: b x frames
        :param src_lengths: b-dim length
        :return: w2v_feature: b x short_frames x feature-dim;
                w2v_lengths: b-dim tensor
                w2v_padding_mask: b x short_frames x feature-dim T/F tensor
        """
        padding_mask = lengths_to_padding_mask(src_lengths)
        w2v_feature, padding_mask = self.wav2vec_model.extract_features(
            src_tokens, padding_mask
        )
        output_length = (1 - padding_mask.int()).sum(dim=1)

        return w2v_feature, padding_mask, output_length

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed

    def forward(self, src_tokens, src_lengths, mode):
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if mode == "st":
            w2v_feature, encoder_padding_mask, input_lengths = self._get_w2v_feature(
                src_tokens, src_lengths
            )
            # pdb.set_trace()
            x = torch.transpose(w2v_feature, 1, 0)
            x = self.dim_proj(x)
            x_emb = self.embed_scale * x
            # bsz, hidden_dim, output_seq_len = x.size()
            # x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)
            # x = self.out(x)
            # x = self.embed_scale * x
            # #

            encoder_padding_mask = lengths_to_padding_mask(input_lengths)
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
            # x = F.dropout(x, p=self.dropout, training=self.training)
            if self.layernorm_embedding is not None:
                x = self.layernorm_embedding(x)
            x = self.dropout_module(x)

        else:
            encoder_padding_mask = src_tokens.eq(self.padding_idx)
            has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()
            x, x_emb = self.forward_embedding(src_tokens)
            if has_pads:
                x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
            x = x.transpose(0, 1)

        for layer in self.transformer_layers:
            x = layer(
                x,
                encoder_padding_mask,
                self.buffered_future_mask(x) if self.uni_encoder else None,
            )

        # if not encoder_padding_mask.any():
        #     maybe_encoder_padding_mask = None
        # else:
        #     maybe_encoder_padding_mask = encoder_padding_mask
        maybe_encoder_padding_mask = encoder_padding_mask

        return {
            "encoder_out": [x],
            "encoder_padding_mask": [maybe_encoder_padding_mask]
            if maybe_encoder_padding_mask is not None
            else [],
            "encoder_embedding": [x_emb],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim], device=tensor.device)),
                1,
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                (encoder_out["encoder_padding_mask"][0]).index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                (encoder_out["encoder_embedding"][0]).index_select(0, new_order)
            ]
        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,
            "encoder_padding_mask": new_encoder_padding_mask,
            "encoder_embedding": new_encoder_embedding,
            "encoder_states": encoder_states,
            "src_tokens": [],
            "src_lengths": [],
        }


class TransformerDecoderNoExtra(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, None


# @register_model_architecture(model_name="convtransformer", arch_name="convtransformer")
# def base_architecture(args):
#     args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
#     args.encoder_layers = getattr(args, "encoder_layers", 6)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
#     args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
#     args.decoder_ffn_embed_dim = getattr(
#         args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
#     )
#     args.decoder_layers = getattr(args, "decoder_layers", 6)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
#     args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
#     args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
#     args.attention_dropout = getattr(args, "attention_dropout", 0.0)
#     args.activation_dropout = getattr(args, "activation_dropout", 0.0)
#     args.activation_fn = getattr(args, "activation_fn", "relu")
#     args.dropout = getattr(args, "dropout", 0.1)
#     args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
#     args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
#     args.share_decoder_input_output_embed = getattr(
#         args, "share_decoder_input_output_embed", False
#     )
#     args.no_token_positional_embeddings = getattr(
#         args, "no_token_positional_embeddings", False
#     )
#     args.adaptive_input = getattr(args, "adaptive_input", False)
#     args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
#
#     args.decoder_output_dim = getattr(
#         args, "decoder_output_dim", args.decoder_embed_dim
#     )
#     args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
#     args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
#     args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
#     args.max_source_positions = getattr(args, "max_source_positions", 3000)
#     args.max_target_positions = getattr(args, "max_target_positions", 1024)
#     args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
#     args.conv_out_channels = getattr(args, "conv_out_channels", args.encoder_embed_dim)


@register_model_architecture(
    "convtransformer_wav2vec", "convtransformer_espnet_wav2vec"
)
def convtransformer_espnet_wav2vec(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)


@register_model_architecture(
    "convtransformer_wav2vec", "convtransformer_espnet_base_wav2vec"
)
def convtransformer_espnet_base_wav2vec(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    # args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    # args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
