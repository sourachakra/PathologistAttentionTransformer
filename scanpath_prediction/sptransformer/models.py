import torch
import torch.nn as nn
from torchvision import models
import math
import os
from os.path import join, dirname
from typing import Optional
from torch import Tensor
import torch.nn.functional as F
from detectron2.layers import ShapeSpec
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .pixel_decoder.fpn import TransformerEncoderPixelDecoder
from .transformer_decoder.transformer import TransformerEncoder, TransformerEncoderLayer
from .transformer_decoder.position_encoding import PositionEmbeddingSine
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from .backbone.swin import D2SwinTransformer
from .config import add_maskformer2_config
import fvcore.nn.weight_init as weight_init
import numpy as np
import cv2
import torch.nn.functional as F
from memory_profiler import profile
import scipy.ndimage as filters

# helper Module that adds positional encoding to the token
# embedding to introduce a notion of word order.
       
        
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, maxlen: int = 100):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) /
                        emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.pos_embedding[:token_embedding.size(0), :]

    def forward_pos(self, pos: torch.Tensor):
        return self.pos_embedding[pos].squeeze(1)


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long())  # * math.sqrt(self.emb_size)

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            #print('x bef:',x.size())
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            #print('x aft:',x.size())
        return x
        
class MLP2(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.onehotlayer = nn.Linear(1,6)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            #print('x bef2:',x.size())
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            #print('x aft2:',x.size())
        x = self.onehotlayer(x)
        #print('onehot:',x.size())
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self,
                    tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt2,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self,
                tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask,
                                    query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask,
                                 query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn_weights

    def forward_pre(self,
                    tgt,
                    memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self,
                tgt,
                memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    def __init__(self,
                 d_model,
                 dim_feedforward=2048,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class ImageFeatureEncoder(nn.Module):
    def __init__(self,
                 cfg_path,
                 dropout,
                 pixel_decoder='MSD',
                 load_segm_decoder=False):
        super(ImageFeatureEncoder, self).__init__()

        # Load Detectrion2 backbone
        cfg = get_cfg()
        add_maskformer2_config(cfg)
        cfg.merge_from_file(cfg_path)
        self.backbone = build_backbone(cfg)
        # if os.path.exists(cfg.MODEL.WEIGHTS):
        bb_weights = torch.load(cfg.MODEL.WEIGHTS,
                                map_location=torch.device('cpu'))
        bb_weights_new = bb_weights.copy()
        for k, v in bb_weights.items():
            if k[:3] == 'res':
                bb_weights_new["stages." + k] = v
                bb_weights_new.pop(k)
        self.backbone.load_state_dict(bb_weights_new)
        self.backbone.eval()
        print('Loaded backbone weights from {}'.format(cfg.MODEL.WEIGHTS))

        # Load deformable pixel decoder
        if cfg.MODEL.BACKBONE.NAME == 'D2SwinTransformer':
            input_shape = {
                "res2": ShapeSpec(channels=128, stride=4),
                "res3": ShapeSpec(channels=256, stride=8),
                "res4": ShapeSpec(channels=512, stride=16),
                "res5": ShapeSpec(channels=1024, stride=32)
            }
        else:
            input_shape = {
                "res2": ShapeSpec(channels=256, stride=4),
                "res3": ShapeSpec(channels=512, stride=8),
                "res4": ShapeSpec(channels=1024, stride=16),
                "res5": ShapeSpec(channels=2048, stride=32)
            }
        args = {
            'input_shape': input_shape,
            'conv_dim': 256,
            'mask_dim': 256,
            'norm': 'GN',
            'transformer_dropout': dropout,
            'transformer_nheads': 8,
            'transformer_dim_feedforward': 1024,
            'transformer_enc_layers': 6,
            'transformer_in_features': ['res3', 'res4', 'res5'],
            'common_stride': 4,
        }
        if pixel_decoder == 'MSD':
            msd = MSDeformAttnPixelDecoder(**args)
            ckpt_path = cfg.MODEL.WEIGHTS[:-4] + '_MSDeformAttnPixelDecoder.pkl'
            # if os.path.exists(ckpt_path):
            msd_weights = torch.load(ckpt_path,
                                     map_location=torch.device('cpu'))
            msd_weights_new = msd_weights.copy()
            for k, v in msd_weights.items():
                if k[:7] == 'adapter':
                    msd_weights_new["lateral_convs." + k] = v
                    msd_weights_new.pop(k)
                elif k[:5] == 'layer':
                    msd_weights_new["output_convs." + k] = v
                    msd_weights_new.pop(k)
            msd.load_state_dict(msd_weights_new)
            print('Loaded MSD pixel decoder weights from {}'.format(ckpt_path))
            self.pixel_decoder = msd
            self.pixel_decoder.eval()
        elif pixel_decoder == 'FPN':
            args.pop('transformer_in_features')
            args.pop('common_stride')
            args['transformer_dim_feedforward'] = 2048
            args['transformer_pre_norm'] = False
            fpn = TransformerEncoderPixelDecoder(**args)
            ckpt_path = cfg.MODEL.WEIGHTS[:-4] + '_FPN.pkl'
            # if os.path.exists(ckpt_path):
            fpn_weights = torch.load(ckpt_path,
                                     map_location=torch.device('cpu'))
            fpn.load_state_dict(fpn_weights)
            self.pixel_decoder = fpn
            print('Loaded FPN pixel decoder weights from {}'.format(ckpt_path))
            self.pixel_decoder.eval()
        else:
            raise NotImplementedError

        # Load segmentation decoder
        self.load_segm_decoder = load_segm_decoder
        if self.load_segm_decoder:
            args = {
                "in_channels": 256,
                "mask_classification": True,
                "num_classes": 133,
                "hidden_dim": 256,
                "num_queries": 100,
                "nheads": 8,
                "dim_feedforward": 2048,
                "dec_layers": 9,
                "pre_norm": False,
                "mask_dim": 256,
                "enforce_input_project": False,
            }
            ckpt_path = cfg.MODEL.WEIGHTS[:-4] + '_transformer_decoder.pkl'
            mtd = MultiScaleMaskedTransformerDecoder(**args)
            mtd_weights = torch.load(ckpt_path,
                                     map_location=torch.device('cpu'))
            mtd.load_state_dict(mtd_weights)
            self.segm_decoder = mtd
            print('Loaded segmentation decoder weights from {}'.format(
                ckpt_path))
            self.segm_decoder.eval()

    def forward(self, x):
        features = self.backbone(x)
        high_res_featmaps, _, ms_feats = \
            self.pixel_decoder.forward_features(features)
        if self.load_segm_decoder:
            segm_predictions = self.segm_decoder.forward(
                ms_feats, high_res_featmaps)
            queries = segm_predictions["out_queries"]

            segm_results = self.segmentation_inference(segm_predictions)
            # segm_results = None
            return high_res_featmaps, queries, segm_results
        else:
            return high_res_featmaps, ms_feats[0], ms_feats[1]

    def segmentation_inference(self, segm_preds):
        """Compute panoptic segmentation from the outputs of the segmentation decoder."""
        mask_cls_results = segm_preds.pop("pred_logits")
        mask_pred_results = segm_preds.pop("pred_masks")

        processed_results = []
        for mask_cls_result, mask_pred_result in zip(mask_cls_results,
                                                     mask_pred_results):
            panoptic_r = self.panoptic_inference(mask_cls_result,
                                                 mask_pred_result)
            processed_results.append(panoptic_r)

        return processed_results

    def panoptic_inference(self,
                           mask_cls,
                           mask_pred,
                           object_mask_threshold=0.8,
                           overlap_threshold=0.8):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        # Remove non-object masks and masks with low confidence
        keep = labels.ne(mask_cls.size(-1) -
                         1) & (scores > object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w),
                                   dtype=torch.int32,
                                   device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        keep_ids = torch.where(keep)[0]

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return [], [], keep
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in range(80)
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item(
                ) > 0:
                    if mask_area / original_area < overlap_threshold:
                        keep[keep_ids[k]] = False
                        continue

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    my, mx = torch.where(mask)
                    segments_info.append({
                        "id":
                        current_segment_id,
                        "isthing":
                        bool(isthing),
                        "category_id":
                        int(pred_class),
                        "mask_area":
                        mask_area,
                        "mask_centroid": (mx.float().mean(), my.float().mean())
                    })
                else:
                    keep[keep_ids[k]] = False

            return panoptic_seg, segments_info, keep

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding2D(nn.Module):
    def __init__(self,
                 pa,
                 d_model: int,
                 dropout: float,
                 height: int = 20,
                 width: int = 32,
                 patch_num: list = None):
        super(PositionalEncoding2D, self).__init__()
        d_model = d_model // 2
        den = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) /
                        d_model)
        self.pa = pa
        self.n_special_symbols = len(pa.special_symbols)
        self.d_model = d_model

        pos_h = torch.arange(0, height).reshape(height, 1)
        pos_h_embedding = torch.zeros((height, d_model))
        pos_h_embedding[:, 0::2] = torch.sin(pos_h * den)
        pos_h_embedding[:, 1::2] = torch.cos(pos_h * den)
        pos_h_embedding = pos_h_embedding

        pos_w = torch.arange(0, width).reshape(width, 1)
        pos_w_embedding = torch.zeros((width, d_model))
        pos_w_embedding[:, 0::2] = torch.sin(pos_w * den)
        pos_w_embedding[:, 1::2] = torch.cos(pos_w * den)
        pos_w_embedding = pos_w_embedding
        self.height = height
        self.width = width

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('pos_w_embedding', pos_w_embedding)
        self.register_buffer('pos_h_embedding', pos_h_embedding)

    def forward(self, tgt_seq, scale=1):
        # Remove special_symbols
        gaze_symbol_idx = torch.logical_and(tgt_seq != self.pa.pad_idx,
                                            tgt_seq != self.pa.eos_idx)
        pe = torch.zeros(*tgt_seq.shape, self.d_model * 2).to(tgt_seq.device)
        if gaze_symbol_idx.sum() == 0:
            return pe
        
        actions = tgt_seq[gaze_symbol_idx] - self.n_special_symbols
        y = actions % (self.width / scale) + scale // 2
        x = actions // (self.width / scale) + scale // 2
        pe_valid = self.forward_pos(x, y)
        pe[gaze_symbol_idx] = pe_valid
        return pe

    def forward_pos(self, x, y):
        assert x.max() < self.height and y.max() < self.width, "out of range"
        pe_x = self.pos_h_embedding[x.long()]
        pe_y = self.pos_w_embedding[y.long()]
        pe = torch.cat([pe_x, pe_y], dim=1)
        return pe

    def forward_featmaps(self, featmaps, scale=1):
        h, w = featmaps.shape[-2:]
        smp_ind_x = torch.arange(scale // 2, h, scale)
        smp_ind_y = torch.arange(scale // 2, w, scale)
        pe_x = self.pos_h_embedding[smp_ind_x].transpose(0, 1)
        pe_y = self.pos_w_embedding[smp_ind_y].transpose(0, 1)
        pe_x = pe_x.unsqueeze(2).repeat(1, 1, w) #[96,1,160]
        pe_y = pe_y.unsqueeze(1).repeat(1, h, 1) #[96,160,1]
        pe = torch.cat([pe_x, pe_y], dim=0)
        return pe.unsqueeze(0)
        
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class MagSequencePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers):
        super(MagSequencePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x) * np.sqrt(x.size(-1))
        x = self.pos_encoder(x.permute(1, 0, 2))
        transformer_out = self.transformer_encoder(x)
        transformer_out = transformer_out[-1, :, :]
        out = self.fc(transformer_out)
        return out

class MagnificationMLP(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, output_size=6):
        super(MagnificationMLP, self).__init__()
        # Input layer with 6 inputs
        self.input_layer = nn.Linear(input_size, 64)
        # Hidden layer
        self.hidden_layer = nn.Linear(64, 32)
        # Output layer with 6 outputs (for 6-way classification)
        self.output_layer = nn.Linear(32, output_size)
        # Activation function
        self.relu = nn.ReLU()
        # Softmax for the final output to get probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        x = self.softmax(x)
        return x
        
class MagPred(nn.Module):
    def __init__(self):
        super(MagPred, self).__init__()
        
        # Convolutional layers for each of the 4 input maps
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # Output: 16 x 20 x 32
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1) # Output: 32 x 20 x 32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)        # Output: 32 x 10 x 16
        
        # Fully connected layers
        self.fc1 = nn.Linear(5120, 128)  # Combining all 4 maps
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)  # 6-way classification

    def forward(self, x1, x2, x3, x4):
        # Apply convolution, activation, and pooling to each input map
        x1 = self.pool(F.relu(self.conv2(F.relu(self.conv1(x1)))))
        x2 = self.pool(F.relu(self.conv2(F.relu(self.conv1(x2)))))
        x3 = self.pool(F.relu(self.conv2(F.relu(self.conv1(x3)))))
        x4 = self.pool(F.relu(self.conv2(F.relu(self.conv1(x4)))))

        # Flatten the outputs
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)
        
        # Concatenate the flattened outputs
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

        
# Dense prediction transformer
class Im2SpDenseTransformer(nn.Module):
    def __init__(self,
                 pa,
                 num_decoder_layers: int,
                 hidden_dim: int,
                 nhead: int,
                 ntask: int,
                 tgt_vocab_size: int,
                 num_output_layers: int,
                 separate_fix_arch: bool = False,
                 train_encoder: bool = False,
                 use_dino: bool = True,
                 pre_norm: bool = False,
                 dropout: float = 0.1,
                 dim_feedforward: int = 512,
                 parallel_arch: bool = False,
                 dorsal_source: list = ["P2"],
                 num_encoder_layers: int = 3,
                 output_centermap: bool = False,
                 output_saliency: bool = False,
                 output_target_map: bool = False,
                 combine_pos_emb: bool = True,
                 combine_all_emb: bool = False):
        super(Im2SpDenseTransformer, self).__init__()
        self.pa = pa
        self.num_decoder_layers = num_decoder_layers
        self.combine_pos_emb = combine_pos_emb
        self.combine_all_emb = combine_all_emb
        self.parallel_arch = parallel_arch
        self.dorsal_source = dorsal_source
        assert len(dorsal_source) > 0, "need to specify dorsal source: P1, P2!"
        self.output_centermap = output_centermap
        self.output_saliency = output_saliency
        self.output_target_map = output_target_map
        self.dropout = dropout
        # Encoder: Deformable Attention Transformer
        self.train_encoder = train_encoder
        self.encoder = ImageFeatureEncoder(pa.backbone_config, dropout, pa.pixel_decoder)
        self.symbol_offset = len(self.pa.special_symbols)
        if not train_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        #if hidden_dim != featmap_channels:
        self.input_proj = nn.Conv2d(384,
                                    192,
                                    kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)

        # Queries
        self.ntask = ntask
        self.aux_queries = 0

        self.query_embed = nn.Embedding(ntask + self.aux_queries, hidden_dim)
        self.query_pos = nn.Embedding(ntask + self.aux_queries, hidden_dim)

        # Decoder
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers_dorsal = nn.ModuleList()
        self.transformer_cross_attention_layers_ventral = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))
            self.transformer_cross_attention_layers_dorsal.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))
            if not self.parallel_arch:
                self.transformer_cross_attention_layers_ventral.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nhead,
                        dropout=dropout,
                        normalize_before=pre_norm,
                    ))
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))

        self.num_encoder_layers = num_encoder_layers
        if self.parallel_arch and num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                normalize_before=pre_norm)
            encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
            self.working_memory_encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm)

        self.decoder_norm = nn.LayerNorm(hidden_dim)
                                         
        self.magnification_change_predictor = MLP(hidden_dim + 1, hidden_dim, 6,num_output_layers)
                                         
        self.magnification_changer = MagPred()
        
        self.fixation_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                  num_output_layers)
                                         
        if self.output_target_map:
            self.target_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                    num_output_layers)
        if self.output_centermap:
            self.centermap_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                       num_output_layers)
        if self.output_saliency:
            self.saliency_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                      num_output_layers)

                                                  
        self.positional_encoding = PositionalEncoding(hidden_dim)

        self.fix_ind_emb = nn.Embedding(pa.max_traj_length, hidden_dim)
        
        self.mag_ind_emb = nn.Embedding(6, hidden_dim)
        
        self.magnification_MLP = MagnificationMLP(input_size=6, hidden_size=64, output_size=6).float()
        
        
        # Embedding for distinguishing dorsal or ventral embeddings
        self.dorsal_ind_emb = nn.Embedding(2, hidden_dim)  # P1 and P2
        self.ventral_ind_emb = nn.Embedding(1, hidden_dim)
        
        vocab_size = 6 #5
        embedding_dim = 10
        num_heads = 2 #4
        num_layers = 1 #2
        
        self.sig = nn.Sigmoid()

    def add_seq_pos_embed(self,tgt_seq, height, width, scale=1):
        val = 96*2
        pe = torch.zeros(*tgt_seq.shape, val * 2).to(tgt_seq.device)
        for i in range(tgt_seq.shape[1]):
            actions = tgt_seq[:,i]
            y1 = actions % (width / scale) + scale // 2
            x1 = actions // (width / scale) + scale // 2
            pe_valid = self.forward_pos_seq(x1,y1,width,height,val)
            pe[:,i,:] = pe_valid
            if i == 0:
                x,y = x1.unsqueeze(1),y1.unsqueeze(1)
            else:
                x = torch.cat((x,x1.unsqueeze(1)),1)
                y = torch.cat((y,y1.unsqueeze(1)),1)
                
        return pe,x,y

    def forward_pos_seq(self, x, y,width,height,val):
        assert x.max() < height and y.max() < width, "out of range"
        
        den = torch.exp(-torch.arange(0, val, 2) * math.log(10000) /val)
                        
        pos_h = torch.arange(0, height).reshape(height, 1)
        pos_h_embedding = torch.zeros((height, val)).to(x.long().device)
        pos_h_embedding[:, 0::2] = torch.sin(pos_h * den)
        pos_h_embedding[:, 1::2] = torch.cos(pos_h * den)

        pos_w = torch.arange(0, width).reshape(width, 1)
        pos_w_embedding = torch.zeros((width, val)).to(x.long().device)
        pos_w_embedding[:, 0::2] = torch.sin(pos_w * den)
        pos_w_embedding[:, 1::2] = torch.cos(pos_w * den)
        pe_x = pos_h_embedding[x.long()]
        pe_y = pos_w_embedding[y.long()]
        pe = torch.cat([pe_x, pe_y], dim=1)
        return pe
        
    def normalize_im(self,im):
        if np.max(im) > 0:
            im = (im-np.min(im))/(np.max(im)-np.min(im))
        else:
            im = im
        return im
        
    def visualize_maps(self,map_2x,map_4x,map_10x,map_20x,path1):
        #print('tt:',map_2x.shape,map_4x.shape,map_10x.shape)
        combo1 = np.concatenate((map_2x,map_4x),0)
        combo1 = np.concatenate((combo1,map_10x),0)
        combo1 = np.concatenate((combo1,map_20x),0)
        cv2.imwrite(path1,combo1)
        
    def forward(self,
                #img: torch.Tensor,
                tgt_seq: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                tgt_seq_high: torch.Tensor,
                mags_seq: torch.Tensor,
                img_names,
                low_res_embed,
                high_res_embed,
                # map_2x,
                # map_4x,
                # map_10x,
                # map_20x,
                next_mag,
                act_len,
                return_attn_weights=False
                ):

        img_embs_s1 = low_res_embed.cuda().permute(0,3,1,2) #.unsqueeze(0)

        high_res_featmaps = high_res_embed.cuda().permute(0,3,1,2) #.unsqueeze(0)
        
        
        dorsal_embs, dorsal_pos, scale_embs = [], [], []
        
        pixel_loc_embed_dorsal = PositionalEncoding2D(self.pa,
                                                  384,
                                                  height=img_embs_s1.size()[2],
                                                  width=img_embs_s1.size()[3],
                                                  dropout=self.dropout)
        pixel_loc_embed_ventral = PositionalEncoding2D(self.pa,
                                                  384,
                                                  height=high_res_featmaps.size()[2],
                                                  width=high_res_featmaps.size()[3],
                                                  dropout=self.dropout)
        
        
        if "P1" in self.dorsal_source:
            # C x 10 x 16
            img_embs = img_embs_s1

            bs, c, h, w = img_embs.shape

            pe = pixel_loc_embed_dorsal.forward_featmaps(img_embs, scale=1)

            img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)

            scale_embs.append(
                self.dorsal_ind_emb.weight[0].unsqueeze(0).unsqueeze(0).expand(
                    img_embs.size(0), bs, c))

            dorsal_embs.append(img_embs)
            dorsal_pos.append(
                pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))

            

        dorsal_embs = torch.cat(dorsal_embs, dim=0)
        dorsal_pos = torch.cat(dorsal_pos, dim=0)
        scale_embs = torch.cat(scale_embs, dim=0)

        # Prepare ventral embeddings
        tgt_seq = tgt_seq.transpose(0, 1)

        tgt_seq_high = tgt_seq_high.transpose(0, 1)

        highres_embs = high_res_featmaps.view(bs, c, -1).permute(2, 0, 1)
        ventral_embs = torch.gather(
            torch.cat([
            torch.zeros(1,*highres_embs.shape[1:],device=img_embs.device), 
            highres_embs],
            dim=0),
            0,
            tgt_seq_high.unsqueeze(-1).expand(*tgt_seq_high.shape,
                                              highres_embs.size(-1)))

        ventral_pos,xs,ys = self.add_seq_pos_embed(tgt_seq_high,high_res_featmaps.size()[2],high_res_featmaps.size()[3])  # Pos for fixation location

        
        # Add pos into embeddings for attention prediction
        if self.combine_pos_emb:
            # Dorsal embeddings
            dorsal_embs += dorsal_pos.to(dorsal_embs.device)
            dorsal_pos.fill_(0)
            # Ventral embeddings
            ventral_embs += ventral_pos.to(dorsal_embs.device)
            ventral_pos.fill_(0)
            high_res_featmaps += pixel_loc_embed_ventral.forward_featmaps(high_res_featmaps).to(high_res_featmaps.device)
        
        # Add embedding indicator embedding into pos embedding
        dorsal_pos += scale_embs.to(dorsal_pos.device)
        ventral_pos += self.ventral_ind_emb.weight.unsqueeze(0).expand(*ventral_pos.shape)

        ventral_pos += self.fix_ind_emb.weight[:ventral_embs.size(0)].unsqueeze(1).repeat(
                                                   1, bs, 1)
                                                   
        h = np.zeros((bs,6))
        for j in range(bs):
            for i in range(len(mags_seq[j])):
                h[j,mags_seq[j][i]] += 1
        h = h/200
        h = torch.tensor(h).cuda().float()

        mag_change = self.magnification_MLP(h)
        

        mag_emb = self.mag_ind_emb(mags_seq).permute(1,0,2)

        ventral_pos += mag_emb
        
        ventral_pos[tgt_padding_mask.transpose(0, 1)] = 0
        
        ventral_pos = ventral_pos.to(high_res_featmaps.device)
        dorsal_pos = dorsal_pos.to(high_res_featmaps.device)
        
        if self.combine_all_emb:
            dorsal_embs += dorsal_pos
            ventral_embs += ventral_pos
            dorsal_pos = ventral_pos = None

        # Update working memory
        if self.parallel_arch:
            working_memory = torch.cat([dorsal_embs, ventral_embs], dim=0)
            padding_mask = torch.cat([
                torch.zeros(bs, dorsal_embs.size(0),
                            device=dorsal_embs.device).bool(), tgt_padding_mask
            ],
                                     dim=1)
            if self.combine_all_emb:
                working_memory_pos = None
            else:
                working_memory_pos = torch.cat([dorsal_pos, ventral_pos],
                                               dim=0)#.to(working_memory_pos.device)

            working_memory = working_memory.to(torch.float32)
            if self.num_encoder_layers > 0:
                working_memory = self.working_memory_encoder(
                    working_memory,
                    src_key_padding_mask=padding_mask,
                    pos=working_memory_pos)
            dorsal_embs = working_memory
            dorsal_pos = working_memory_pos
        else:
            padding_mask = None

        # Update queries with attention
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, bs, 1)

        num_fixs = (tgt_padding_mask.size(1) -
                    torch.sum(tgt_padding_mask, dim=1)).unsqueeze(0).expand(
                        self.ntask, bs)
                        
        attn_weights_all_layers = []
        for i in range(self.num_decoder_layers):
            # Dorsal cross attention
            query_embed, attn_weights = self.transformer_cross_attention_layers_dorsal[
                i](query_embed,
                   dorsal_embs,
                   memory_mask=None,
                   memory_key_padding_mask=padding_mask,
                   pos=dorsal_pos,
                   query_pos=query_pos)
            if return_attn_weights:
                attn_weights_all_layers.append(attn_weights)

            if not self.parallel_arch:
                # Ventral cross attention
                query_embed, _ = self.transformer_cross_attention_layers_ventral[
                    i](query_embed,
                       ventral_embs,
                       memory_mask=None,
                       memory_key_padding_mask=tgt_padding_mask,
                       pos=ventral_pos,
                       query_pos=query_pos)

            if self.ntask > 1:
                # Self attention
                query_embed = self.transformer_self_attention_layers[i](
                    query_embed,
                    query_pos=query_pos,
                )

            # FFN
            query_embed = self.transformer_ffn_layers[i](query_embed)

        # Predictions
        out = {}
        
        out["pred_magnification"] = mag_change #output_magnification#.transpose(0, 1)
        
        fixation_embed = self.fixation_embed(query_embed[:self.ntask])
        outputs_fixation_map = torch.einsum("lbc,bchw->lbhw", fixation_embed.double(),high_res_featmaps.double())
        outputs_fixation_map = outputs_fixation_map.transpose(0, 1)
        out["pred_fixation_map"] = outputs_fixation_map #.transpose(0, 1)
        
        if return_attn_weights:
            out['cross_attn_weights'] = attn_weights_all_layers
        return out
        
    
    def encode(self,
                img_names,
                low_res_embed,
                high_res_embed
                ):
        # Prepare dorsal embeddings
        img_embs_s1 = torch.from_numpy(low_res_embed).cuda().permute(2,0,1).unsqueeze(0)
        high_res_embed = torch.from_numpy(high_res_embed).cuda().permute(2,0,1).unsqueeze(0)
        
        dorsal_embs, dorsal_pos, scale_embs = [], [], []
        
        pixel_loc_embed_dorsal = PositionalEncoding2D(self.pa,
                                                  384,
                                                  height=img_embs_s1.size()[2],
                                                  width=img_embs_s1.size()[3],
                                                  dropout=self.dropout)
                                                  
        if "P1" in self.dorsal_source:
            # C x 10 x 16
            img_embs = img_embs_s1
            bs, c, h, w = img_embs.shape
            pe = pixel_loc_embed_dorsal.forward_featmaps(img_embs, scale=1)
            img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
            scale_embs.append(self.dorsal_ind_emb.weight[0].unsqueeze(0).unsqueeze(0).expand(img_embs.size(0), bs, c))
            dorsal_embs.append(img_embs)
            dorsal_pos.append(
                pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))
                

        dorsal_embs = torch.cat(dorsal_embs, dim=0)
        dorsal_pos = torch.cat(dorsal_pos, dim=0)
        scale_embs = torch.cat(scale_embs, dim=0)
        dorsal_pos = (dorsal_pos, scale_embs)

        return dorsal_embs, dorsal_pos, None, high_res_embed
        
    
    def decode_and_predict(self,
                           # map_2x,map_4x,map_10x,map_20x,
                           low_res_embed,
                           high_res_embed,
                           dorsal_embs: torch.Tensor,
                           dorsal_pos: tuple,
                           dorsal_mask: torch.Tensor,
                           high_res_featmaps: torch.Tensor,
                           tgt_seq: torch.Tensor,
                           tgt_padding_mask: torch.Tensor,
                           tgt_seq_high: torch.Tensor,
                           mags_seq: torch.Tensor,
                           act_len,
                           mode,
                           task_ids: torch.Tensor = None,
                           return_attn_weights: bool = False):
                           
        mags1 = torch.transpose(mags_seq, 0, 1)
        img_embs_s1 = torch.from_numpy(low_res_embed).cuda().permute(2,0,1).unsqueeze(0)
        
        
        pixel_loc_embed_ventral = PositionalEncoding2D(self.pa,
                                                  384,
                                                  height=high_res_featmaps.size()[2],
                                                  width=high_res_featmaps.size()[3],
                                                  dropout=self.dropout)

                                                  
        # Prepare ventral embeddings
        dorsal_pos, scale_embs = dorsal_pos
        bs, c = high_res_featmaps.shape[:2]
        
        highres_embs = high_res_featmaps.view(bs, c, -1).permute(2, 0, 1)
        tgt_seq_high = tgt_seq_high.transpose(0, 1)

        if dorsal_mask is None:
            dorsal_mask = torch.zeros(1, *highres_embs.shape[1:], device=dorsal_embs.device)

        ventral_embs = torch.gather(
            torch.cat([dorsal_mask, highres_embs], dim=0), 0,
            tgt_seq_high.unsqueeze(-1).expand(*tgt_seq_high.shape,
                                              highres_embs.size(-1)))
                                              
        ventral_pos,xs,ys = self.add_seq_pos_embed(tgt_seq_high,high_res_featmaps.size()[2],high_res_featmaps.size()[3])  # Pos for fixation 

        # Add pos into embeddings for attention prediction
        if self.combine_pos_emb:
            # Dorsal embeddings
            dorsal_embs += dorsal_pos.to(dorsal_embs.device)
            dorsal_pos = torch.zeros_like(dorsal_pos)
            # Ventral embeddings
            ventral_embs += ventral_pos.to(dorsal_embs.device)
            ventral_pos = torch.zeros_like(ventral_pos)
            
            high_res_featmaps += pixel_loc_embed_ventral.forward_featmaps(high_res_featmaps).to(high_res_featmaps.device)
            
        # Add embedding indicator embedding into pos embedding
        dorsal_pos += scale_embs.to(dorsal_pos.device)
        ventral_pos += self.ventral_ind_emb.weight.unsqueeze(0).expand(
            *ventral_pos.shape)


        # Temporal embedding for fixations
        ventral_pos += self.fix_ind_emb.weight[:ventral_embs.
                                               size(0)].unsqueeze(1).repeat(
                                                   1, bs, 1)
        
        if mode == 'test':
            mag_emb = self.mag_ind_emb(mags_seq)#.permute(1,0,2)
            ventral_pos += mag_emb
        else:
            mags_seq_ext = mags_seq[0].tolist()
            mags_seq_ext = mags_seq_ext + [mags_seq_ext[-1]] * (self.pa.max_traj_length - len(mags_seq_ext))
            mags_seq_ext = torch.tensor(mags_seq_ext).unsqueeze(0).cuda()
            mag_emb = self.mag_ind_emb(mags_seq_ext).permute(1,0,2)
            ventral_pos += mag_emb
            

        ventral_pos = ventral_pos.to(high_res_featmaps.device)
        dorsal_pos = dorsal_pos.to(high_res_featmaps.device)

        
        if self.combine_all_emb:
            dorsal_embs += dorsal_pos
            ventral_embs += ventral_pos
            dorsal_pos = ventral_pos = None

        # Update working memory with both dorsal and ventral memory
        # if using parallel architecture
        if self.parallel_arch:
            working_memory = torch.cat([dorsal_embs, ventral_embs], dim=0)
            padding_mask = torch.cat(
                [
                    torch.zeros(
                        bs, dorsal_embs.size(0),
                        device=dorsal_embs.device).bool(), tgt_padding_mask
                ],
                dim=1) if tgt_padding_mask is not None else None
            if self.combine_all_emb:
                working_memory_pos = None
            else:
                working_memory_pos = torch.cat([dorsal_pos, ventral_pos],
                                               dim=0)

            working_memory = working_memory.to(torch.float32)
            
            if self.num_encoder_layers > 0:
                working_memory = self.working_memory_encoder(
                    working_memory,
                    src_key_padding_mask=padding_mask,
                    pos=working_memory_pos)
            dorsal_embs = working_memory
            dorsal_pos = working_memory_pos
        else:
            padding_mask = None

        # Update queries with attention
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, bs, 1)
        num_fixs = torch.ones(self.ntask, bs).to(
            query_embed.device) * tgt_seq_high.size(0)

        attn_weights_all_layers = []
        for i in range(self.num_decoder_layers):
            # Dorsal cross attention
            query_embed, attn_weights = self.transformer_cross_attention_layers_dorsal[
                i](query_embed,
                   dorsal_embs,
                   memory_mask=None,
                   memory_key_padding_mask=padding_mask,
                   pos=dorsal_pos,
                   query_pos=query_pos)
            if return_attn_weights:
                attn_weights_all_layers.append(attn_weights)

            if not self.parallel_arch:
                # Ventral cross attention
                query_embed, _ = self.transformer_cross_attention_layers_ventral[
                    i](query_embed,
                       ventral_embs,
                       memory_mask=None,
                       memory_key_padding_mask=tgt_padding_mask,
                       pos=ventral_pos,
                       query_pos=query_pos)
            if self.ntask > 1:
                # Self attention
                query_embed = self.transformer_self_attention_layers[i](
                    query_embed,
                    query_pos=query_pos,
                )

            # FFN
            query_embed = self.transformer_ffn_layers[i](query_embed)

        # Prediction

        h = np.zeros((1,6))
        if mode == 'test':
            mags_seq = mags_seq.T
        for j in range(1):
            for i in range(len(mags_seq[j])):
                h[j,mags_seq[j][i]] += 1
        h = h/200
        h = torch.tensor(h)
        combo = h.cuda().float()
        mag_change = self.magnification_MLP(combo)
        
        
        out = {
            "pred_magnification": mag_change
        }

        fixation_embed = self.fixation_embed(query_embed[:self.ntask])
        outputs_fixation_map = torch.einsum("lbc,bchw->lbhw", fixation_embed.double(),
                                            high_res_featmaps.double())
        outputs_fixation_map = outputs_fixation_map.transpose(0, 1)
        #outputs_fixation_map = self.sig(outputs_fixation_map)
        
        outputs_fixation_map = torch.sigmoid(
            outputs_fixation_map[task_ids, torch.arange(bs)])
        outputs_fixation_map = F.interpolate(outputs_fixation_map.unsqueeze(1),
                                             size=(self.pa.im_h,
                                                   self.pa.im_w)).squeeze(1)
        outputs_fixation_map = outputs_fixation_map.view(bs, -1)
        out["pred_fixation_map"] = outputs_fixation_map #.view(bs, -1)

        del outputs_fixation_map,fixation_embed,high_res_featmaps,img_embs_s1,highres_embs
        
        if return_attn_weights:
            out['cross_attn_weights'] = attn_weights_all_layers
        return out
