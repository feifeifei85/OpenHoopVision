import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# PARSeq architecture adapted from Scene Text Recognition Model Hub (PARSeq),
# Copyright 2022 Darwin Bautista, licensed under Apache-2.0.
# https://www.apache.org/licenses/LICENSE-2.0
class _PARSeqTokenizer:
    def __init__(self, charset):
        self._itos = ("[E]", *charset, "[B]", "[P]")
        self.eos_id = 0


class _TokenEmbedding(nn.Module):
    def __init__(self, charset_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class _PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, images):
        if tuple(images.shape[-2:]) != self.img_size:
            raise ValueError(f"Expected image size {self.img_size}, got {tuple(images.shape[-2:])}")
        return self.proj(images).flatten(2).transpose(1, 2)


class _VisionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x):
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(x))


class _VisionMlp(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x):
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


class _VisionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = _VisionAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = _VisionMlp(embed_dim, int(embed_dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _PARSeqEncoder(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio):
        super().__init__()
        self.patch_embed = _PatchEmbed(img_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.empty(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(0.0)
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.Sequential(*[
            _VisionBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, images):
        x = self.patch_embed(images)
        x = self.pos_drop(x + self.pos_embed)
        return self.norm(self.blocks(self.norm_pre(self.patch_drop(x))))


class _PARSeqDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm_q = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm_c = nn.LayerNorm(embed_dim, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.gelu

    def _forward_stream(self, target, target_norm, target_kv, memory, target_mask, padding_mask):
        target2, _ = self.self_attn(
            target_norm, target_kv, target_kv, attn_mask=target_mask, key_padding_mask=padding_mask
        )
        target = target + self.dropout1(target2)
        target2, _ = self.cross_attn(self.norm1(target), memory, memory)
        target = target + self.dropout2(target2)
        target2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(target)))))
        return target + self.dropout3(target2)

    def forward(self, query, content, memory, query_mask, content_mask, padding_mask, update_content):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self._forward_stream(query, query_norm, content_norm, memory, query_mask, padding_mask)
        if update_content:
            content = self._forward_stream(content, content_norm, content_norm, memory, content_mask, padding_mask)
        return query, content


class _PARSeqDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, depth, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            _PARSeqDecoderLayer(embed_dim, num_heads, embed_dim * mlp_ratio, dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, content, memory, query_mask, content_mask, padding_mask):
        for index, layer in enumerate(self.layers):
            query, content = layer(
                query, content, memory, query_mask, content_mask, padding_mask,
                update_content=index < len(self.layers) - 1,
            )
        return self.norm(query)


class _PARSeq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = tuple(config["img_size"])
        self.max_label_length = int(config["max_label_length"])
        self.decode_ar = bool(config["decode_ar"])
        self.refine_iters = int(config["refine_iters"])
        embed_dim = int(config["embed_dim"])
        self.tokenizer = _PARSeqTokenizer(config["charset_train"])
        self.eos_id = self.tokenizer.eos_id
        self.bos_id = len(self.tokenizer._itos) - 2
        self.pad_id = len(self.tokenizer._itos) - 1
        self.encoder = _PARSeqEncoder(
            self.img_size,
            config["patch_size"],
            embed_dim,
            int(config["enc_depth"]),
            int(config["enc_num_heads"]),
            int(config["enc_mlp_ratio"]),
        )
        self.decoder = _PARSeqDecoder(
            embed_dim,
            int(config["dec_num_heads"]),
            int(config["dec_mlp_ratio"]),
            int(config["dec_depth"]),
            float(config["dropout"]),
        )
        self.head = nn.Linear(embed_dim, len(self.tokenizer._itos) - 2)
        self.text_embed = _TokenEmbedding(len(self.tokenizer._itos), embed_dim)
        self.pos_queries = nn.Parameter(torch.empty(1, self.max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(float(config["dropout"]))

    def _decode(self, target, memory, target_mask=None, padding_mask=None, target_query=None, target_query_mask=None):
        batch, length = target.shape
        null_context = self.text_embed(target[:, :1])
        target_embed = self.pos_queries[:, :length - 1] + self.text_embed(target[:, 1:])
        target_embed = self.dropout(torch.cat([null_context, target_embed], dim=1))
        if target_query is None:
            target_query = self.pos_queries[:, :length].expand(batch, -1, -1)
        target_query = self.dropout(target_query)
        return self.decoder(
            target_query, target_embed, memory, target_query_mask, target_mask, padding_mask
        )

    def forward(self, images, max_length=None):
        testing = max_length is None
        max_length = self.max_label_length if testing else min(max_length, self.max_label_length)
        batch = images.shape[0]
        num_steps = max_length + 1
        memory = self.encoder(images)
        pos_queries = self.pos_queries[:, :num_steps].expand(batch, -1, -1)
        mask = torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=images.device), 1)
        tgt_mask = query_mask = mask

        if self.decode_ar:
            tgt_in = torch.full((batch, num_steps), self.pad_id, dtype=torch.long, device=images.device)
            tgt_in[:, 0] = self.bos_id
            logits = []
            for i in range(num_steps):
                j = i + 1
                tgt_out = self._decode(
                    tgt_in[:, :j], memory, tgt_mask[:j, :j],
                    target_query=pos_queries[:, i:j], target_query_mask=query_mask[i:j, :j],
                )
                next_logits = self.head(tgt_out)
                logits.append(next_logits)
                if j < num_steps:
                    tgt_in[:, j] = next_logits.squeeze().argmax(-1)
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break
            logits = torch.cat(logits, dim=1)
        else:
            tgt_in = torch.full((batch, 1), self.bos_id, dtype=torch.long, device=images.device)
            logits = self.head(self._decode(tgt_in, memory, target_query=pos_queries))

        if self.refine_iters:
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=images.device), 2)] = False
            bos = torch.full((batch, 1), self.bos_id, dtype=torch.long, device=images.device)
            for _ in range(self.refine_iters):
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                padding_mask = (tgt_in == self.eos_id).int().cumsum(-1) > 0
                logits = self.head(self._decode(
                    tgt_in,
                    memory,
                    tgt_mask,
                    padding_mask,
                    target_query=pos_queries,
                    target_query_mask=query_mask[:, :tgt_in.shape[1]],
                ))
        return logits


class JerseyNumberReader:
    """Read jersey numbers with PARSeq."""

    def __init__(self, parameters, device):
        self.parameters = parameters or {}
        self.device = torch.device(device)
        self.model = None
        self.transform = None
        self.image_type = None
        self.error = None
        self.observations = 0
        self.accepted = 0

    def _load(self):
        if self.model is not None:
            return True
        if self.error is not None or not self.parameters.get("enabled", True):
            return False
        try:
            from PIL import Image
            from torchvision import transforms as T

            checkpoint_path = self.parameters.get("checkpoint")
            if checkpoint_path is None:
                raise ValueError("PARSeq checkpoint path is not configured")
            checkpoint = Path(checkpoint_path).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"PARSeq checkpoint not found: {checkpoint}")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model = _PARSeq(saved["hyper_parameters"])
            self.model.load_state_dict(saved["state_dict"], strict=True)
            self.model = self.model.eval().to(self.device)
            self.transform = T.Compose(
                [
                    T.Resize(self.model.img_size, T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(0.5, 0.5),
                ]
            )
            self.image_type = Image
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"warning: jersey-number OCR is unavailable: {self.error}")
            return False

    @staticmethod
    def _decode(probabilities, tokenizer):
        digits, scores = [], []
        for position in range(3):
            token = int(probabilities[position].argmax())
            if token == tokenizer.eos_id:
                break
            digit = tokenizer._itos[token]
            if not digit.isdigit():
                break
            digits.append(digit)
            scores.append(float(probabilities[position, token]))
        number = "".join(digits) if 1 <= len(digits) <= 2 else ""
        confidence = float(np.mean(scores)) if number else 0.0
        return number, confidence

    def predict(self, crops):
        if not crops or not self._load():
            return []
        inputs = []
        for row in crops:
            crop = row.get("full")
            if crop is None or not crop.size:
                inputs.append(None)
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inputs.append(self.transform(self.image_type.fromarray(rgb)))
        valid = [(index, image) for index, image in enumerate(inputs) if image is not None]
        if not valid:
            return [{} for _ in crops]
        self.observations += len(crops)
        batch = torch.stack([image for _, image in valid]).to(self.device, non_blocking=True)
        with torch.inference_mode():
            probabilities = self.model(batch, max_length=2)[:, :3, :11].softmax(-1).float().cpu()
        result = [{} for _ in crops]
        threshold = float(self.parameters.get("min_read_confidence", 0.55))
        accepted_threshold = float(self.parameters.get("confidence", 0.80))
        for (index, _), probs in zip(valid, probabilities):
            number, confidence = self._decode(probs, self.model.tokenizer)
            if number and confidence >= threshold:
                result[index] = {"number": number, "confidence": float(confidence)}
                self.accepted += confidence >= accepted_threshold
        return result



