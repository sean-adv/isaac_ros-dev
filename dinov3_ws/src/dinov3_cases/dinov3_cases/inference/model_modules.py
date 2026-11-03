"""Tensor-only wrappers around official Hugging Face model modules."""
import torch
from transformers.models.sam3.modeling_sam3 import inverse_sigmoid, box_cxcywh_to_xyxy


class Cat2RealFeatures(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).last_hidden_state


class Sam3VisionFeatures(torch.nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, pixel_values):
        outputs = self.vision_encoder(pixel_values=pixel_values)
        return (outputs.last_hidden_state, *outputs.fpn_hidden_states, *outputs.fpn_position_encoding)


class Sam3BboxHead(torch.nn.Module):
    """Stock SAM3 bbox computation, without allocating segmentation masks."""
    def __init__(self, model):
        super().__init__()
        self.encoder = model.detr_encoder
        self.decoder = model.detr_decoder
        self.scoring = model.dot_product_scoring

    def forward(self, image_features, image_position, text_features, text_mask):
        count = text_features.shape[0]
        enc = self.encoder(
            vision_features=[image_features.expand(count, -1, -1, -1)],
            text_features=text_features,
            vision_pos_embeds=[image_position.expand(count, -1, -1, -1)],
            text_mask=text_mask,
        )
        dec = self.decoder(
            vision_features=enc.last_hidden_state, text_features=enc.text_features,
            vision_pos_encoding=enc.pos_embeds_flattened,
            text_mask=text_mask, spatial_shapes=enc.spatial_shapes,
        )
        offsets = self.decoder.box_head(dec.intermediate_hidden_states)
        boxes = box_cxcywh_to_xyxy((inverse_sigmoid(dec.reference_boxes) + offsets).sigmoid())[-1]
        logits = self.scoring(
            decoder_hidden_states=dec.intermediate_hidden_states,
            text_features=enc.text_features, text_mask=text_mask,
        ).squeeze(-1)[-1]
        return boxes, logits, dec.presence_logits[-1]
