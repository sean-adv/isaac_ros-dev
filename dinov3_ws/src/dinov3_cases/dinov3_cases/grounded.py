"""Text-guided proposals followed by per-object DINOv3 gallery retrieval."""
import math
import re
import copy
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F


DEFAULT_PROMPT = 'medicine bottle. medicine box. bottle. box.'
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def validate_settings(prompt, box_threshold, text_threshold, recognition_threshold, margin):
    if not isinstance(prompt, str) or not prompt.strip(' .\t\r\n') or len(prompt) > 1000:
        raise ValueError('Prompt must contain 1–1000 characters.')
    for name, value, upper in [('box_threshold', box_threshold, 1),
                               ('text_threshold', text_threshold, 1),
                               ('recognition_threshold', recognition_threshold, 1),
                               ('recognition_margin', margin, 2)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= upper:
            raise ValueError(f'{name} must be a finite number between 0 and {upper}.')
    return prompt.strip().rstrip('.') + '.'


def clean_proposals(boxes, scores, labels, size, nms_iou=0.5, limit=20):
    """Clip original-image XYXY boxes; suppress duplicate prompts, never union boxes."""
    width, height = size
    candidates = []
    for box, score, label in zip(boxes, scores, labels):
        if len(box) != 4 or not np.isfinite([*box, score]).all():
            continue
        x0, y0, x1, y1 = box
        x0, y0 = max(0, math.floor(x0)), max(0, math.floor(y0))
        x1, y1 = min(width, math.ceil(x1)), min(height, math.ceil(y1))
        if x1 <= x0 or y1 <= y0:
            continue
        candidates.append(dict(bbox=[x0, y0, x1, y1], detection_score=float(score),
                               detection_label=str(label), area=(x1-x0)*(y1-y0)))
    kept = []
    for item in sorted(candidates, key=lambda x: x['detection_score'], reverse=True):
        a = item['bbox']
        duplicate = False
        for other in kept:
            b = other['bbox']
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
            if intersection / (item['area'] + other['area'] - intersection) > nms_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
            if len(kept) >= limit:
                break
    return kept


class GroundingDetector:
    def __init__(self, model_id, device, cache_dir=None, backend='torch', engine_path=''):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        if backend == 'trt':
            from .grounding_trt import GroundingTRT
            self.model = GroundingTRT(engine_path, model_id, device)
            self.processor = AutoProcessor.from_pretrained(
                self.model.processor_path, local_files_only=True)
        elif backend == 'torch':
            self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            # The community repository supplies ONNX only. Use its upstream
            # weights when explicitly selecting the PyTorch backend.
            torch_model_id = ('IDEA-Research/grounding-dino-tiny'
                              if model_id == 'onnx-community/grounding-dino-tiny-ONNX' else model_id)
            torch_cache = (str(Path(cache_dir).parent / 'grounding_dino')
                           if cache_dir and model_id == 'onnx-community/grounding-dino-tiny-ONNX'
                           else cache_dir)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                torch_model_id, cache_dir=torch_cache, disable_custom_kernels=True).to(device).eval()
        else:
            raise ValueError('grounding_backend must be torch or trt.')
        self.backend = backend
        self.device = device

    def validate_prompt(self, prompt):
        if self.backend == 'trt':
            self.model.validate_prompt(self.processor, prompt)

    @torch.inference_mode()
    def detect(self, image, prompt, box_threshold, text_threshold, nms_iou, limit):
        if self.backend == 'trt':
            inputs = self.model.prepare_inputs(self.processor, image, prompt)
        else:
            inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(self.device)
        if inputs.input_ids.shape[-1] > self.model.config.max_text_len:
            raise ValueError('Prompt exceeds Grounding DINO token limit; use fewer categories.')
        outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)])[0]
        return clean_proposals(result['boxes'].cpu().tolist(), result['scores'].cpu().tolist(),
                               result['text_labels'], image.size, nms_iou, limit)


def prepare_object(image, max_side=224, stride=16, minimum=32):
    """Apply the same aspect-preserving resize/padding to references and proposals."""
    if max_side < minimum or stride <= 0:
        raise ValueError('Object size must cover the minimum input size and stride must be positive.')
    image = image.convert('RGB')
    scale = max_side / max(image.size)
    width, height = [max(1, round(n * scale)) for n in image.size]
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    padded_w = max(minimum, math.ceil(width / stride) * stride)
    padded_h = max(minimum, math.ceil(height / stride) * stride)
    pixels = np.pad(np.asarray(image), ((0, padded_h-height), (0, padded_w-width), (0, 0)), mode='edge')
    return Image.fromarray(pixels)


@torch.inference_mode()
def encode_object(encoder, image, max_side, stride):
    prepared = prepare_object(image, max_side, stride)
    features = encoder.encode(prepared)
    vector = features.float().mean(dim=0)
    if not torch.isfinite(vector).all() or vector.norm() < 1e-8:
        raise ValueError('Invalid object descriptor.')
    return prepared, features, F.normalize(vector, dim=0)


def object_descriptor(encoder, image, max_side, stride):
    return encode_object(encoder, image, max_side, stride)[2]


def gallery_signature(root):
    entries = []
    for folder in sorted(Path(root).iterdir()):
        if folder.is_symlink() or not folder.is_dir() or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', folder.name):
            continue
        for path in sorted(folder.iterdir()):
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            stat = path.stat()
            entries.append((folder.name, str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(entries)


class ObjectGallery:
    def __init__(self):
        self.key = None
        self.features = None
        self.labels = []
        self.paths = []
        self.patch_banks = []
        self.reference_inputs = []

    def refresh(self, root, encoder, max_side, stride):
        signature = gallery_signature(root)
        key = (signature, max_side, stride)
        if key == self.key:
            return False
        features, labels, patch_banks, reference_inputs = [], [], [], []
        for label, filename, _, _ in signature:
            with Image.open(filename) as source:
                if source.width * source.height > 25_000_000:
                    raise ValueError(f'Reference exceeds 25 megapixels: {filename}')
                image = ImageOps.exif_transpose(source).convert('RGB')
            prepared, patches, descriptor = encode_object(encoder, image, max_side, stride)
            features.append(descriptor)
            patch_banks.append(patches)
            reference_inputs.append(prepared)
            labels.append(label)
        # Publish a complete gallery only after all references have encoded successfully.
        self.features = torch.stack(features) if features else None
        self.labels, self.key = labels, key
        self.paths = [entry[1] for entry in signature]
        self.patch_banks = patch_banks
        self.reference_inputs = reference_inputs
        return True

    def best_reference(self, descriptor, case=''):
        indices = [i for i, label in enumerate(self.labels) if not case or label == case]
        if not indices:
            return None, None
        scores = (self.features[indices] @ descriptor).clamp(-1, 1)
        best = int(scores.argmax())
        return indices[best], float(scores[best])

    def classify(self, descriptor, threshold, margin):
        if self.features is None:
            return dict(label='unknown', score=None, candidate=None, margin=None, reason='empty_gallery')
        similarities = (self.features @ descriptor).clamp(-1, 1).cpu().tolist()
        by_class = {}
        for label, score in zip(self.labels, similarities):
            by_class[label] = max(by_class.get(label, -1.), score)
        ranked = sorted(by_class.items(), key=lambda item: (-item[1], item[0]))
        label, score = ranked[0]
        best_reference = max((i for i, name in enumerate(self.labels) if name == label),
                             key=lambda i: similarities[i])
        gap = score - ranked[1][1] if len(ranked) > 1 else None
        reason = ('low_similarity' if score < threshold else
                  'ambiguous' if gap is not None and gap < margin else 'matched')
        return dict(label=label if reason == 'matched' else 'unknown', score=score,
                    candidate=label, margin=gap, reason=reason,
                    reference=dict(case=label, name=Path(self.paths[best_reference]).name),
                    _reference_index=best_reference)


def object_debug_views(roi, prepared, features, reference_patches, max_side, stride, roi_size=None):
    """Use the existing forward's patch tokens; remove input padding before ROI mapping."""
    from .patches import compute_similarity_maps, reference_grid_image
    gh, gw = prepared.height // stride, prepared.width // stride
    roi_size = roi.size if roi_size is None else roi_size
    scale = max_side / max(roi_size)
    content_w, content_h = [max(1, round(n * scale)) for n in roi_size]
    preview = roi.copy()
    preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
    views = dict(crop=cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2BGR),
                 grid=cv2.cvtColor(np.asarray(reference_grid_image(prepared, stride)), cv2.COLOR_RGB2BGR))
    # Mark the real ROI's edge; the right/bottom remainder is replicate padding.
    cv2.rectangle(views['grid'], (0, 0), (content_w-1, content_h-1), (255, 130, 50), 1)
    metadata = dict(roi_size=list(roi_size), input_size=list(prepared.size),
                    preview_size=list(preview.size),
                    content_size=[content_w, content_h], patch_grid=[gh, gw],
                    heatmap_available=reference_patches is not None)
    if reference_patches is None:
        return views, metadata
    scores = compute_similarity_maps(features.reshape(1, gh, gw, -1), reference_patches)[0].cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite object patch similarities.')
    def color(values):
        return cv2.applyColorMap((np.clip(values, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    # Patch rendering shows the full model grid. The ROI heatmap excludes padding.
    views['patches'] = cv2.resize(color(scores), prepared.size, interpolation=cv2.INTER_NEAREST)
    pixels = cv2.resize(scores, prepared.size, interpolation=cv2.INTER_LINEAR)[:content_h, :content_w]
    mapped = cv2.resize(pixels, preview.size, interpolation=cv2.INTER_LINEAR)
    views['similarity'] = color(mapped)
    views['heatmap'] = cv2.addWeighted(views['crop'], .55, views['similarity'], .45, 0)
    metadata.update(similarity_min=float(mapped.min()), similarity_max=float(mapped.max()))
    return views, metadata


def build_object_diagnostic(context, gallery, max_side, stride, reference_case='', reference_bank=None):
    index, score = gallery.best_reference(context['descriptor'], reference_case)
    count = sum(label == reference_case for label in gallery.labels) if reference_case else int(index is not None)
    bank = None if index is None else gallery.patch_banks[index]
    if reference_case and index is not None:
        bank = reference_bank if reference_bank is not None else torch.cat(
            [patches for label, patches in zip(gallery.labels, gallery.patch_banks) if label == reference_case])
    views, metadata = object_debug_views(context['roi'], context['prepared'], context['patches'],
                                        bank, max_side, stride, context['roi_size'])
    reference = None
    if index is not None:
        reference = dict(case=gallery.labels[index], name=Path(gallery.paths[index]).name)
        views['reference'] = cv2.cvtColor(np.asarray(gallery.reference_inputs[index]), cv2.COLOR_RGB2BGR)
    metadata.update(comparison_case=reference_case or (reference['case'] if reference else None),
                    comparison_score=score, comparison_reference=reference, comparison_reference_count=count)
    return dict(metadata=metadata, views=views, context=context)


def encode_object_diagnostics(detections, diagnostics):
    images, items = {}, []
    for index, (item, diagnostic) in enumerate(zip(detections, diagnostics)):
        for view, pixels in diagnostic['views'].items():
            if max(pixels.shape[:2]) > 640:
                scale = 640 / max(pixels.shape[:2])
                pixels = cv2.resize(pixels, (max(1, round(pixels.shape[1]*scale)),
                                           max(1, round(pixels.shape[0]*scale))))
            ok, data = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError(f'Object {index} {view} JPEG encoding failed.')
            images[index, view] = data.tobytes()
        items.append(dict(item, index=index, **diagnostic['metadata'], views=list(diagnostic['views'])))
    return items, images


class ObjectVisualizationSnapshot:
    """Render reference comparisons from one immutable inference/gallery revision."""
    def __init__(self, detections, diagnostics, gallery, max_side, stride):
        self.detections = detections
        self.contexts = [item['context'] for item in diagnostics]
        # Gallery refresh replaces its lists/tensors; the snapshot keeps the old complete set.
        self.gallery = copy.copy(gallery)
        self.reference_cases = sorted(set(gallery.labels))
        self.max_side, self.stride = max_side, stride
        self.lock = threading.Lock()
        self.cache = OrderedDict()

    def render(self, reference_case=''):
        if reference_case and reference_case not in self.reference_cases:
            raise ValueError('Reference label has no images in this result.')
        with self.lock:
            if reference_case not in self.cache:
                bank = (torch.cat([patches for label, patches in zip(self.gallery.labels, self.gallery.patch_banks)
                                   if label == reference_case]) if reference_case else None)
                diagnostics = [build_object_diagnostic(context, self.gallery, self.max_side,
                                                      self.stride, reference_case, bank)
                               for context in self.contexts]
                self.cache[reference_case] = encode_object_diagnostics(self.detections, diagnostics)
                # Keep automatic views and at most two selected-label comparisons per frame.
                if len(self.cache) > 3:
                    self.cache.pop(next(key for key in self.cache if key))
            self.cache.move_to_end(reference_case)
            return self.cache[reference_case]


def recognize_objects(image, proposals, encoder, gallery, max_side, stride, threshold, margin,
                      diagnostics=None, defer_diagnostics=False):
    results = []
    for proposal in proposals:
        roi = image.crop(proposal['bbox'])
        prepared, patches, descriptor = encode_object(encoder, roi, max_side, stride)
        classification = gallery.classify(descriptor, threshold, margin)
        classification.pop('_reference_index', None)
        result = dict(proposal, **classification)
        if diagnostics is not None:
            preview = roi.copy()
            preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
            context = dict(roi=preview, roi_size=roi.size, prepared=prepared,
                           patches=patches, descriptor=descriptor)
            diagnostics.append({'context': context} if defer_diagnostics else
                               build_object_diagnostic(context, gallery, max_side, stride))
        # Detection rectangles are not segmentation masks for surface pose estimation.
        result.update(pose=None, pose_error='Object segmentation is required for surface pose')
        results.append(result)
    return results


def recognition_message(items, header=None):
    from .boxes import detection_message
    from vision_msgs.msg import Detection2DArray
    message = Detection2DArray()
    if header is not None:
        message.header = header
    for index, item in enumerate(items):
        score = item['score'] if item['score'] is not None else 0.0
        detection = detection_message([(item['bbox'], score)], item['label'], header).detections[0]
        detection.id = str(index)
        message.detections.append(detection)
    return message
