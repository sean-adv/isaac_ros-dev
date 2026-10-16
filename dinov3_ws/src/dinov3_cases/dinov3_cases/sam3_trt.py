"""SAM3 DARTF W8A8 backbone + FP16 grounding heads."""
from pathlib import Path
import ctypes
import threading
import numpy as np
from PIL import Image


def parse_prompts(prompt):
    if not isinstance(prompt, str):
        raise ValueError("SAM3 prompt must be text")
    labels = [s.strip() for s in prompt.strip().rstrip('.').split(',') if s.strip()]
    if not 1 <= len(labels) <= 4:
        raise ValueError("SAM3 requires one to four comma-separated class names")
    return labels


class Sam3Detector:
    def __init__(self, engine_path, device):
        # Preserve the bundle directory: its backbone may symlink to out_bbox/.
        engine_path = Path(engine_path).expanduser().absolute()
        self.assets = engine_path.parent
        from .runtime import SAM3_PLUGIN
        required = [engine_path, SAM3_PLUGIN,
                    self.assets / 'text_c16.onnx', self.assets / 'bpe_simple_vocab_16e6.txt.gz',
                    *[self.assets / f'ground_c{k}m_fp16.plan' for k in (1, 4)],
                    *[self.assets / f'img_pos_c{k}.npy' for k in (1, 4)]]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                'Incomplete SAM3 deployment: ' + ', '.join(missing)
                + '. Run DART/dartf/scripts/build_bbox.sh to prepare the bbox bundle.')
        import tensorrt as trt
        if trt.__version__.split('.')[0] != "10":
            raise RuntimeError(f"SAM3 deployment requires TensorRT 10; got {trt.__version__}. Use DART/.venv/bin/python.")
        if str(device) not in ('cuda', 'cuda:0'):
            raise ValueError("SAM3 deployment requires cuda:0")
        self.plugin = ctypes.CDLL(str(SAM3_PLUGIN), mode=ctypes.RTLD_GLOBAL)
        self.plugin.lq_register()
        from .sam3_runtime.trt_util import Runner, load_engine
        from .sam3_runtime.tokenizer import Sam3Tokenizer
        import onnxruntime as ort
        self.vision = Runner(load_engine(str(engine_path)))
        self.heads = {k: Runner(load_engine(str(self.assets / f'ground_c{k}m_fp16.plan'))) for k in (1, 4)}
        self.positions = {k: np.load(self.assets / f'img_pos_c{k}.npy') for k in (1, 4)}
        self.tokenizer = Sam3Tokenizer(self.assets / 'bpe_simple_vocab_16e6.txt.gz')
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.text = ort.InferenceSession(str(self.assets / 'text_c16.onnx'), sess_options=opts, providers=['CPUExecutionProvider'])
        self.precision = 'W8A8 backbone / FP16 head'
        self.cached_prompt = None
        self.lock = threading.RLock()

    def validate_prompt(self, prompt):
        labels = parse_prompts(prompt)
        for label in labels:
            self.tokenizer.encode(label)
        return labels

    def _text_features(self, labels):
        key = tuple(labels)
        if key != self.cached_prompt:
            ids = np.zeros((16, 32), np.int32)
            ids[:, 0] = 49406
            ids[:, 1] = 49407
            for i, label in enumerate(labels):
                tokens = self.tokenizer.encode(label)
                ids[i] = 0
                ids[i, :len(tokens)] = tokens
            tf, tm = self.text.run(None, {'token_ids': ids})
            bucket = 1 if len(labels) == 1 else 4
            self.features = (bucket, tf[:, :bucket].astype(np.float16), tm[:bucket].astype(np.float32))
            self.cached_prompt = key
        return self.features

    def detect(self, image, prompt, box_threshold, nms_iou, limit,
               *, min_width=0, min_height=0, min_area=0):
        from .grounded import clean_proposals
        with self.lock:
            labels = self.validate_prompt(prompt)
            bucket, tf, tm = self._text_features(labels)
            pixels = np.asarray(image.convert('RGB').resize((1008, 1008), Image.Resampling.BILINEAR), dtype=np.uint8)
            x = np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32)
            x *= np.float32(1 / 127.5)
            x -= np.float32(1)
            fpn = self.vision({'images': x[None]}, want=['fpn_2'], device_outputs=['fpn_2'])['fpn_2']
            out = self.heads[bucket]({'img_feat': fpn,
                'img_pos': self.positions[bucket], 'text_feats': tf, 'text_mask': tm}, want=['scores', 'boxes', 'presence'])
            if not all(np.isfinite(v).all() for v in out.values()):
                raise RuntimeError('Non-finite SAM3 outputs')
            sigmoid = lambda v: 1 / (1 + np.exp(-np.clip(v.astype(np.float32), -80, 80)))
            scores = sigmoid(out['scores'][..., 0]) * sigmoid(out['presence'][:, 0])[:, None]
            boxes, confidence, names = [], [], []
            for c, label in enumerate(labels):
                for q in np.flatnonzero(scores[c] >= box_threshold):
                    cx, cy, w, h = out['boxes'][c, q].astype(np.float32)
                    boxes.append((np.clip([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 0, 1) * [*image.size, *image.size]).tolist())
                    confidence.append(float(scores[c, q]))
                    names.append(label)
            return clean_proposals(boxes, confidence, names, image.size, nms_iou, limit,
                min_width=min_width, min_height=min_height, min_area=min_area)
