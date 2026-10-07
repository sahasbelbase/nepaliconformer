"""Export nepali-conformer-offline (.nemo) to ONNX for runtimes without NeMo/PyTorch.

Produces, in --out:
  encoder.int8.onnx, decoder_joint.int8.onnx   TDT/RNNT path (the decoder the published WER uses)
  ctc.int8.onnx                                 CTC head, kept for comparison
  tokens.json                                   SentencePiece pieces, index = token id (blank = len)
  model_config.json                             preprocessor + decoding config from the checkpoint
  onnx_io.json                                  input/output names and shapes of every exported graph
  reference/<clip>.audio.npy                    16 kHz float32 input exactly as NeMo loaded it
  reference/<clip>.feats.npy                    NeMo log-mel features for that input (dither off)
  reference/transcripts.json                    NeMo TDT and CTC transcripts per clip

The reference files let a NumPy/ONNX Runtime port prove it matches the original model.

Usage:
    python export/export_onnx.py --nemo nepali_conformer_offline.nemo --out dist clip1.wav clip2.wav
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _io_signature(path: Path) -> dict:
    import onnx

    m = onnx.load(str(path), load_external_data=False)

    def dims(t):
        return [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]

    return {
        "inputs": [{"name": i.name, "shape": dims(i)} for i in m.graph.input],
        "outputs": [{"name": o.name, "shape": dims(o)} for o in m.graph.output],
    }


def _quantize(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("clips", nargs="*", help="audio files for reference outputs")
    a = ap.parse_args()

    import numpy as np
    import torch
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
    from omegaconf import OmegaConf

    out = Path(a.out)
    raw = out / "fp32"
    ref = out / "reference"
    for d in (out, raw, ref):
        d.mkdir(parents=True, exist_ok=True)

    model = EncDecHybridRNNTCTCBPEModel.restore_from(a.nemo, map_location="cpu")
    model.eval()
    # Deterministic features: no dither, no padding to a multiple
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    # ── Config and vocabulary ────────────────────────────────────────────
    cfg = OmegaConf.to_container(model.cfg, resolve=True)
    keep = {k: cfg.get(k) for k in ("preprocessor", "decoding", "model_defaults", "joint", "decoder", "sample_rate")}
    (out / "model_config.json").write_text(json.dumps(keep, indent=2, ensure_ascii=False, default=str))

    vocab_size = model.tokenizer.vocab_size
    tokens = [model.tokenizer.ids_to_tokens([i])[0] for i in range(vocab_size)]
    (out / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False))
    print(f"[export] vocab={vocab_size} blank_id={vocab_size}")

    # ── ONNX export ───────────────────────────────────────────────────────
    model.set_export_config({"decoder_type": "rnnt"})
    model.export(str(raw / "rnnt.onnx"))
    model.set_export_config({"decoder_type": "ctc"})
    model.export(str(raw / "ctc.onnx"))
    exported = sorted(p.name for p in raw.glob("*.onnx"))
    print(f"[export] exported: {exported}")

    # NeMo names the RNNT pair encoder-rnnt.onnx / decoder_joint-rnnt.onnx
    mapping = {}
    for p in raw.glob("*.onnx"):
        if p.name.startswith("encoder"):
            mapping[p] = "encoder"
        elif p.name.startswith("decoder_joint"):
            mapping[p] = "decoder_joint"
        elif p.name.startswith("ctc"):
            mapping[p] = "ctc"

    signatures = {}
    for src, name in mapping.items():
        signatures[name] = _io_signature(src)
        dst = out / f"{name}.int8.onnx"
        _quantize(src, dst)
        print(f"[export] {name}: {src.stat().st_size >> 20} MB fp32 -> {dst.stat().st_size >> 20} MB int8")
    (out / "onnx_io.json").write_text(json.dumps(signatures, indent=2))

    # ── Reference outputs from the original model ─────────────────────────
    transcripts = {}
    for clip in a.clips:
        stem = Path(clip).stem
        seg = AudioSegment.from_file(clip, target_sr=16000)
        audio = np.asarray(seg.samples, dtype=np.float32)
        np.save(ref / f"{stem}.audio.npy", audio)
        with torch.no_grad():
            feats, _ = model.preprocessor(
                input_signal=torch.from_numpy(audio)[None], length=torch.tensor([len(audio)])
            )
        np.save(ref / f"{stem}.feats.npy", feats[0].numpy().astype(np.float32))
        transcripts[stem] = {}

    for decoder in ("rnnt", "ctc"):
        model.change_decoding_strategy(decoder_type=decoder, verbose=False)
        outs = model.transcribe(list(a.clips), batch_size=1, verbose=False) if a.clips else []
        if isinstance(outs, tuple):  # older NeMo returns (best, all)
            outs = outs[0]
        for clip, o in zip(a.clips, outs):
            text = o.text if hasattr(o, "text") else str(o)
            transcripts[Path(clip).stem][decoder] = " ".join(t for t in text.split() if t != "<breath>")

    (ref / "transcripts.json").write_text(json.dumps(transcripts, indent=2, ensure_ascii=False))
    print(json.dumps(transcripts, indent=2, ensure_ascii=False))

    shutil.rmtree(raw)  # only the int8 graphs are published
    return 0


if __name__ == "__main__":
    sys.exit(main())
