# nepali-conformer-offline — ONNX int8 conversion

This release is an **ONNX conversion** of Ampixa's
[**nepali-conformer-offline**](https://huggingface.co/ampixa/nepali-conformer-offline),
made so the model can run with ONNX Runtime alone (no NeMo or PyTorch), for example in the
free, open-source [Just Talk](https://github.com/sahasbelbase/JustTalk) voice keyboard.

**All credit for the model goes to Ampixa.** Original model, benchmark and full results:
[github.com/Ampixa/nepaliconformer](https://github.com/Ampixa/nepaliconformer).

## License

- **Model weights (these files): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**,
  as released by Ampixa. Non-commercial use only. You must give appropriate credit, link to the
  license, and indicate changes. See `LICENSE-model-CC-BY-NC-4.0.txt`.
- Conversion code (`export/`): MIT, same as the original repository. See `LICENSE-code.txt`.
- NepTel references: CC BY 4.0 (audio © InfoBayAI, CC BY 4.0).

## Changes from the original

Converted by Sahas Belbase for Just Talk:

1. Exported from the `.nemo` checkpoint to ONNX with NeMo's exporter: the encoder and the TDT
   decoder/joint network (`encoder.int8.onnx`, `decoder_joint.int8.onnx`), plus the CTC head
   (`ctc.int8.onnx`) for comparison.
2. Weights dynamically quantized to int8 with ONNX Runtime.
3. Feature settings (`model_config.json`) and the SentencePiece vocabulary (`tokens.json`)
   extracted so decoding can be reimplemented without NeMo.
4. `reference.tar.gz` holds the original NeMo model's features and transcripts on six public
   clips, used to check that the converted model matches the original.

Quantization and the reimplemented feature extraction can change results slightly. For the
model's measured accuracy (36.3% WER on NepTel real-call audio) and its documented limitations,
see Ampixa's [RESULTS.md](https://github.com/Ampixa/nepaliconformer/blob/master/RESULTS.md).
Numbers for this conversion are reported separately and are not Ampixa's.

## Citation

If you use the model, cite the original repository, as Ampixa requests:

```bibtex
@misc{ampixa_nepaliconformer,
  title        = {NepaliConformer: Nepali speech recognition models and the NepTel benchmark},
  author       = {{Ampixa}},
  year         = {2026},
  howpublished = {\url{https://github.com/Ampixa/nepaliconformer}},
  note         = {Model: https://huggingface.co/ampixa/nepali-conformer-offline (CC BY-NC 4.0)}
}
```

A technical report from Ampixa is in preparation; cite it instead once it is published.
