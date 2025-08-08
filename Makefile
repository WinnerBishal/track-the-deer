# -------- variable definitions --------
PY      := python                     # Which interpreter to use for every recipe
CFG     ?= cfg/run_example.yaml       # Default config file; CLI can override:  make train CFG=...
HPARAMS ?=                            # Optional hyperparameters file
RUN     ?=                            # Placeholder for a run-folder name used by eval/bench/export

# Tell Make these “targets” are not real files; always run the recipe
.PHONY: train

# --------- target: train --------------
train:                                # Usage:  make train CFG=cfg/my_run.yaml
	$(PY) -m src.training.train --cfg $(CFG)

eval2export:
	$(PY) notebooks/eval2export/evaluate.py
	$(PY) notebooks/eval2export/pt2onnx.py
	$(PY) notebooks/eval2export/move_rename_onnx.py

export_onnx:
	$(PY) notebooks/eval2export/pt2onnx.py
	$(PY) notebooks/eval2export/move_rename_onnx.py