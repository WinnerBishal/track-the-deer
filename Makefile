# -------- variable definitions --------
PY      := python                     # Which interpreter to use for every recipe
CFG     ?= cfg/run_example.yaml       # Default config file; CLI can override:  make train CFG=...
HPARAMS ?=                            # Optional hyperparameters file
RUN     ?=                            # Placeholder for a run-folder name used by eval/bench/export
DEVICE  ?= gpu                        # Default device for benchmarking (gpu / cpu / auto)

# Tell Make these “targets” are not real files; always run the recipe
.PHONY: train eval bench export

# --------- target: train --------------
train:                                # Usage:  make train CFG=cfg/my_run.yaml
	$(PY) -m src.training.train --cfg $(CFG)

# --------- target: eval ---------------
eval:                                 # Usage:  make eval RUN=2025-07-08_my_run
	$(PY) -m src.evaluation.metrics --run $(RUN)

# --------- target: bench --------------
bench:                                # Usage:  make bench RUN=... DEVICE=cpu
	$(PY) -m src.evaluation.runtime_bench --run $(RUN) --device $(DEVICE)

# --------- target: export -------------
export:                               # Usage:  make export RUN=... FORMAT=onnx
	$(PY) -m src.training.export_model --run $(RUN) --format onnx
