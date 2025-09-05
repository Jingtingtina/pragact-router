.PHONY: test en zh cost tag

test:
	python -m pytest -q

en:
	PRAGACT_FAST_CUES=1 PRAGACT_FAST_TAU=0.30 \
		python -m scripts.eval_confusion --config configs/default_en.yaml --data data/en/probe_150.jsonl

zh:
	PRAGACT_FAST_CUES=1 PRAGACT_FAST_TAU=0.20 \
		python -m scripts.eval_confusion --config configs/default_zh.yaml --data data/zh/probe_150.jsonl

cost:
	PRAGACT_FAST_CUES=1 PRAGACT_FAST_TAU=0.30 python -m scripts.eval_cost --config configs/default_en.yaml --data data/en/probe_150.jsonl
	PRAGACT_FAST_CUES=1 PRAGACT_FAST_TAU=0.20 python -m scripts.eval_cost --config configs/default_zh.yaml --data data/zh/probe_150.jsonl

tag:
	git tag -a v0.2.3-fastcue -m "fast-cue + router gate; EN τ=0.30, ZH τ=0.20"
	git push --tags
