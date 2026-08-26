"""ColBERT / late-interaction reranker 평가 (MTEB 2.x, 정답 후보 항상 포함).

evaluate_reranker.py 와 같은 harness(gold-injection, 공식 9개 kMTEB task, top-50 후보,
subset 평균 집계)를 쓰되, 모델만 late-interaction 으로 바꾼 독립 스크립트다.

Cross-encoder 는 (q, d) 를 한 번에 넣어 점수를 내지만, ColBERT 는 query/document 를 각각
토큰별 128-d 벡터로 encode 한 뒤 MaxSim(sum_i max_j q_i·d_j) 으로 채점한다. PLAID/fast-plaid
없이 encode + MaxSim 만 필요하므로 pylate 만 의존한다.

  uv run --extra colbert python eval/evaluate_colbert.py \
      --model_names nlpai-lab/KURE-v2 --gpu_id 0 --batch_size 32
"""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch

import mteb
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.models.model_meta import ModelMeta

V2_ROOT = Path(__file__).resolve().parents[1]
POOL_DIR = V2_ROOT / "eval/results/stage1/top_1k_qrels"
OUT_DIR = V2_ROOT / "eval/results/stage2"
NEG_TOP_K = 50  # 정답 외 negative 후보로 사용할 BM25 상위 개수 (evaluate_reranker 와 동일)
MAX_LENGTH = 8192  # document_length 상한 (전 모델 공통, 공정 비교)

DEFAULT_TASKS = [
    "Ko-StrategyQA",
    "AutoRAGRetrieval",
    "PublicHealthQA",
    "BelebeleRetrieval",
    "MIRACLRetrieval",
    "MrTidyRetrieval",
    "MultiLongDocRetrieval",
    "SQuADKorV1Retrieval",
    "LawIRKo",
]
DEFAULT_MODEL_NAMES = [
    "nlpai-lab/KURE-v2",
    "nlpai-lab/KURE-v2-unsupervised",
]
KOR_LANG = ["kor-Hang"]


# ============================================================================
# gold-injection monkey-patch (정답 항상 후보 포함) — evaluate_reranker.py 와 동일
# ============================================================================
_pool_cache: dict[str, dict[str, list[str]]] = {}


def _load_pool(task_name: str) -> dict[str, list[str]]:
    if task_name in _pool_cache:
        return _pool_cache[task_name]
    path = POOL_DIR / f"{task_name}_id.jsonl"
    pool: dict[str, list[str]] = {}
    if path.exists():
        for line in open(path):
            d = json.loads(line)
            pool[str(d["query_id"])] = [str(x) for x in d["relevance_ids"]]
    else:
        print(f"[gold-inject] WARNING: pool file 없음: {path}")
    _pool_cache[task_name] = pool
    return pool


def _ids(container) -> set:
    if hasattr(container, "keys"):
        return {str(k) for k in container.keys()}
    return {str(r.get("id") if hasattr(r, "get") else r) for r in container}


_orig_evaluate_subset = AbsTaskRetrieval._evaluate_subset


def _gold_inject_evaluate_subset(self, model, data_split, *, hf_split, hf_subset, **kwargs):
    """data_split['top_ranked'] 를 {정답 전부}∪{BM25 top-NEG_TOP_K} 로 교정."""
    pool = _load_pool(self.metadata.name)
    corpus = data_split["corpus"]
    queries = data_split["queries"]
    rels = data_split["relevant_docs"]

    cids = _ids(corpus)
    if hasattr(queries, "keys"):
        qids = [str(q) for q in queries.keys()]
    else:
        qids = [str(r.get("id") if hasattr(r, "get") else r) for r in queries]

    tr: dict[str, list[str]] = {}
    n_noneg = 0
    for qid in qids:
        gr = rels.get(qid, {}) if hasattr(rels, "get") else {}
        golds = [g for g in gr if g in cids]
        gset = set(golds)
        bm = [d for d in pool.get(qid, [])[:NEG_TOP_K] if d not in gset and d in cids]
        tr[qid] = golds + bm
        if not bm:
            n_noneg += 1
    data_split["top_ranked"] = tr
    self._top_k = max((len(v) for v in tr.values()), default=1) + 1
    msg = f"[gold-inject] {self.metadata.name}[{hf_subset}/{hf_split}] q={len(tr)} top_k={self._top_k}"
    if n_noneg:
        msg += f" (no-BM25-neg q={n_noneg})"
    print(msg, flush=True)
    return _orig_evaluate_subset(self, model, data_split, hf_split=hf_split, hf_subset=hf_subset, **kwargs)


AbsTaskRetrieval._evaluate_subset = _gold_inject_evaluate_subset


# ============================================================================
# ColBERT reranker — mteb CrossEncoderProtocol(predict(inputs1, inputs2, ...)) 직접 구현
# ============================================================================
class ColBERTReranker:
    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 16,
                 document_length: int = MAX_LENGTH):
        from pylate import models

        self.model = models.ColBERT(
            model_name_or_path=model_name, device=device, document_length=document_length
        )
        self.model.eval()
        self.batch_size = batch_size
        self.eval_max_length = document_length
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(name=model_name, revision="gold-inject", loader=None)
        )

    @staticmethod
    def _texts(loader) -> list[str]:
        return [t for batch in loader for t in batch["text"]]

    @torch.no_grad()
    def predict(self, inputs1, inputs2, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        queries = self._texts(inputs1)
        docs = self._texts(inputs2)
        # 같은 query 가 후보 수만큼 반복되므로 dedup 후 한 번만 encode.
        uq = list(dict.fromkeys(queries))
        ud = list(dict.fromkeys(docs))
        q_emb = self.model.encode(uq, is_query=True, batch_size=self.batch_size,
                                  convert_to_tensor=True, show_progress_bar=False)
        d_emb = self.model.encode(ud, is_query=False, batch_size=self.batch_size,
                                  convert_to_tensor=True, show_progress_bar=False)
        qmap = dict(zip(uq, q_emb))
        dmap = dict(zip(ud, d_emb))
        scores = [float((qmap[q] @ dmap[d].T).max(dim=1).values.sum())  # MaxSim
                  for q, d in zip(queries, docs)]
        return np.asarray(scores, dtype=np.float32)


# ============================================================================
# task 로딩 + 점수 추출 + 드라이버 — evaluate_reranker.py 와 동일
# ============================================================================
def get_task(task_name: str):
    try:
        ts = mteb.get_tasks(tasks=[task_name], languages=KOR_LANG)
        if ts:
            return ts[0]
    except Exception:
        pass
    ts = mteb.get_tasks(tasks=[task_name])
    return ts[0] if ts else None


def extract_scores(res) -> dict:
    metrics = ["ndcg_at_1", "ndcg_at_5", "ndcg_at_10", "map_at_10", "mrr_at_10", "recall_at_10"]
    found: dict[str, list[float]] = {m: [] for m in metrics}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in found and isinstance(v, (int, float)):
                    found[k].append(float(v))
                walk(v)
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        else:
            for attr in ("scores", "task_results", "results"):
                if hasattr(o, attr):
                    try:
                        walk(getattr(o, attr))
                    except Exception:
                        pass

    walk(res)
    out = {}
    for m, vals in found.items():
        if vals:
            out[m] = sum(vals) / len(vals)
            out[m + "__per_subset"] = vals
    return out


def eval_one_model(model_name: str, tasks: list[str], device: str, overwrite: bool, batch_size: int):
    model = ColBERTReranker(model_name, device=device, batch_size=batch_size, document_length=MAX_LENGTH)
    out_model_dir = OUT_DIR / model_name
    out_model_dir.mkdir(parents=True, exist_ok=True)
    for task_name in tasks:
        out_path = out_model_dir / f"{task_name}.json"
        if out_path.exists() and not overwrite:
            print(f"[skip] {model_name} / {task_name} (이미 있음)", flush=True)
            continue
        try:
            t = get_task(task_name)
            if t is None:
                print(f"[error] task 없음: {task_name}", flush=True)
                continue
            print(f"\n=== {model_name} × {task_name} ===", flush=True)
            res = mteb.evaluate(model, t, cache=None, overwrite_strategy="always",
                                show_progress_bar=False, encode_kwargs={"batch_size": batch_size})
            raw = extract_scores(res)
            scores = {k: v for k, v in raw.items() if not k.endswith("__per_subset")}
            scores["main_score"] = scores.get("ndcg_at_10")
            scores["_model"] = model_name
            scores["_task"] = task_name
            scores["_split"] = "+".join(t.metadata.eval_splits)
            scores["_max_length"] = getattr(model, "eval_max_length", MAX_LENGTH)
            scores["_neg_top_k"] = NEG_TOP_K
            json.dump(scores, open(out_path, "w"), indent=2)
            print(f"[done] {task_name}: NDCG@10={scores.get('ndcg_at_10')}", flush=True)
        except Exception as ex:
            print(f"[error] {model_name} / {task_name}: {ex}", flush=True)
            traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(description="MTEB 2.x ColBERT(late-interaction) 평가 (정답 후보 항상 포함)")
    ap.add_argument("--model_names", nargs="+", default=DEFAULT_MODEL_NAMES)
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | models={len(args.model_names)} tasks={len(args.tasks)}", flush=True)
    for model_name in args.model_names:
        try:
            eval_one_model(model_name, args.tasks, device, args.overwrite, args.batch_size)
        except Exception as ex:
            print(f"[error] model {model_name}: {ex}", flush=True)
            traceback.print_exc()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
