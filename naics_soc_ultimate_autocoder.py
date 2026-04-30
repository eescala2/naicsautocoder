#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAICS-first + SOC ultimate autocoder for SPSS .sav files
============================================================

Pipeline order
--------------
1) Train/predict NAICS first, default target qa20a.
2) Fill/update NAICS in the scoring data.
3) Train/predict SOC qe4ar and qe5ar using the updated NAICS as a feature.
4) Save/load a complete model bundle for future scoring.

Design goals
------------
* Accuracy-first, but safe on a laptop-class workstation.
* Robust fallback if optional packages are not installed.
* Sparse lexical NLP + categorical + numeric + optional semantic embeddings.
* Exact and optional fuzzy dictionary overrides for high-purity repeated answers.
* Hierarchical SOC: major group first, then SOC6 within each major group.
* Ensemble/model selection across multiple families, with optional Optuna tuning.
* SPSS 26-compatible usage via an external modern Python executable.

Recommended full install
------------------------
python -m pip install -r requirements_naics_soc_state_of_art.txt

Example train + score
---------------------
python naics_soc_state_of_art_autocoder.py ^
  --mode train_and_score ^
  --train_sav "C:\\laborshed_autocode\\laborshed_train.sav" ^
  --score_sav "C:\\laborshed_autocode\\laborshed_to_score.sav" ^
  --out_sav   "C:\\laborshed_autocode\\laborshed_scored.sav" ^
  --log_txt   "C:\\laborshed_autocode\\naics_soc_log.txt" ^
  --model_out "C:\\laborshed_autocode\\naics_soc_state_of_art.joblib" ^
  --save_model ^
  --profile accuracy

Example score only
------------------
python naics_soc_state_of_art_autocoder.py ^
  --mode score ^
  --model_in  "C:\\laborshed_autocode\\naics_soc_state_of_art.joblib" ^
  --score_sav "C:\\laborshed_autocode\\new_to_score.sav" ^
  --out_sav   "C:\\laborshed_autocode\\new_scored.sav" ^
  --log_txt   "C:\\laborshed_autocode\\new_naics_soc_log.txt"
"""

from __future__ import annotations

import os

# These must be set before importing NumPy/SciPy/sklearn to avoid nested
# thread storms on Windows laptops. CLI --native_threads overrides the
# actual threadpoolctl context later, but these protect import-time defaults.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import json
import math
import platform
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import pyreadstat
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

# Optional packages. The script uses them when present and safely skips them
# when absent.
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:
    lgb = None
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    xgb = None
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None
    HAS_CATBOOST = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    optuna = None
    HAS_OPTUNA = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except Exception:
    fuzz = None
    rf_process = None
    HAS_RAPIDFUZZ = False

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

try:
    from llama_cpp import Llama  # type: ignore
    HAS_LLAMA_CPP = True
except Exception:
    Llama = None
    HAS_LLAMA_CPP = False

try:
    import psutil  # type: ignore
    HAS_PSUTIL = True
except Exception:
    psutil = None
    HAS_PSUTIL = False

VERSION = "2026.04.30.ultimate-naics-first-soc-semantic-llm"

VALID_SOC_MAJOR_GROUPS = {
    11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33,
    35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55,
}

COMMON_CODE_MISSINGS = {"", "0", "00", "000", "0000", "00000", "000000", "999", "9999", "99999", "999999"}

ABBREV_MAP = [
    (r"\basst\b", "assistant"),
    (r"\bmgr\b", "manager"),
    (r"\bmgmt\b", "management"),
    (r"\bsvcs\b", "services"),
    (r"\bsvc\b", "service"),
    (r"\btech\b", "technician"),
    (r"\badmin\b", "administrative"),
    (r"\bdept\b", "department"),
    (r"\bsupv\b", "supervisor"),
    (r"\bsupvr\b", "supervisor"),
    (r"\bmaint\b", "maintenance"),
    (r"\bdir\b", "director"),
    (r"\bexec\b", "executive"),
    (r"\bceo\b", "chief executive officer"),
    (r"\bcfo\b", "chief financial officer"),
    (r"\bcoo\b", "chief operating officer"),
    (r"\brn\b", "registered nurse"),
    (r"\blpn\b", "licensed practical nurse"),
    (r"\blvn\b", "licensed vocational nurse"),
    (r"\bcna\b", "certified nursing assistant"),
    (r"\bemt\b", "emergency medical technician"),
    (r"\bcdl\b", "commercial drivers license"),
    (r"\ba\+\b", "a plus"),
]

NON_ALNUM_RE = re.compile(r"[^a-z0-9\+\#\/\-\s]+")
SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------

@dataclass
class ParallelConfig:
    max_workers: int
    native_threads: int


@dataclass
class TrainConfig:
    profile: str
    random_seed: int

    # Dictionary behavior
    dict_purity: float
    dict_min_count: int
    naics_dict_purity: float
    naics_dict_min_count: int
    use_fuzzy_dictionary: bool
    fuzzy_threshold: int
    fuzzy_max_choices_per_bucket: int

    # Semantic vector memory / dense retrieval behavior
    use_semantic_memory: bool
    semantic_backend: str
    semantic_k: int
    semantic_min_similarity: float
    semantic_min_confidence: float
    semantic_fusion_weight: float
    ml_fusion_weight: float
    semantic_reference_weight: float
    naics_reference_csv: str
    soc_reference_csv: str

    # Local air-gapped LLM fallback / reranker
    use_local_llm_fallback: bool
    llm_model_path: str
    llm_trigger_confidence_naics: float
    llm_trigger_confidence_soc: float
    llm_max_rows: int
    llm_threads: int
    llm_ctx_size: int
    llm_temperature: float
    llm_top_p: float
    llm_max_tokens: int
    llm_rerank_confidence: float
    llm_allow_free_code: bool

    # Top-k output
    topk_naics: int
    topk_soc: int
    top_groups: int

    # Rare class thresholds
    min_naics_n: int
    min_soc_n: int

    # CV and search
    nfolds_naics: int
    nfolds_major: int
    nfolds_within: int
    model_search: str
    ensemble_size: int
    ensemble_power: float
    optuna_trials_naics: int
    optuna_trials_major: int
    optuna_trials_within: int
    optuna_timeout_seconds: Optional[int]

    # Feature parameters
    max_word_terms: int
    max_char_terms: int
    word_min_count: int
    char_min_count: int
    doc_prop_max: float
    use_embeddings: bool
    embedding_model: str
    embedding_backend: str
    embedding_batch_size: int
    embedding_device: Optional[str]
    svd_components: int

    # Base model parameters
    logreg_C: float
    logreg_max_iter: int
    sgd_alpha: float
    sgd_max_iter: int
    sgd_tol: float
    linearsvc_C: float
    boost_n_estimators: int
    boost_max_classes: int
    boost_max_features: int

    # Scoring/write behavior
    score_batch_size: int
    min_write_confidence_naics: float
    min_write_confidence_soc: float
    missing_soc_values: Tuple[int, ...]
    missing_naics_values: Tuple[str, ...]


@dataclass
class CandidateScore:
    name: str
    macro_f1: Optional[float]
    log_loss_value: Optional[float]
    failed: bool
    reason: str = ""


@dataclass
class FittedClassifier:
    stage_name: str
    model_names: List[str]
    estimators: List[Any]
    weights: List[float]
    classes: np.ndarray
    cv_scores: List[CandidateScore]
    n_rows: int
    n_features: int
    n_classes: int

    @property
    def model_name(self) -> str:
        return "+".join(self.model_names)

    @property
    def cv_macro_f1(self) -> Optional[float]:
        vals = [s.macro_f1 for s in self.cv_scores if s.name in self.model_names and s.macro_f1 is not None]
        if not vals:
            return None
        return float(np.average(vals, weights=self.weights[:len(vals)] if len(vals) == len(self.weights) else None))

    @property
    def cv_log_loss(self) -> Optional[float]:
        vals = [s.log_loss_value for s in self.cv_scores if s.name in self.model_names and s.log_loss_value is not None and np.isfinite(s.log_loss_value)]
        if not vals:
            return None
        return float(np.mean(vals))

    def predict_proba(self, x: sp.spmatrix) -> np.ndarray:
        out = np.zeros((x.shape[0], len(self.classes)), dtype=np.float64)
        total_w = 0.0
        for est, w in zip(self.estimators, self.weights):
            p = estimator_predict_proba(est, x, self.classes)
            out += float(w) * p
            total_w += float(w)
        if total_w <= 0:
            total_w = 1.0
        out /= total_w
        row_sum = out.sum(axis=1)
        bad = row_sum <= 0
        if np.any(bad):
            out[bad, :] = 1.0 / max(1, out.shape[1])
            row_sum = out.sum(axis=1)
        return out / row_sum[:, None]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def eprint(msg: str) -> None:
    print(msg, flush=True)


def parse_csv_ints(s: Optional[str]) -> Tuple[int, ...]:
    if s is None or str(s).strip() == "":
        return tuple()
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except Exception:
            pass
    return tuple(out)


def parse_csv_strs(s: Optional[str]) -> Tuple[str, ...]:
    if s is None or str(s).strip() == "":
        return tuple()
    return tuple(p.strip() for p in str(s).split(",") if p.strip())


def recommend_max_workers(raw_value: Optional[str]) -> int:
    logical = os.cpu_count() or 1
    raw = "auto" if raw_value is None else str(raw_value).strip().lower()
    if raw == "auto":
        # Designed for the user's Latitude 5550 class machine: 18 logical processors.
        # Keep room for Windows/SPSS/disk cache and avoid nested native thread storms.
        if logical >= 18:
            return 6
        if logical >= 14:
            return 5
        if logical >= 10:
            return 4
        if logical >= 6:
            return 3
        return 1
    val = int(raw)
    return max(1, min(val, logical))


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = str(value).strip().lower()
    if not s:
        return ""
    s = NON_ALNUM_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    for pattern, repl in ABBREV_MAP:
        s = re.sub(pattern, repl, s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def clean_code_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        if abs(value - round(value)) < 1e-8:
            return str(int(round(value)))
        return str(value).strip()
    s = str(value).strip()
    if not s:
        return None
    # SPSS numeric values sometimes arrive as "123.0" after conversions.
    if re.fullmatch(r"\d+\.0+", s):
        return str(int(float(s)))
    return s


def clean_code_series(s: pd.Series) -> pd.Series:
    return s.map(clean_code_value).astype("object")


def cat_to_string_series(s: pd.Series) -> pd.Series:
    return clean_code_series(s)


def soc6_to_major2(soc6: Any) -> Optional[int]:
    try:
        x = int(round(float(soc6)))
    except Exception:
        return None
    if x < 100000 or x > 999999:
        return None
    return int(x // 10000)


def format_soc(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "MISSING"
    except Exception:
        pass
    try:
        x = int(round(float(value)))
        if 0 <= x <= 999999:
            return f"{x:06d}"
    except Exception:
        pass
    return str(value)


def format_code(value: Any) -> str:
    cv = clean_code_value(value)
    return cv if cv is not None else "MISSING"


def is_missing_soc_series(s: pd.Series, missing_soc_values: Sequence[int]) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    miss = x.isna() | (x < 100000) | (x > 999999)
    xi = x.fillna(-1).round().astype(np.int64)
    major = xi // 10000
    miss = miss | (~major.isin(list(VALID_SOC_MAJOR_GROUPS)))
    if missing_soc_values:
        miss = miss | x.isin(list(missing_soc_values))
    return miss


def is_missing_code_series(s: pd.Series, missing_values: Sequence[str]) -> pd.Series:
    codes = clean_code_series(s)
    miss = codes.isna() | codes.map(lambda v: str(v).strip() in COMMON_CODE_MISSINGS if v is not None else True)
    if missing_values:
        miss = miss | codes.isin(list(missing_values))
    return miss


def sparse_nbytes(x: sp.spmatrix) -> int:
    x = x.tocsr()
    return int(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)


def safe_top_indices(row: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), row.size))
    if row.size <= k:
        return np.argsort(-row)
    idx = np.argpartition(-row, kth=k - 1)[:k]
    return idx[np.argsort(-row[idx])]


def batched_indices(idxs: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(idxs), batch_size):
        yield idxs[start:start + batch_size]


def filter_meta_dict(d: Optional[Dict[str, Any]], columns: Sequence[str]) -> Optional[Dict[str, Any]]:
    if not d:
        return None
    out = {k: v for k, v in d.items() if k in columns}
    return out or None


def has_any_normalized_text(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    present = np.zeros(len(df), dtype=bool)
    for c in cols:
        if c not in df.columns:
            continue
        present |= df[c].map(lambda v: len(normalize_text(v)) > 0).to_numpy(dtype=bool)
    return present


# ---------------------------------------------------------------------
# Optional estimator wrappers
# ---------------------------------------------------------------------

class XGBLabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        n_jobs: int = 1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: sp.spmatrix, y: np.ndarray) -> "XGBLabelEncodedClassifier":
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost is not installed")
        self.encoder_ = LabelEncoder()
        y_enc = self.encoder_.fit_transform(y)
        self.classes_ = self.encoder_.classes_
        self.model_ = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(self.classes_),
            eval_metric="mlogloss",
            tree_method="hist",
            n_estimators=int(self.n_estimators),
            max_depth=int(self.max_depth),
            learning_rate=float(self.learning_rate),
            subsample=float(self.subsample),
            colsample_bytree=float(self.colsample_bytree),
            reg_lambda=float(self.reg_lambda),
            reg_alpha=float(self.reg_alpha),
            n_jobs=int(self.n_jobs),
            random_state=int(self.random_state),
            verbosity=0,
        )
        self.model_.fit(X, y_enc)
        return self

    def predict_proba(self, X: sp.spmatrix) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(X), dtype=np.float64)

    def predict(self, X: sp.spmatrix) -> np.ndarray:
        enc = np.asarray(self.model_.predict(X), dtype=int)
        return self.encoder_.inverse_transform(enc)


class CatBoostLabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        thread_count: int = 1,
        random_state: int = 42,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.thread_count = thread_count
        self.random_state = random_state

    def fit(self, X: sp.spmatrix, y: np.ndarray) -> "CatBoostLabelEncodedClassifier":
        if not HAS_CATBOOST:
            raise RuntimeError("catboost is not installed")
        self.encoder_ = LabelEncoder()
        y_enc = self.encoder_.fit_transform(y)
        self.classes_ = self.encoder_.classes_
        self.model_ = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=int(self.iterations),
            depth=int(self.depth),
            learning_rate=float(self.learning_rate),
            l2_leaf_reg=float(self.l2_leaf_reg),
            thread_count=int(self.thread_count),
            random_seed=int(self.random_state),
            verbose=False,
            allow_writing_files=False,
        )
        self.model_.fit(X, y_enc)
        return self

    def predict_proba(self, X: sp.spmatrix) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(X), dtype=np.float64)

    def predict(self, X: sp.spmatrix) -> np.ndarray:
        enc = np.asarray(self.model_.predict(X)).reshape(-1).astype(int)
        return self.encoder_.inverse_transform(enc)


def estimator_predict_proba(estimator: Any, x: sp.spmatrix, expected_classes: np.ndarray) -> np.ndarray:
    expected_classes = np.asarray(expected_classes)
    if hasattr(estimator, "predict_proba"):
        p = np.asarray(estimator.predict_proba(x), dtype=np.float64)
        est_classes = np.asarray(getattr(estimator, "classes_", expected_classes))
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        if len(est_classes) == len(expected_classes) and np.array_equal(est_classes, expected_classes):
            row_sum = p.sum(axis=1)
            bad = row_sum <= 0
            if np.any(bad):
                p[bad, :] = 1.0 / max(1, p.shape[1])
                row_sum = p.sum(axis=1)
            return p / row_sum[:, None]
        out = np.zeros((x.shape[0], len(expected_classes)), dtype=np.float64)
        lookup = {c: i for i, c in enumerate(expected_classes.tolist())}
        for j, c in enumerate(est_classes.tolist()):
            if c in lookup and j < p.shape[1]:
                out[:, lookup[c]] = p[:, j]
        row_sum = out.sum(axis=1)
        bad = row_sum <= 0
        if np.any(bad):
            out[bad, :] = 1.0 / max(1, out.shape[1])
            row_sum = out.sum(axis=1)
        return out / row_sum[:, None]

    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(x), dtype=np.float64)
        if scores.ndim == 1:
            p1 = expit(scores)
            p = np.vstack([1.0 - p1, p1]).T
        else:
            p = softmax(scores, axis=1)
        est_classes = np.asarray(getattr(estimator, "classes_", expected_classes))
        out = np.zeros((x.shape[0], len(expected_classes)), dtype=np.float64)
        lookup = {c: i for i, c in enumerate(expected_classes.tolist())}
        for j, c in enumerate(est_classes.tolist()):
            if c in lookup and j < p.shape[1]:
                out[:, lookup[c]] = p[:, j]
        row_sum = out.sum(axis=1)
        bad = row_sum <= 0
        if np.any(bad):
            out[bad, :] = 1.0 / max(1, out.shape[1])
            row_sum = out.sum(axis=1)
        return out / row_sum[:, None]

    pred = estimator.predict(x)
    out = np.zeros((x.shape[0], len(expected_classes)), dtype=np.float64)
    lookup = {c: i for i, c in enumerate(expected_classes.tolist())}
    for i, c in enumerate(pred.tolist()):
        if c in lookup:
            out[i, lookup[c]] = 1.0
        else:
            out[i, :] = 1.0 / max(1, len(expected_classes))
    return out


# ---------------------------------------------------------------------
# Feature creation
# ---------------------------------------------------------------------

class HybridFeaturizer:
    def __init__(
        self,
        text_cols: Sequence[str],
        cat_cols: Sequence[str],
        num_cols: Sequence[str],
        prefixes: Dict[str, str],
        cfg: TrainConfig,
    ) -> None:
        self.text_cols = list(text_cols)
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.prefixes = dict(prefixes)
        self.cfg_snapshot = {
            "max_word_terms": int(cfg.max_word_terms),
            "max_char_terms": int(cfg.max_char_terms),
            "word_min_count": int(cfg.word_min_count),
            "char_min_count": int(cfg.char_min_count),
            "doc_prop_max": float(cfg.doc_prop_max),
            "use_embeddings": bool(cfg.use_embeddings),
            "embedding_model": str(cfg.embedding_model),
            "embedding_backend": str(cfg.embedding_backend),
            "embedding_batch_size": int(cfg.embedding_batch_size),
            "embedding_device": cfg.embedding_device,
            "svd_components": int(cfg.svd_components),
            "random_seed": int(cfg.random_seed),
        }

        self.word_vectorizer: Optional[TfidfVectorizer] = None
        self.char_vectorizer: Optional[TfidfVectorizer] = None
        self.cat_encoder: Optional[OneHotEncoder] = None
        self.num_imputer: Optional[SimpleImputer] = None
        self.num_scaler: Optional[StandardScaler] = None
        self.svd: Optional[TruncatedSVD] = None
        self._sentence_model: Any = None
        self.features_are_nonnegative_: bool = True

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        # Do not pickle the transformer model object. It is large and less stable
        # across machines. The model name/path is saved and lazily reloaded.
        state["_sentence_model"] = None
        return state

    def _make_docs(self, df: pd.DataFrame) -> List[str]:
        docs: List[str] = []
        for _, row in df.iterrows():
            parts: List[str] = []
            for c in self.text_cols:
                if c not in df.columns:
                    continue
                v = normalize_text(row.get(c))
                if not v:
                    continue
                prefix = self.prefixes.get(c, "")
                parts.append(f"{prefix} {v}".strip())
            doc = " | ".join(parts).strip()
            docs.append(doc if doc else "__EMPTY__")
        return docs

    def _embed_docs(self, docs: List[str], fit: bool = False) -> sp.csr_matrix:
        if not self.cfg_snapshot["use_embeddings"]:
            return sp.csr_matrix((len(docs), 0), dtype=np.float32)
        if not HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("--use_embeddings was requested, but sentence-transformers is not installed")
        if self._sentence_model is None:
            kwargs: Dict[str, Any] = {}
            device = self.cfg_snapshot.get("embedding_device")
            if device:
                kwargs["device"] = device
            backend = str(self.cfg_snapshot.get("embedding_backend", "torch")).lower()
            if backend in {"onnx", "openvino"}:
                try:
                    self._sentence_model = SentenceTransformer(self.cfg_snapshot["embedding_model"], backend=backend, **kwargs)
                except TypeError:
                    eprint(f"WARNING: sentence-transformers backend={backend} is not supported by this install; falling back to torch backend.")
                    self._sentence_model = SentenceTransformer(self.cfg_snapshot["embedding_model"], **kwargs)
                except Exception as exc:
                    eprint(f"WARNING: failed to load embedding backend={backend} ({exc}); falling back to torch backend.")
                    self._sentence_model = SentenceTransformer(self.cfg_snapshot["embedding_model"], **kwargs)
            else:
                self._sentence_model = SentenceTransformer(self.cfg_snapshot["embedding_model"], **kwargs)
        emb = self._sentence_model.encode(
            docs,
            batch_size=int(self.cfg_snapshot["embedding_batch_size"]),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)
        return sp.csr_matrix(emb, dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        df = df.copy()
        docs = self._make_docs(df)

        self.word_vectorizer = TfidfVectorizer(
            lowercase=False,
            ngram_range=(1, 2),
            min_df=int(self.cfg_snapshot["word_min_count"]),
            max_df=float(self.cfg_snapshot["doc_prop_max"]),
            max_features=int(self.cfg_snapshot["max_word_terms"]),
            sublinear_tf=True,
            dtype=np.float32,
        )
        x_word = self.word_vectorizer.fit_transform(docs).astype(np.float32)

        self.char_vectorizer = TfidfVectorizer(
            lowercase=False,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=int(self.cfg_snapshot["char_min_count"]),
            max_df=float(self.cfg_snapshot["doc_prop_max"]),
            max_features=int(self.cfg_snapshot["max_char_terms"]),
            sublinear_tf=True,
            dtype=np.float32,
        )
        x_char = self.char_vectorizer.fit_transform(docs).astype(np.float32)

        if self.cat_cols:
            cat_df = pd.DataFrame(index=df.index)
            for c in self.cat_cols:
                if c in df.columns:
                    cat_df[c] = cat_to_string_series(df[c]).fillna("MISSING").astype(str)
                else:
                    cat_df[c] = "MISSING"
            self.cat_encoder = make_one_hot_encoder()
            x_cat = self.cat_encoder.fit_transform(cat_df).astype(np.float32)
        else:
            x_cat = sp.csr_matrix((len(df), 0), dtype=np.float32)

        if self.num_cols:
            num_df = pd.DataFrame(index=df.index)
            for c in self.num_cols:
                if c in df.columns:
                    num_df[c] = pd.to_numeric(df[c], errors="coerce")
                else:
                    num_df[c] = np.nan
            self.num_imputer = SimpleImputer(strategy="median")
            self.num_scaler = StandardScaler(with_mean=True, with_std=True)
            arr = self.num_imputer.fit_transform(num_df)
            arr = self.num_scaler.fit_transform(arr).astype(np.float32)
            x_num = sp.csr_matrix(arr, dtype=np.float32)
        else:
            x_num = sp.csr_matrix((len(df), 0), dtype=np.float32)

        pieces: List[sp.spmatrix] = [x_word, x_char, x_cat, x_num]

        if int(self.cfg_snapshot["svd_components"]) > 0 and (x_word.shape[1] + x_char.shape[1]) > 2:
            x_text = sp.hstack([x_word, x_char], format="csr", dtype=np.float32)
            n_comp = min(int(self.cfg_snapshot["svd_components"]), max(1, min(x_text.shape) - 1))
            if n_comp > 0:
                self.svd = TruncatedSVD(n_components=n_comp, random_state=int(self.cfg_snapshot["random_seed"]))
                x_svd = self.svd.fit_transform(x_text).astype(np.float32)
                pieces.append(sp.csr_matrix(x_svd, dtype=np.float32))
                self.features_are_nonnegative_ = False

        x_emb = self._embed_docs(docs, fit=True)
        if x_emb.shape[1] > 0:
            pieces.append(x_emb)
            self.features_are_nonnegative_ = False

        return sp.hstack(pieces, format="csr", dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        if self.word_vectorizer is None or self.char_vectorizer is None:
            raise RuntimeError("Featurizer has not been fitted")
        df = df.copy()
        docs = self._make_docs(df)
        x_word = self.word_vectorizer.transform(docs).astype(np.float32)
        x_char = self.char_vectorizer.transform(docs).astype(np.float32)

        if self.cat_cols and self.cat_encoder is not None:
            cat_df = pd.DataFrame(index=df.index)
            for c in self.cat_cols:
                if c in df.columns:
                    cat_df[c] = cat_to_string_series(df[c]).fillna("MISSING").astype(str)
                else:
                    cat_df[c] = "MISSING"
            x_cat = self.cat_encoder.transform(cat_df).astype(np.float32)
        else:
            x_cat = sp.csr_matrix((len(df), 0), dtype=np.float32)

        if self.num_cols and self.num_imputer is not None and self.num_scaler is not None:
            num_df = pd.DataFrame(index=df.index)
            for c in self.num_cols:
                if c in df.columns:
                    num_df[c] = pd.to_numeric(df[c], errors="coerce")
                else:
                    num_df[c] = np.nan
            arr = self.num_imputer.transform(num_df)
            arr = self.num_scaler.transform(arr).astype(np.float32)
            x_num = sp.csr_matrix(arr, dtype=np.float32)
        else:
            x_num = sp.csr_matrix((len(df), 0), dtype=np.float32)

        pieces: List[sp.spmatrix] = [x_word, x_char, x_cat, x_num]
        if self.svd is not None:
            x_text = sp.hstack([x_word, x_char], format="csr", dtype=np.float32)
            x_svd = self.svd.transform(x_text).astype(np.float32)
            pieces.append(sp.csr_matrix(x_svd, dtype=np.float32))
        x_emb = self._embed_docs(docs, fit=False)
        if x_emb.shape[1] > 0:
            pieces.append(x_emb)
        return sp.hstack(pieces, format="csr", dtype=np.float32)


# ---------------------------------------------------------------------
# High-purity dictionaries
# ---------------------------------------------------------------------

def build_exact_dictionary(
    df: pd.DataFrame,
    target_values: Sequence[Any],
    key_specs: Sequence[Tuple[str, Sequence[str]]],
    min_count: int,
) -> Dict[str, Dict[str, Any]]:
    y = [clean_code_value(v) for v in target_values]
    out: Dict[str, Dict[str, Any]] = {}
    for spec_name, cols in key_specs:
        if not all(c in df.columns for c in cols):
            continue
        buckets: Dict[str, List[str]] = defaultdict(list)
        for pos, (_, row) in enumerate(df.iterrows()):
            parts: List[str] = []
            for c in cols:
                val = normalize_text(row.get(c)) if df[c].dtype == "object" else format_code(row.get(c))
                parts.append(val if val else "MISSING")
            key_body = "||".join(parts)
            if not key_body or key_body.replace("MISSING", "").replace("||", "") == "":
                continue
            target = y[pos]
            if target is not None and target not in COMMON_CODE_MISSINGS:
                buckets[f"{spec_name}::{key_body}"].append(target)
        for key, vals in buckets.items():
            if len(vals) < min_count:
                continue
            counts = Counter(vals)
            total = float(sum(counts.values()))
            ordered = counts.most_common()
            out[key] = {
                "top": ordered[0][0],
                "top_prob": float(ordered[0][1] / total),
                "alts": [{"code": code, "prob": float(n / total)} for code, n in ordered[1:6]],
                "n": int(total),
            }
    return out


def exact_dictionary_predict(
    df: pd.DataFrame,
    dictionary: Dict[str, Dict[str, Any]],
    key_specs: Sequence[Tuple[str, Sequence[str]]],
    purity: float,
    min_count: int,
    topk: int,
) -> Dict[str, Any]:
    n = len(df)
    pred: List[Optional[str]] = [None] * n
    conf = np.zeros(n, dtype=np.float32)
    used = np.zeros(n, dtype=bool)
    alts: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    if not dictionary:
        return {"pred": pred, "conf": conf, "used": used, "alts": alts}

    for pos, (_, row) in enumerate(df.iterrows()):
        for spec_name, cols in key_specs:
            if not all(c in df.columns for c in cols):
                continue
            parts: List[str] = []
            for c in cols:
                val = normalize_text(row.get(c)) if df[c].dtype == "object" else format_code(row.get(c))
                parts.append(val if val else "MISSING")
            key = f"{spec_name}::{'||'.join(parts)}"
            entry = dictionary.get(key)
            if entry and entry["n"] >= min_count and entry["top_prob"] >= purity:
                pred[pos] = str(entry["top"])
                conf[pos] = float(entry["top_prob"])
                used[pos] = True
                alts[pos] = list(entry.get("alts", []))[:max(0, topk - 1)]
                break
    return {"pred": pred, "conf": conf, "used": used, "alts": alts}


def build_soc_title_naics_dictionary(
    df: pd.DataFrame,
    title_col: str,
    naics_col: str,
    y_soc: Sequence[int],
    min_count: int,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    exact: Dict[str, Dict[str, Any]] = {}
    fuzzy_buckets: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    if title_col not in df.columns:
        return {"exact": exact, "fuzzy_buckets": {}}

    titles = df[title_col].map(normalize_text).tolist()
    if naics_col in df.columns:
        naics_vals = cat_to_string_series(df[naics_col]).fillna("MISSING").astype(str).tolist()
    else:
        naics_vals = ["MISSING"] * len(df)
    buckets: Dict[str, List[int]] = defaultdict(list)
    title_bucket_counts: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for t, nv, soc in zip(titles, naics_vals, y_soc):
        if not t:
            continue
        key = f"{t}||{nv}"
        buckets[key].append(int(soc))
        title_bucket_counts[(str(nv), t)].append(int(soc))
        # Also learn a title-only fallback for cases where NAICS is missing/new.
        buckets[f"{t}||MISSING"].append(int(soc))
        title_bucket_counts[("MISSING", t)].append(int(soc))

    for key, vals in buckets.items():
        if len(vals) < min_count:
            continue
        counts = Counter(vals)
        total = float(sum(counts.values()))
        ordered = counts.most_common()
        exact[key] = {
            "top_soc": int(ordered[0][0]),
            "top_prob": float(ordered[0][1] / total),
            "top_alts": [{"soc": int(soc), "prob": float(n / total)} for soc, n in ordered[1:6]],
            "n": int(total),
        }

    if cfg.use_fuzzy_dictionary and HAS_RAPIDFUZZ:
        for (nv, title), vals in title_bucket_counts.items():
            if len(vals) < min_count:
                continue
            counts = Counter(vals)
            total = float(sum(counts.values()))
            ordered = counts.most_common()
            entry = {
                "title": title,
                "top_soc": int(ordered[0][0]),
                "top_prob": float(ordered[0][1] / total),
                "top_alts": [{"soc": int(soc), "prob": float(n / total)} for soc, n in ordered[1:6]],
                "n": int(total),
            }
            if entry["top_prob"] >= cfg.dict_purity:
                fuzzy_buckets[str(nv)][title] = entry
        # Cap each bucket to avoid huge fuzzy scans.
        cap = int(cfg.fuzzy_max_choices_per_bucket)
        if cap > 0:
            for nv in list(fuzzy_buckets.keys()):
                items = sorted(fuzzy_buckets[nv].items(), key=lambda kv: kv[1]["n"], reverse=True)[:cap]
                fuzzy_buckets[nv] = dict(items)

    return {"exact": exact, "fuzzy_buckets": dict(fuzzy_buckets)}


def soc_dictionary_predict(
    df: pd.DataFrame,
    dictionary: Dict[str, Any],
    title_col: str,
    naics_col: str,
    purity: float,
    min_count: int,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    n = len(df)
    pred = np.full(n, -1, dtype=np.int32)
    conf = np.zeros(n, dtype=np.float32)
    used = np.zeros(n, dtype=bool)
    alts: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    exact = dictionary.get("exact", {}) if dictionary else {}
    fuzzy_buckets = dictionary.get("fuzzy_buckets", {}) if dictionary else {}
    if not exact and not fuzzy_buckets:
        return {"pred": pred, "conf": conf, "used": used, "alts": alts}
    if title_col not in df.columns:
        return {"pred": pred, "conf": conf, "used": used, "alts": alts}

    titles = df[title_col].map(normalize_text).tolist()
    naics_vals = cat_to_string_series(df[naics_col]).fillna("MISSING").astype(str).tolist() if naics_col in df.columns else ["MISSING"] * n
    for i, (t, nv) in enumerate(zip(titles, naics_vals)):
        if not t:
            continue
        entry = exact.get(f"{t}||{nv}") or exact.get(f"{t}||MISSING")
        if entry and entry["n"] >= min_count and entry["top_prob"] >= purity:
            pred[i] = int(entry["top_soc"])
            conf[i] = float(entry["top_prob"])
            used[i] = True
            alts[i] = list(entry.get("top_alts", []))[:max(0, cfg.topk_soc - 1)]
            continue

        if cfg.use_fuzzy_dictionary and HAS_RAPIDFUZZ and fuzzy_buckets:
            choices = fuzzy_buckets.get(str(nv)) or fuzzy_buckets.get("MISSING")
            if choices:
                match = rf_process.extractOne(t, choices.keys(), scorer=fuzz.token_sort_ratio, score_cutoff=int(cfg.fuzzy_threshold))
                if match:
                    matched_title = match[0]
                    entry = choices.get(matched_title)
                    if entry and entry["n"] >= min_count and entry["top_prob"] >= purity:
                        # Penalize confidence slightly by fuzzy similarity.
                        sim = float(match[1]) / 100.0
                        pred[i] = int(entry["top_soc"])
                        conf[i] = float(entry["top_prob"] * sim)
                        used[i] = True
                        alts[i] = list(entry.get("top_alts", []))[:max(0, cfg.topk_soc - 1)]
    return {"pred": pred, "conf": conf, "used": used, "alts": alts}


# ---------------------------------------------------------------------
# Semantic dense-vector memory and local LLM reranking
# ---------------------------------------------------------------------

def make_docs_from_columns(df: pd.DataFrame, text_cols: Sequence[str], prefixes: Dict[str, str]) -> List[str]:
    docs: List[str] = []
    for _, row in df.iterrows():
        parts: List[str] = []
        for c in text_cols:
            if c not in df.columns:
                continue
            v = normalize_text(row.get(c))
            if not v:
                continue
            prefix = prefixes.get(c, "")
            parts.append(f"{prefix} {v}".strip())
        docs.append(" | ".join(parts).strip() or "__EMPTY__")
    return docs


def _detect_first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_to_orig = {str(c).lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_to_orig:
            return lower_to_orig[c.lower()]
    return None


def load_reference_codebook(path: str, kind: str) -> Tuple[List[str], List[Any]]:
    """Load an optional SOC/NAICS reference codebook.

    Accepts CSV/XLSX with flexible column names. Useful columns include:
    code/soc/soc_code/naics/naics_code, title/name/occupation/industry,
    description/definition/tasks/examples.
    """
    if not path:
        return [], []
    if not os.path.exists(path):
        eprint(f"WARNING: reference codebook not found: {path}")
        return [], []
    try:
        if path.lower().endswith(('.xlsx', '.xls')):
            ref = pd.read_excel(path)
        else:
            ref = pd.read_csv(path)
    except Exception as exc:
        eprint(f"WARNING: could not read reference codebook {path}: {exc}")
        return [], []
    if ref.empty:
        return [], []
    if kind.lower() == "soc":
        code_col = _detect_first_existing(ref.columns, ["soc", "soc_code", "soc6", "code", "occupation_code"])
    else:
        code_col = _detect_first_existing(ref.columns, ["naics", "naics_code", "code", "industry_code"])
    if code_col is None:
        eprint(f"WARNING: no code column found in reference codebook {path}")
        return [], []
    text_cols = [c for c in ref.columns if c != code_col]
    docs: List[str] = []
    labels: List[Any] = []
    for _, row in ref.iterrows():
        code = clean_code_value(row.get(code_col))
        if code is None or code in COMMON_CODE_MISSINGS:
            continue
        parts: List[str] = []
        for c in text_cols:
            v = normalize_text(row.get(c))
            if v:
                parts.append(f"{c}: {v}")
        if not parts:
            continue
        docs.append(" | ".join(parts))
        labels.append(int(code) if kind.lower() == "soc" and str(code).isdigit() else str(code))
    eprint(f"[{kind.upper()}] Loaded {len(labels)} reference codebook rows from {path}")
    return docs, labels


class SemanticMemory:
    """Dense vector KNN memory for semantic title/industry matching.

    It stores normalized sentence-transformer embeddings and aggregates nearest
    neighbor labels. FAISS is used when available and requested; otherwise a
    scikit-learn brute-force cosine search is used. This makes the improvement
    air-gapped and laptop-safe.
    """
    def __init__(
        self,
        name: str,
        embedding_model: str,
        embedding_backend: str = "torch",
        embedding_device: Optional[str] = None,
        embedding_batch_size: int = 32,
        backend: str = "auto",
        k: int = 35,
        random_seed: int = 42,
    ) -> None:
        self.name = name
        self.embedding_model = embedding_model
        self.embedding_backend = embedding_backend
        self.embedding_device = embedding_device
        self.embedding_batch_size = int(embedding_batch_size)
        self.backend = backend
        self.k = int(k)
        self.random_seed = int(random_seed)
        self.labels: np.ndarray = np.asarray([], dtype=object)
        self.weights: np.ndarray = np.asarray([], dtype=np.float32)
        self.embeddings: Optional[np.ndarray] = None
        self.index: Any = None
        self.nn: Any = None
        self._model: Any = None
        self.actual_backend_: str = "none"

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_model"] = None
        state["index"] = None
        state["nn"] = None
        return state

    def _load_model(self) -> Any:
        if not HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers is required for semantic memory")
        if self._model is None:
            kwargs: Dict[str, Any] = {}
            if self.embedding_device:
                kwargs["device"] = self.embedding_device
            backend = str(self.embedding_backend or "torch").lower()
            if backend in {"onnx", "openvino"}:
                try:
                    self._model = SentenceTransformer(self.embedding_model, backend=backend, **kwargs)
                except TypeError:
                    eprint(f"[{self.name}] WARNING: backend={backend} unsupported; semantic memory falling back to torch.")
                    self._model = SentenceTransformer(self.embedding_model, **kwargs)
                except Exception as exc:
                    eprint(f"[{self.name}] WARNING: backend={backend} failed ({exc}); semantic memory falling back to torch.")
                    self._model = SentenceTransformer(self.embedding_model, **kwargs)
            else:
                self._model = SentenceTransformer(self.embedding_model, **kwargs)
        return self._model

    def encode(self, docs: Sequence[str]) -> np.ndarray:
        model = self._load_model()
        emb = model.encode(
            list(docs),
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)
        # Defensive normalization if a backend ignores normalize_embeddings.
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms <= 0] = 1.0
        return emb / norms

    def fit(self, docs: Sequence[str], labels: Sequence[Any], weights: Optional[Sequence[float]] = None) -> "SemanticMemory":
        good_docs: List[str] = []
        good_labels: List[Any] = []
        good_weights: List[float] = []
        for i, (doc, label) in enumerate(zip(docs, labels)):
            if label is None:
                continue
            if not str(doc).strip() or str(doc).strip() == "__EMPTY__":
                continue
            good_docs.append(str(doc))
            good_labels.append(label)
            if weights is None:
                good_weights.append(1.0)
            else:
                try:
                    good_weights.append(float(weights[i]))
                except Exception:
                    good_weights.append(1.0)
        self.labels = np.asarray(good_labels, dtype=object)
        self.weights = np.asarray(good_weights, dtype=np.float32)
        if not good_docs:
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
            return self
        eprint(f"[{self.name}] Encoding {len(good_docs):,} semantic-memory exemplars with {self.embedding_model} ({self.embedding_backend})...")
        self.embeddings = self.encode(good_docs).astype(np.float32, copy=False)
        self._build_index()
        return self

    def _build_index(self) -> None:
        self.index = None
        self.nn = None
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return
        requested = str(self.backend or "auto").lower()
        use_faiss = requested in {"auto", "faiss"} and HAS_FAISS
        if use_faiss:
            try:
                self.index = faiss.IndexFlatIP(int(self.embeddings.shape[1]))
                self.index.add(np.ascontiguousarray(self.embeddings.astype(np.float32, copy=False)))
                self.actual_backend_ = "faiss"
                eprint(f"[{self.name}] Semantic-memory backend: FAISS IndexFlatIP, vectors={self.embeddings.shape[0]:,}, dim={self.embeddings.shape[1]}")
                return
            except Exception as exc:
                eprint(f"[{self.name}] WARNING: FAISS index failed ({exc}); falling back to sklearn cosine search.")
        self.nn = None
        try:
            from sklearn.neighbors import NearestNeighbors
            self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
            self.nn.fit(self.embeddings)
            self.actual_backend_ = "sklearn_cosine"
            eprint(f"[{self.name}] Semantic-memory backend: sklearn cosine, vectors={self.embeddings.shape[0]:,}, dim={self.embeddings.shape[1]}")
        except Exception as exc:
            self.actual_backend_ = "none"
            eprint(f"[{self.name}] WARNING: semantic-memory index failed: {exc}")

    def ensure_index(self) -> None:
        if (self.index is None and self.nn is None) and self.embeddings is not None and self.embeddings.shape[0] > 0:
            self._build_index()

    def predict_topk(self, docs: Sequence[str], topk: int = 4, k: Optional[int] = None) -> Dict[str, Any]:
        n = len(docs)
        out_codes: List[List[Any]] = [[None for _ in range(topk)] for _ in range(n)]
        out_prob = np.zeros((n, topk), dtype=np.float32)
        out_max_sim = np.zeros(n, dtype=np.float32)
        out_conf = np.zeros(n, dtype=np.float32)
        if n == 0 or self.embeddings is None or self.embeddings.shape[0] == 0 or len(self.labels) == 0:
            return {"codes": out_codes, "prob": out_prob, "max_sim": out_max_sim, "conf": out_conf}
        self.ensure_index()
        q = self.encode(docs)
        nn_k = max(1, min(int(k or self.k), int(self.embeddings.shape[0])))
        if self.index is not None:
            sims, idxs = self.index.search(np.ascontiguousarray(q.astype(np.float32, copy=False)), nn_k)
        elif self.nn is not None:
            distances, idxs = self.nn.kneighbors(q, n_neighbors=nn_k, return_distance=True)
            sims = 1.0 - distances
        else:
            return {"codes": out_codes, "prob": out_prob, "max_sim": out_max_sim, "conf": out_conf}
        sims = np.asarray(sims, dtype=np.float32)
        idxs = np.asarray(idxs, dtype=np.int64)
        for i in range(n):
            agg: Dict[Any, float] = defaultdict(float)
            best_sim = 0.0
            for sim, idx in zip(sims[i], idxs[i]):
                if idx < 0 or idx >= len(self.labels):
                    continue
                # Inner product of normalized embeddings equals cosine similarity.
                sim_float = max(0.0, float(sim))
                best_sim = max(best_sim, sim_float)
                label = self.labels[int(idx)]
                exemplar_weight = float(self.weights[int(idx)]) if len(self.weights) > int(idx) else 1.0
                # Square the similarity to reward truly close semantic neighbors.
                agg[label] += (sim_float ** 2) * exemplar_weight
            total = float(sum(agg.values()))
            if total <= 0:
                continue
            ordered = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:topk]
            out_max_sim[i] = float(best_sim)
            out_conf[i] = float(ordered[0][1] / total) if ordered else 0.0
            for rank, (label, val) in enumerate(ordered):
                out_codes[i][rank] = label
                out_prob[i, rank] = float(val / total)
        return {"codes": out_codes, "prob": out_prob, "max_sim": out_max_sim, "conf": out_conf}


def build_semantic_memory(
    name: str,
    df: pd.DataFrame,
    labels: Sequence[Any],
    text_cols: Sequence[str],
    prefixes: Dict[str, str],
    cfg: TrainConfig,
    kind: str,
    reference_csv: str = "",
) -> Optional[SemanticMemory]:
    if not cfg.use_semantic_memory:
        return None
    if not HAS_SENTENCE_TRANSFORMERS:
        eprint(f"[{name}] Semantic memory requested but sentence-transformers is unavailable; skipping.")
        return None
    docs = make_docs_from_columns(df, text_cols, prefixes)
    clean_labels: List[Any] = []
    clean_docs: List[str] = []
    weights: List[float] = []
    for doc, label in zip(docs, labels):
        cv = clean_code_value(label)
        if cv is None or cv in COMMON_CODE_MISSINGS:
            continue
        clean_docs.append(doc)
        clean_labels.append(int(cv) if kind.lower() == "soc" and str(cv).isdigit() else str(cv))
        weights.append(1.0)
    ref_docs, ref_labels = load_reference_codebook(reference_csv, kind) if reference_csv else ([], [])
    for doc, label in zip(ref_docs, ref_labels):
        clean_docs.append(doc)
        clean_labels.append(label)
        weights.append(float(cfg.semantic_reference_weight))
    if len(clean_docs) < 5:
        eprint(f"[{name}] Too few semantic-memory exemplars ({len(clean_docs)}); skipping.")
        return None
    mem = SemanticMemory(
        name=name,
        embedding_model=cfg.embedding_model,
        embedding_backend=cfg.embedding_backend,
        embedding_device=cfg.embedding_device,
        embedding_batch_size=cfg.embedding_batch_size,
        backend=cfg.semantic_backend,
        k=cfg.semantic_k,
        random_seed=cfg.random_seed,
    )
    mem.fit(clean_docs, clean_labels, weights=weights)
    return mem


def merge_ranked_candidates(
    ml_codes: Sequence[Any],
    ml_probs: Sequence[float],
    sem_codes: Optional[Sequence[Any]],
    sem_probs: Optional[Sequence[float]],
    sem_conf: float,
    sem_sim: float,
    cfg: TrainConfig,
    topk: int,
    code_kind: str,
) -> Tuple[List[Any], List[float]]:
    merged: Dict[Any, float] = defaultdict(float)
    ml_factor = max(0.0, float(cfg.ml_fusion_weight))
    sem_factor = 0.0
    for code, prob in zip(ml_codes, ml_probs):
        if code is None:
            continue
        try:
            p = float(prob)
        except Exception:
            continue
        if p <= 0:
            continue
        merged[code] += ml_factor * p
    if sem_codes is not None and sem_probs is not None:
        if float(sem_sim) >= float(cfg.semantic_min_similarity) and float(sem_conf) >= float(cfg.semantic_min_confidence):
            sem_factor = max(0.0, float(cfg.semantic_fusion_weight)) * max(0.25, float(sem_sim))
            for code, prob in zip(sem_codes, sem_probs):
                if code is None:
                    continue
                try:
                    p = float(prob)
                except Exception:
                    continue
                if p <= 0:
                    continue
                merged[code] += sem_factor * p
    if not merged:
        return [None for _ in range(topk)], [0.0 for _ in range(topk)]
    ordered = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    denom = max(1e-9, ml_factor + sem_factor)
    codes = [c for c, _ in ordered]
    probs = [float(max(0.0, min(1.0, v / denom))) for _, v in ordered]
    while len(codes) < topk:
        codes.append(None)
        probs.append(0.0)
    if code_kind == "soc":
        fixed_codes: List[Any] = []
        for c in codes:
            try:
                fixed_codes.append(int(c) if c is not None else None)
            except Exception:
                fixed_codes.append(None)
        codes = fixed_codes
    else:
        codes = [str(c) if c is not None else None for c in codes]
    return codes, probs


_LLM_CACHE: Dict[Tuple[str, int, int, int], Any] = {}


def get_local_llm(cfg: TrainConfig) -> Any:
    if not cfg.use_local_llm_fallback:
        return None
    if not HAS_LLAMA_CPP:
        eprint("WARNING: --use_local_llm_fallback requested but llama-cpp-python is unavailable; skipping LLM reranker.")
        return None
    if not cfg.llm_model_path or not os.path.exists(cfg.llm_model_path):
        eprint("WARNING: --use_local_llm_fallback requested but --llm_model_path is missing/not found; skipping LLM reranker.")
        return None
    key = (cfg.llm_model_path, int(cfg.llm_ctx_size), int(cfg.llm_threads), int(cfg.random_seed))
    if key not in _LLM_CACHE:
        eprint(f"Loading local LLM reranker from {cfg.llm_model_path}")
        _LLM_CACHE[key] = Llama(
            model_path=cfg.llm_model_path,
            n_ctx=int(cfg.llm_ctx_size),
            n_threads=int(cfg.llm_threads),
            n_gpu_layers=0,
            seed=int(cfg.random_seed),
            verbose=False,
        )
    return _LLM_CACHE[key]


def llm_choose_candidate(
    cfg: TrainConfig,
    kind: str,
    row_context: str,
    candidates: Sequence[Any],
) -> Optional[Any]:
    llm = get_local_llm(cfg)
    if llm is None:
        return None
    allowed = [str(c) for c in candidates if c is not None]
    if not allowed:
        return None
    prompt = (
        "You are an expert labor-market survey coder. Choose exactly one code from the allowed list.\n"
        f"Task: select the best {kind.upper()} code for this survey response.\n"
        f"Survey context:\n{row_context}\n\n"
        f"Allowed candidate codes: {', '.join(allowed)}\n"
        "Return only compact JSON exactly like {\"code\":\"123456\"}. Do not explain."
    )
    try:
        out = llm(
            prompt,
            max_tokens=int(cfg.llm_max_tokens),
            temperature=float(cfg.llm_temperature),
            top_p=float(cfg.llm_top_p),
            stop=["\n\n"],
        )
        text = str(out.get("choices", [{}])[0].get("text", ""))
    except Exception as exc:
        eprint(f"WARNING: local LLM rerank failed: {exc}")
        return None
    # Parse JSON-ish or bare code.
    m = re.search(r'"code"\s*:\s*"?([0-9A-Za-z\-]+)"?', text)
    code = m.group(1) if m else None
    if code is None:
        m2 = re.search(r"\b\d{2,6}\b", text)
        code = m2.group(0) if m2 else None
    if code is None:
        return None
    if code in allowed:
        return int(code) if kind.lower() == "soc" and code.isdigit() else code
    if cfg.llm_allow_free_code:
        return int(code) if kind.lower() == "soc" and code.isdigit() else code
    return None


def row_context_for_llm(df: pd.DataFrame, row_pos: int, cols: Sequence[str]) -> str:
    if row_pos < 0 or row_pos >= len(df):
        return ""
    row = df.iloc[row_pos]
    parts: List[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        val = row.get(c)
        norm = normalize_text(val)
        if norm:
            parts.append(f"{c}: {norm}")
    return "\n".join(parts)[:3000]


def maybe_llm_rerank_naics(pred: Dict[str, Any], score_df: pd.DataFrame, cfg: TrainConfig, text_cols: Sequence[str]) -> Dict[str, Any]:
    if not cfg.use_local_llm_fallback:
        return pred
    n = len(score_df)
    used = 0
    for i in range(n):
        if used >= int(cfg.llm_max_rows):
            break
        if pred["top_prob"][i, 0] >= float(cfg.llm_trigger_confidence_naics):
            continue
        candidates = [c for c in pred["top_code"][i] if c is not None]
        if not candidates:
            continue
        ctx = row_context_for_llm(score_df, i, text_cols)
        if not ctx:
            continue
        chosen = llm_choose_candidate(cfg, "naics", ctx, candidates)
        if chosen is None:
            continue
        # Move chosen to rank 1 and keep the remaining candidates in order.
        old_codes = [c for c in pred["top_code"][i] if c is not None and str(c) != str(chosen)]
        new_codes = [str(chosen)] + old_codes
        for k in range(pred["top_prob"].shape[1]):
            pred["top_code"][i][k] = new_codes[k] if k < len(new_codes) else None
            pred["top_prob"][i, k] = float(cfg.llm_rerank_confidence) if k == 0 else max(0.0, float(pred["top_prob"][i, k]) * 0.75)
        used += 1
    if used:
        eprint(f"[LLM] Reranked {used} low-confidence NAICS rows with local llama.cpp model")
    return pred


def maybe_llm_rerank_soc(pred: Dict[str, Any], score_df: pd.DataFrame, cfg: TrainConfig, text_cols: Sequence[str]) -> Dict[str, Any]:
    if not cfg.use_local_llm_fallback:
        return pred
    n = len(score_df)
    used = 0
    for i in range(n):
        if used >= int(cfg.llm_max_rows):
            break
        if pred["top_prob"][i, 0] >= float(cfg.llm_trigger_confidence_soc):
            continue
        candidates = [int(c) for c in pred["top_soc"][i].tolist() if int(c) > 0]
        if not candidates:
            continue
        ctx = row_context_for_llm(score_df, i, text_cols)
        if not ctx:
            continue
        chosen = llm_choose_candidate(cfg, "soc", ctx, candidates)
        if chosen is None:
            continue
        chosen_int = int(chosen)
        old_codes = [int(c) for c in candidates if int(c) != chosen_int]
        new_codes = [chosen_int] + old_codes
        for k in range(pred["top_prob"].shape[1]):
            pred["top_soc"][i, k] = new_codes[k] if k < len(new_codes) else -1
            pred["top_prob"][i, k] = float(cfg.llm_rerank_confidence) if k == 0 else max(0.0, float(pred["top_prob"][i, k]) * 0.75)
        used += 1
    if used:
        eprint(f"[LLM] Reranked {used} low-confidence SOC rows with local llama.cpp model")
    return pred



# ---------------------------------------------------------------------
# Model selection and ensembling
# ---------------------------------------------------------------------

def choose_penalty(alpha: float) -> Tuple[str, Optional[float]]:
    alpha = float(alpha)
    if alpha <= 0:
        return "l2", None
    if alpha >= 1:
        return "l1", None
    return "elasticnet", alpha


def choose_cv_splits(y: np.ndarray, requested_folds: int) -> int:
    counts = Counter(y.tolist())
    if not counts:
        return 0
    min_class_n = min(counts.values())
    if min_class_n < 2:
        return 0
    return max(2, min(int(requested_folds), int(min_class_n)))


def can_use_boosters(stage_name: str, n_rows: int, n_features: int, n_classes: int, cfg: TrainConfig) -> bool:
    if cfg.model_search not in {"robust", "max"}:
        return False
    if n_classes > int(cfg.boost_max_classes):
        return False
    if n_features > int(cfg.boost_max_features):
        # High-dimensional TF-IDF + multiclass boosted trees can become a hedgehog made of RAM.
        return False
    if stage_name.startswith("within") and n_rows < 80:
        return False
    return True


def candidate_estimators(
    cfg: TrainConfig,
    stage_name: str,
    n_rows: int,
    n_features: int,
    n_classes: int,
    features_nonnegative: bool,
    parallel_cfg: ParallelConfig,
) -> List[Tuple[str, Any]]:
    candidates: List[Tuple[str, Any]] = []
    l1_ratio = 0.25 if cfg.profile == "accuracy" else 0.15

    # Strong sparse linear workhorse.
    candidates.append((
        "sgd_log_elasticnet",
        SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=float(l1_ratio),
            alpha=float(cfg.sgd_alpha),
            max_iter=int(cfg.sgd_max_iter),
            tol=float(cfg.sgd_tol),
            class_weight="balanced",
            random_state=int(cfg.random_seed),
        ),
    ))

    # More expensive but often stronger on text when rows/classes are not huge.
    include_logreg = True
    if stage_name.startswith("within") and (n_rows > 70000 or n_classes > 180):
        include_logreg = False
    if stage_name.startswith("naics") and (n_rows > 100000 or n_classes > 250):
        include_logreg = False
    if include_logreg:
        penalty, chosen_l1 = choose_penalty(l1_ratio)
        kwargs: Dict[str, Any] = dict(
            solver="saga",
            max_iter=int(cfg.logreg_max_iter),
            C=float(cfg.logreg_C),
            class_weight="balanced",
            random_state=int(cfg.random_seed),
            n_jobs=1,
        )
        if penalty == "elasticnet":
            kwargs.update(penalty="elasticnet", l1_ratio=float(chosen_l1))
        else:
            kwargs.update(penalty=penalty)
        candidates.append(("logreg_saga_balanced", LogisticRegression(**kwargs)))

    # Linear SVM has excellent high-dimensional text behavior. Probabilities are
    # approximated from decision scores for ranking and ensembling.
    if cfg.model_search in {"robust", "max"} and n_classes <= 120 and n_rows <= 90000:
        try:
            candidates.append((
                "linear_svc_balanced",
                LinearSVC(C=float(cfg.linearsvc_C), class_weight="balanced", max_iter=6000, dual="auto", random_state=int(cfg.random_seed)),
            ))
        except TypeError:
            candidates.append((
                "linear_svc_balanced",
                LinearSVC(C=float(cfg.linearsvc_C), class_weight="balanced", max_iter=6000, random_state=int(cfg.random_seed)),
            ))

    # ComplementNB is very strong for pure non-negative count/TF-IDF style data.
    if features_nonnegative and cfg.model_search in {"light", "robust", "max"}:
        candidates.append(("complement_nb", ComplementNB(alpha=0.5)))

    use_boost = can_use_boosters(stage_name, n_rows, n_features, n_classes, cfg)
    if use_boost and HAS_LIGHTGBM:
        candidates.append((
            "lightgbm_multiclass",
            lgb.LGBMClassifier(
                objective="multiclass",
                n_estimators=int(cfg.boost_n_estimators),
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                class_weight="balanced",
                n_jobs=int(parallel_cfg.native_threads),
                random_state=int(cfg.random_seed),
                verbose=-1,
            ),
        ))
    if use_boost and HAS_XGBOOST and cfg.model_search == "max":
        candidates.append((
            "xgboost_hist",
            XGBLabelEncodedClassifier(
                n_estimators=max(150, int(cfg.boost_n_estimators)),
                max_depth=5,
                learning_rate=0.045,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                n_jobs=int(parallel_cfg.native_threads),
                random_state=int(cfg.random_seed),
            ),
        ))
    if use_boost and HAS_CATBOOST and cfg.model_search == "max" and n_rows <= 50000:
        candidates.append((
            "catboost_sparse",
            CatBoostLabelEncodedClassifier(
                iterations=max(180, int(cfg.boost_n_estimators)),
                depth=6,
                learning_rate=0.045,
                l2_leaf_reg=5.0,
                thread_count=int(parallel_cfg.native_threads),
                random_state=int(cfg.random_seed),
            ),
        ))

    return candidates


def evaluate_candidate_cv(
    name: str,
    proto: Any,
    x: sp.csr_matrix,
    y: np.ndarray,
    classes: np.ndarray,
    splits: int,
    cfg: TrainConfig,
) -> CandidateScore:
    if splits < 2:
        return CandidateScore(name=name, macro_f1=None, log_loss_value=None, failed=False)
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=int(cfg.random_seed))
    fold_f1: List[float] = []
    fold_ll: List[float] = []
    try:
        for tr_idx, va_idx in splitter.split(np.zeros(len(y)), y):
            est = clone(proto)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", FutureWarning)
                est.fit(x[tr_idx], y[tr_idx])
            pred = est.predict(x[va_idx])
            fold_f1.append(float(f1_score(y[va_idx], pred, average="macro", zero_division=0)))
            try:
                p = estimator_predict_proba(est, x[va_idx], classes)
                fold_ll.append(float(log_loss(y[va_idx], p, labels=classes)))
            except Exception:
                fold_ll.append(float("inf"))
    except Exception as exc:
        return CandidateScore(name=name, macro_f1=None, log_loss_value=None, failed=True, reason=str(exc)[:500])
    return CandidateScore(
        name=name,
        macro_f1=float(np.mean(fold_f1)) if fold_f1 else None,
        log_loss_value=float(np.mean(fold_ll)) if fold_ll else None,
        failed=False,
    )


def make_optuna_candidate(
    x: sp.csr_matrix,
    y: np.ndarray,
    classes: np.ndarray,
    cfg: TrainConfig,
    stage_name: str,
    trials: int,
    parallel_cfg: ParallelConfig,
    features_nonnegative: bool,
    base_splits: int,
) -> Optional[Tuple[str, Any]]:
    if trials <= 0 or not HAS_OPTUNA or cfg.model_search != "max" or base_splits < 2:
        return None

    use_boost = can_use_boosters(stage_name, x.shape[0], x.shape[1], len(classes), cfg)
    families = ["sgd", "logreg"]
    if use_boost and HAS_LIGHTGBM:
        families.append("lightgbm")
    if use_boost and HAS_XGBOOST and len(classes) <= min(80, cfg.boost_max_classes):
        families.append("xgboost")

    eval_splits = min(base_splits, 3)

    def build_from_trial(trial: Any) -> Tuple[str, Any]:
        fam = trial.suggest_categorical("family", families)
        if fam == "sgd":
            return "optuna_sgd", SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                l1_ratio=trial.suggest_float("sgd_l1_ratio", 0.01, 0.7),
                alpha=trial.suggest_float("sgd_alpha", 1e-6, 5e-3, log=True),
                max_iter=int(cfg.sgd_max_iter),
                tol=float(cfg.sgd_tol),
                class_weight="balanced",
                random_state=int(cfg.random_seed),
            )
        if fam == "logreg":
            return "optuna_logreg", LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                l1_ratio=trial.suggest_float("lr_l1_ratio", 0.01, 0.6),
                C=trial.suggest_float("lr_C", 0.05, 8.0, log=True),
                max_iter=int(cfg.logreg_max_iter),
                class_weight="balanced",
                random_state=int(cfg.random_seed),
                n_jobs=1,
            )
        if fam == "lightgbm":
            return "optuna_lightgbm", lgb.LGBMClassifier(
                objective="multiclass",
                n_estimators=trial.suggest_int("lgb_estimators", 120, max(160, cfg.boost_n_estimators * 2)),
                learning_rate=trial.suggest_float("lgb_lr", 0.015, 0.12, log=True),
                num_leaves=trial.suggest_int("lgb_leaves", 15, 95),
                max_depth=trial.suggest_int("lgb_depth", -1, 12),
                subsample=trial.suggest_float("lgb_subsample", 0.65, 1.0),
                colsample_bytree=trial.suggest_float("lgb_colsample", 0.55, 1.0),
                reg_lambda=trial.suggest_float("lgb_lambda", 0.1, 10.0, log=True),
                class_weight="balanced",
                n_jobs=int(parallel_cfg.native_threads),
                random_state=int(cfg.random_seed),
                verbose=-1,
            )
        return "optuna_xgboost", XGBLabelEncodedClassifier(
            n_estimators=trial.suggest_int("xgb_estimators", 120, max(160, cfg.boost_n_estimators * 2)),
            max_depth=trial.suggest_int("xgb_depth", 3, 8),
            learning_rate=trial.suggest_float("xgb_lr", 0.015, 0.12, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.65, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample", 0.55, 1.0),
            reg_lambda=trial.suggest_float("xgb_lambda", 0.1, 10.0, log=True),
            reg_alpha=trial.suggest_float("xgb_alpha", 1e-8, 2.0, log=True),
            n_jobs=int(parallel_cfg.native_threads),
            random_state=int(cfg.random_seed),
        )

    def objective(trial: Any) -> float:
        nm, est = build_from_trial(trial)
        score = evaluate_candidate_cv(nm, est, x, y, classes, eval_splits, cfg)
        if score.failed or score.macro_f1 is None:
            return -1.0
        # Macro-F1 is primary. Log-loss nudges the model toward better confidence.
        ll_penalty = 0.0 if score.log_loss_value is None or not np.isfinite(score.log_loss_value) else min(score.log_loss_value, 5.0) * 0.005
        return float(score.macro_f1 - ll_penalty)

    try:
        sampler = optuna.samplers.TPESampler(seed=int(cfg.random_seed))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=int(trials), timeout=cfg.optuna_timeout_seconds, n_jobs=1, show_progress_bar=False)
        if study.best_trial is None or study.best_value < 0:
            return None
        # Rebuild best estimator using the saved best trial object.
        nm, est = build_from_trial(study.best_trial)
        return (nm, est)
    except Exception as exc:
        eprint(f"[{stage_name}] Optuna skipped after error: {exc}")
        return None


def fit_best_classifier(
    x: sp.csr_matrix,
    y: np.ndarray,
    cfg: TrainConfig,
    stage_name: str,
    nfolds: int,
    features_nonnegative: bool,
    parallel_cfg: ParallelConfig,
    optuna_trials: int = 0,
) -> FittedClassifier:
    y = np.asarray(y)
    classes = np.sort(np.unique(y))
    if classes.size < 2:
        raise ValueError(f"{stage_name}: need at least 2 classes to train, got {classes.size}")

    candidates = candidate_estimators(cfg, stage_name, x.shape[0], x.shape[1], len(classes), features_nonnegative, parallel_cfg)
    splits = choose_cv_splits(y, nfolds)

    tuned = make_optuna_candidate(x, y, classes, cfg, stage_name, optuna_trials, parallel_cfg, features_nonnegative, splits)
    if tuned is not None:
        candidates.insert(0, tuned)

    if not candidates:
        raise RuntimeError(f"{stage_name}: no model candidates are available")

    cv_scores: List[CandidateScore] = []
    if splits >= 2 and cfg.model_search != "none":
        for name, proto in candidates:
            eprint(f"[{stage_name}] CV candidate: {name}")
            score = evaluate_candidate_cv(name, proto, x, y, classes, splits, cfg)
            cv_scores.append(score)
            if score.failed:
                eprint(f"[{stage_name}]   failed: {score.reason}")
            else:
                f1s = "NA" if score.macro_f1 is None else f"{score.macro_f1:.4f}"
                lls = "NA" if score.log_loss_value is None or not np.isfinite(score.log_loss_value) else f"{score.log_loss_value:.4f}"
                eprint(f"[{stage_name}]   macro-F1={f1s} logloss={lls}")
    else:
        cv_scores = [CandidateScore(name=candidates[0][0], macro_f1=None, log_loss_value=None, failed=False)]

    usable = [s for s in cv_scores if not s.failed]
    if usable and any(s.macro_f1 is not None for s in usable):
        def sort_key(s: CandidateScore) -> Tuple[float, float]:
            f1 = -1.0 if s.macro_f1 is None else float(s.macro_f1)
            ll = 999.0 if s.log_loss_value is None or not np.isfinite(s.log_loss_value) else float(s.log_loss_value)
            return (f1, -ll)
        usable_sorted = sorted(usable, key=sort_key, reverse=True)
        chosen_names = [s.name for s in usable_sorted[:max(1, int(cfg.ensemble_size))]]
    else:
        chosen_names = [candidates[0][0]]

    # Fit selected models on the full training matrix.
    proto_by_name = {name: proto for name, proto in candidates}
    score_by_name = {s.name: s for s in cv_scores}
    estimators: List[Any] = []
    model_names: List[str] = []
    weights: List[float] = []
    for name in chosen_names:
        proto = proto_by_name.get(name)
        if proto is None:
            continue
        try:
            est = clone(proto)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", FutureWarning)
                est.fit(x, y)
            estimators.append(est)
            model_names.append(name)
            sc = score_by_name.get(name)
            if sc and sc.macro_f1 is not None and sc.macro_f1 > 0:
                weights.append(float(sc.macro_f1) ** float(cfg.ensemble_power))
            else:
                weights.append(1.0)
        except Exception as exc:
            eprint(f"[{stage_name}] Final fit failed for {name}: {exc}")

    if not estimators:
        # Last-ditch fallback: fit first candidate.
        name, proto = candidates[0]
        est = clone(proto)
        est.fit(x, y)
        estimators = [est]
        model_names = [name]
        weights = [1.0]

    return FittedClassifier(
        stage_name=stage_name,
        model_names=model_names,
        estimators=estimators,
        weights=weights,
        classes=classes,
        cv_scores=cv_scores,
        n_rows=int(x.shape[0]),
        n_features=int(x.shape[1]),
        n_classes=int(len(classes)),
    )


# ---------------------------------------------------------------------
# NAICS model
# ---------------------------------------------------------------------

def train_naics_target(
    train_df: pd.DataFrame,
    target_col: str,
    text_cols: Sequence[str],
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    prefixes: Dict[str, str],
    cfg: TrainConfig,
    parallel_cfg: ParallelConfig,
) -> Dict[str, Any]:
    if target_col not in train_df.columns:
        raise ValueError(f"Training data missing NAICS target column: {target_col}")

    y_codes = clean_code_series(train_df[target_col])
    keep = y_codes.notna() & (~y_codes.map(lambda v: str(v).strip() in COMMON_CODE_MISSINGS if v is not None else True))
    if cfg.missing_naics_values:
        keep = keep & (~y_codes.isin(list(cfg.missing_naics_values)))

    df = train_df.loc[keep].copy()
    y_codes = y_codes.loc[keep].astype(str)

    # Keep only classes with enough support for stable CV. Predictions for classes
    # below this threshold are still represented by the dictionary if they recur.
    counts = y_codes.value_counts()
    keep_classes = counts[counts >= cfg.min_naics_n].index
    rare = sorted(set(y_codes.unique()) - set(keep_classes))
    if rare:
        eprint(f"[NAICS] Dropping {len(rare)} rare NAICS classes with <{cfg.min_naics_n} rows for ML training")
    keep2 = y_codes.isin(keep_classes)
    df_ml = df.loc[keep2].copy()
    y_ml = y_codes.loc[keep2].to_numpy(dtype=object)

    if len(df_ml) < 50 or len(np.unique(y_ml)) < 2:
        raise ValueError(f"Not enough labeled NAICS rows/classes after filtering: rows={len(df_ml)}, classes={len(np.unique(y_ml))}")

    eprint(f"\n[NAICS] Labeled rows used for ML training: {len(df_ml)}")
    eprint(f"[NAICS] Unique classes used for ML: {len(np.unique(y_ml))}")

    key_specs = [
        ("employer", ["qa20"]),
        ("employer_title", ["qa20", "qe4"]),
        ("industry_other", ["qe1oth"]),
        ("selfemp_other", ["qe8aoth"]),
        ("title_industry", ["qe4", "qe1oth"]),
    ]
    dictionary = build_exact_dictionary(df, y_codes.to_numpy(dtype=object), key_specs, cfg.naics_dict_min_count)
    eprint(f"[NAICS] Dictionary entries: {len(dictionary)}")

    semantic_memory = build_semantic_memory(
        name="NAICS_semantic_memory",
        df=df,
        labels=y_codes.to_numpy(dtype=object),
        text_cols=text_cols,
        prefixes=prefixes,
        cfg=cfg,
        kind="naics",
        reference_csv=cfg.naics_reference_csv,
    )

    feat = HybridFeaturizer(
        text_cols=[c for c in text_cols if c in df_ml.columns],
        cat_cols=[c for c in cat_cols if c in df_ml.columns and c != target_col],
        num_cols=[c for c in num_cols if c in df_ml.columns],
        prefixes=prefixes,
        cfg=cfg,
    )
    eprint("[NAICS] Building features...")
    x = feat.fit_transform(df_ml)
    eprint(f"[NAICS] Feature matrix: {x.shape[0]} rows x {x.shape[1]} cols | nnz={x.nnz:,} | approx_mem={sparse_nbytes(x)/1024**3:.2f} GB")

    with threadpool_limits(limits=parallel_cfg.native_threads):
        clf = fit_best_classifier(
            x=x,
            y=y_ml,
            cfg=cfg,
            stage_name="naics",
            nfolds=cfg.nfolds_naics,
            features_nonnegative=feat.features_are_nonnegative_,
            parallel_cfg=parallel_cfg,
            optuna_trials=cfg.optuna_trials_naics,
        )
    eprint(f"[NAICS] Selected model(s): {clf.model_name}")

    del x
    gc.collect()
    return {
        "version": VERSION,
        "kind": "naics_flat",
        "target_col": target_col,
        "dictionary": dictionary,
        "dictionary_key_specs": key_specs,
        "semantic_memory": semantic_memory,
        "featurizer": feat,
        "classifier": clf,
        "text_cols": list(text_cols),
        "cat_cols": list(cat_cols),
        "num_cols": list(num_cols),
    }


def predict_naics_topk(model_obj: Dict[str, Any], score_df: pd.DataFrame, cfg: TrainConfig) -> Dict[str, Any]:
    n = len(score_df)
    top_code: List[List[Optional[str]]] = [[None for _ in range(cfg.topk_naics)] for _ in range(n)]
    top_prob = np.zeros((n, cfg.topk_naics), dtype=np.float32)

    d = exact_dictionary_predict(
        df=score_df,
        dictionary=model_obj.get("dictionary", {}),
        key_specs=model_obj.get("dictionary_key_specs", []),
        purity=cfg.naics_dict_purity,
        min_count=cfg.naics_dict_min_count,
        topk=cfg.topk_naics,
    )
    for i in np.where(d["used"])[0].tolist():
        top_code[i][0] = d["pred"][i]
        top_prob[i, 0] = float(d["conf"][i])
        for j, alt in enumerate(d["alts"][i][:max(0, cfg.topk_naics - 1)]):
            top_code[i][j + 1] = str(alt["code"])
            top_prob[i, j + 1] = float(alt["prob"])

    title_cols = model_obj.get("text_cols", [])
    text_present = has_any_normalized_text(score_df, title_cols)
    rows_model = np.where((~d["used"]) & text_present)[0]
    if rows_model.size == 0:
        pred = {"top_code": top_code, "top_prob": top_prob, "dict_used": d["used"], "text_present": text_present}
        return maybe_llm_rerank_naics(pred, score_df, cfg, title_cols)

    # Semantic vector memory gives synonym-aware candidates before/fused with ML.
    sem_codes_global: List[Optional[List[Any]]] = [None] * n
    sem_probs_global: List[Optional[List[float]]] = [None] * n
    sem_conf_global = np.zeros(n, dtype=np.float32)
    sem_sim_global = np.zeros(n, dtype=np.float32)
    sem_mem: Optional[SemanticMemory] = model_obj.get("semantic_memory")
    if sem_mem is not None:
        try:
            sem_docs = make_docs_from_columns(score_df.iloc[rows_model], title_cols, model_obj.get("featurizer").prefixes if model_obj.get("featurizer") is not None else {})
            sem = sem_mem.predict_topk(sem_docs, topk=cfg.topk_naics, k=cfg.semantic_k)
            for local_i, global_i in enumerate(rows_model.tolist()):
                sem_codes_global[global_i] = sem["codes"][local_i]
                sem_probs_global[global_i] = sem["prob"][local_i].tolist()
                sem_conf_global[global_i] = float(sem["conf"][local_i])
                sem_sim_global[global_i] = float(sem["max_sim"][local_i])
        except Exception as exc:
            eprint(f"[NAICS] Semantic memory prediction skipped after error: {exc}")

    feat: HybridFeaturizer = model_obj["featurizer"]
    clf: FittedClassifier = model_obj["classifier"]
    classes = np.asarray(clf.classes)

    for batch in batched_indices(rows_model, cfg.score_batch_size):
        xs = feat.transform(score_df.iloc[batch])
        p = clf.predict_proba(xs)
        for local_i, global_i in enumerate(batch.tolist()):
            ord_idx = safe_top_indices(p[local_i], min(cfg.topk_naics, p.shape[1]))
            ml_codes = [str(classes[j]) for j in ord_idx.tolist()]
            ml_probs = [float(p[local_i, j]) for j in ord_idx.tolist()]
            codes, probs = merge_ranked_candidates(
                ml_codes=ml_codes,
                ml_probs=ml_probs,
                sem_codes=sem_codes_global[global_i],
                sem_probs=sem_probs_global[global_i],
                sem_conf=float(sem_conf_global[global_i]),
                sem_sim=float(sem_sim_global[global_i]),
                cfg=cfg,
                topk=cfg.topk_naics,
                code_kind="naics",
            )
            for rank in range(cfg.topk_naics):
                top_code[global_i][rank] = codes[rank]
                top_prob[global_i, rank] = float(probs[rank])
        del xs
        gc.collect()

    pred = {"top_code": top_code, "top_prob": top_prob, "dict_used": d["used"], "text_present": text_present}
    return maybe_llm_rerank_naics(pred, score_df, cfg, title_cols)


# ---------------------------------------------------------------------
# SOC hierarchical model
# ---------------------------------------------------------------------

def train_hierarchical_soc_target(
    train_df: pd.DataFrame,
    y_col: str,
    title_col: str,
    naics_col: str,
    text_cols: Sequence[str],
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    prefixes: Dict[str, str],
    cfg: TrainConfig,
    parallel_cfg: ParallelConfig,
) -> Dict[str, Any]:
    if y_col not in train_df.columns:
        raise ValueError(f"Training data missing SOC target column: {y_col}")

    y_soc = pd.to_numeric(train_df[y_col], errors="coerce").round().astype("Int64")
    keep = y_soc.notna() & (y_soc >= 100000) & (y_soc <= 999999)
    if cfg.missing_soc_values:
        keep = keep & (~y_soc.isin(list(cfg.missing_soc_values)))

    df = train_df.loc[keep].copy()
    y_soc = y_soc.loc[keep].astype(int)

    if title_col in df.columns:
        has_title = df[title_col].map(lambda v: len(normalize_text(v)) > 0)
        df = df.loc[has_title].copy()
        y_soc = y_soc.loc[has_title].astype(int)

    if len(df) < 100:
        raise ValueError(f"Not enough labeled rows for {y_col}: {len(df)}")

    y_major = y_soc.map(soc6_to_major2)
    keep_major = y_major.notna() & y_major.isin(list(VALID_SOC_MAJOR_GROUPS))
    if (~keep_major).sum():
        eprint(f"[{y_col}] Dropping {int((~keep_major).sum())} rows with invalid SOC major group")
    df = df.loc[keep_major].copy()
    y_soc = y_soc.loc[keep_major].astype(int)
    y_major = y_major.loc[keep_major].astype(int)

    major_counts = y_major.value_counts()
    rare_major = major_counts[major_counts < 3].index.tolist()
    if rare_major:
        eprint(f"[{y_col}] Dropping rare major groups (<3 rows): " + ", ".join(f"{g:02d}" for g in rare_major))
        keep2 = ~y_major.isin(rare_major)
        df = df.loc[keep2].copy()
        y_soc = y_soc.loc[keep2].astype(int)
        y_major = y_major.loc[keep2].astype(int)

    eprint(f"\n[{y_col}] Labeled rows used for SOC training: {len(df)}")
    eprint(f"[{y_col}] Unique major groups: {y_major.nunique()} | Unique SOC6: {y_soc.nunique()}")

    dictionary = build_soc_title_naics_dictionary(
        df=df,
        title_col=title_col,
        naics_col=naics_col,
        y_soc=y_soc.to_numpy(dtype=np.int32),
        min_count=cfg.dict_min_count,
        cfg=cfg,
    )
    eprint(f"[{y_col}] SOC exact dictionary entries: {len(dictionary.get('exact', {}))}")

    semantic_memory = build_semantic_memory(
        name=f"{y_col}_SOC_semantic_memory",
        df=df,
        labels=y_soc.to_numpy(dtype=np.int32),
        text_cols=text_cols,
        prefixes=prefixes,
        cfg=cfg,
        kind="soc",
        reference_csv=cfg.soc_reference_csv,
    )

    feat = HybridFeaturizer(
        text_cols=[c for c in text_cols if c in df.columns],
        cat_cols=[c for c in cat_cols if c in df.columns],
        num_cols=[c for c in num_cols if c in df.columns],
        prefixes=prefixes,
        cfg=cfg,
    )
    eprint(f"[{y_col}] Building features...")
    x = feat.fit_transform(df)
    eprint(f"[{y_col}] Feature matrix: {x.shape[0]} rows x {x.shape[1]} cols | nnz={x.nnz:,} | approx_mem={sparse_nbytes(x)/1024**3:.2f} GB")

    with threadpool_limits(limits=parallel_cfg.native_threads):
        major_model = fit_best_classifier(
            x=x,
            y=y_major.to_numpy(dtype=np.int32),
            cfg=cfg,
            stage_name=f"{y_col}_major",
            nfolds=cfg.nfolds_major,
            features_nonnegative=feat.features_are_nonnegative_,
            parallel_cfg=parallel_cfg,
            optuna_trials=cfg.optuna_trials_major,
        )
    eprint(f"[{y_col}] Major-group selected model(s): {major_model.model_name}")

    within_models: Dict[str, FittedClassifier] = {}
    within_baselines: Dict[str, Dict[str, Any]] = {}
    y_major_np = y_major.to_numpy(dtype=np.int32)
    y_soc_np = y_soc.to_numpy(dtype=np.int32)
    major_classes = sorted(np.unique(y_major_np).tolist())

    def _train_one_group(g: int) -> Tuple[str, str, Any, str]:
        try:
            idx_g = np.where(y_major_np == g)[0]
            y_g = y_soc_np[idx_g]
            counts = Counter(y_g.tolist())
            keep_codes = sorted([soc for soc, n in counts.items() if n >= cfg.min_soc_n])
            if len(keep_codes) < 2:
                ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                total = float(sum(n for _, n in ordered))
                baseline = {"soc": [int(soc) for soc, _ in ordered[:75]], "prob": [float(n / total) for _, n in ordered[:75]]}
                return str(g), "baseline", baseline, f"[{y_col}] Major {g:02d}: baseline only (unique={len(counts)}, kept={len(keep_codes)})"

            keep_mask = np.isin(y_g, np.array(keep_codes, dtype=np.int32))
            keep_rows = idx_g[keep_mask]
            y_train_g = y_soc_np[keep_rows]
            if len(keep_rows) < 50 or len(np.unique(y_train_g)) < 2:
                ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                total = float(sum(n for _, n in ordered))
                baseline = {"soc": [int(soc) for soc, _ in ordered[:75]], "prob": [float(n / total) for _, n in ordered[:75]]}
                return str(g), "baseline", baseline, f"[{y_col}] Major {g:02d}: baseline only (too small after filtering)"

            x_g = x[keep_rows]
            model = fit_best_classifier(
                x=x_g,
                y=y_train_g.astype(np.int32),
                cfg=cfg,
                stage_name=f"{y_col}_within_{g:02d}",
                nfolds=cfg.nfolds_within,
                features_nonnegative=feat.features_are_nonnegative_,
                parallel_cfg=parallel_cfg,
                optuna_trials=cfg.optuna_trials_within,
            )
            msg = f"[{y_col}] Major {g:02d}: trained {model.model_name} (rows={x_g.shape[0]}, classes={model.n_classes}"
            if model.cv_macro_f1 is not None:
                msg += f", CV macro-F1={model.cv_macro_f1:.4f}"
            msg += ")"
            return str(g), "model", model, msg
        except Exception as exc:
            return str(g), "error", None, f"[{y_col}] Major {g:02d}: training failed, no baseline/model created: {exc}"

    with threadpool_limits(limits=parallel_cfg.native_threads):
        if parallel_cfg.max_workers > 1 and len(major_classes) > 1:
            results = Parallel(n_jobs=parallel_cfg.max_workers, prefer="threads")(
                delayed(_train_one_group)(g) for g in major_classes
            )
        else:
            results = [_train_one_group(g) for g in major_classes]

    for gkey, kind, payload, msg in results:
        eprint(msg)
        if kind == "model" and payload is not None:
            within_models[gkey] = payload
        elif kind == "baseline" and payload is not None:
            within_baselines[gkey] = payload

    del x
    gc.collect()
    return {
        "version": VERSION,
        "kind": "soc_hierarchical",
        "target_col": y_col,
        "title_col": title_col,
        "naics_col": naics_col,
        "dictionary": dictionary,
        "semantic_memory": semantic_memory,
        "featurizer": feat,
        "major_model": major_model,
        "within_models": within_models,
        "within_baselines": within_baselines,
        "text_cols": list(text_cols),
        "cat_cols": list(cat_cols),
        "num_cols": list(num_cols),
    }


def predict_soc_target_topk(model_obj: Dict[str, Any], score_df: pd.DataFrame, cfg: TrainConfig) -> Dict[str, Any]:
    n = len(score_df)
    top_soc = np.full((n, cfg.topk_soc), -1, dtype=np.int32)
    top_prob = np.zeros((n, cfg.topk_soc), dtype=np.float32)

    d = soc_dictionary_predict(
        df=score_df,
        dictionary=model_obj.get("dictionary", {}),
        title_col=model_obj["title_col"],
        naics_col=model_obj["naics_col"],
        purity=cfg.dict_purity,
        min_count=cfg.dict_min_count,
        cfg=cfg,
    )
    for i in np.where(d["used"])[0].tolist():
        top_soc[i, 0] = int(d["pred"][i])
        top_prob[i, 0] = float(d["conf"][i])
        for j, alt in enumerate(d["alts"][i][:max(0, cfg.topk_soc - 1)]):
            top_soc[i, j + 1] = int(alt["soc"])
            top_prob[i, j + 1] = float(alt["prob"])

    title_col = model_obj["title_col"]
    if title_col in score_df.columns:
        title_present = score_df[title_col].map(lambda v: len(normalize_text(v)) > 0).to_numpy(dtype=bool)
    else:
        title_present = np.zeros(n, dtype=bool)
    rows_model = np.where((~d["used"]) & title_present)[0]
    text_cols = model_obj.get("text_cols", [title_col])
    if rows_model.size == 0:
        pred = {"top_soc": top_soc, "top_prob": top_prob, "dict_used": d["used"], "title_present": title_present}
        return maybe_llm_rerank_soc(pred, score_df, cfg, text_cols)

    # Semantic vector memory for synonym-aware SOC candidates. It is fused with
    # hierarchical ML candidates when close neighbors are found.
    sem_codes_global: List[Optional[List[Any]]] = [None] * n
    sem_probs_global: List[Optional[List[float]]] = [None] * n
    sem_conf_global = np.zeros(n, dtype=np.float32)
    sem_sim_global = np.zeros(n, dtype=np.float32)
    sem_mem: Optional[SemanticMemory] = model_obj.get("semantic_memory")
    if sem_mem is not None:
        try:
            sem_docs = make_docs_from_columns(score_df.iloc[rows_model], text_cols, model_obj.get("featurizer").prefixes if model_obj.get("featurizer") is not None else {})
            sem = sem_mem.predict_topk(sem_docs, topk=cfg.topk_soc, k=cfg.semantic_k)
            for local_i, global_i in enumerate(rows_model.tolist()):
                sem_codes_global[global_i] = sem["codes"][local_i]
                sem_probs_global[global_i] = sem["prob"][local_i].tolist()
                sem_conf_global[global_i] = float(sem["conf"][local_i])
                sem_sim_global[global_i] = float(sem["max_sim"][local_i])
        except Exception as exc:
            eprint(f"[{model_obj.get('target_col', 'SOC')}] Semantic memory prediction skipped after error: {exc}")

    feat: HybridFeaturizer = model_obj["featurizer"]
    major_model: FittedClassifier = model_obj["major_model"]
    major_classes = np.asarray(major_model.classes)
    kG = min(cfg.top_groups, len(major_classes))

    for batch in batched_indices(rows_model, cfg.score_batch_size):
        xs = feat.transform(score_df.iloc[batch])
        major_proba = major_model.predict_proba(xs)
        top_major_idx = np.vstack([safe_top_indices(r, kG) for r in major_proba])
        top_major_codes = major_classes[top_major_idx]

        grouped: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for g in np.unique(top_major_codes).tolist():
            local_rows = np.where((top_major_codes == g).any(axis=1))[0]
            if local_rows.size == 0:
                continue
            global_rows = batch[local_rows]
            col_g = int(np.where(major_classes == g)[0][0])
            p_major = major_proba[local_rows, col_g]
            gkey = str(int(g))
            wm: Optional[FittedClassifier] = model_obj["within_models"].get(gkey)
            if wm is not None:
                xg = xs[local_rows]
                p_within = wm.predict_proba(xg)
                codes_g = np.asarray(wm.classes)
                for ridx in range(p_within.shape[0]):
                    ord_idx = safe_top_indices(p_within[ridx], min(cfg.topk_soc, p_within.shape[1]))
                    for o in ord_idx.tolist():
                        grouped[int(global_rows[ridx])].append((int(codes_g[o]), float(p_within[ridx, o] * p_major[ridx])))
            else:
                base = model_obj["within_baselines"].get(gkey)
                if base is None:
                    continue
                m = min(cfg.topk_soc, len(base["soc"]))
                for ridx, global_row in enumerate(global_rows.tolist()):
                    for soc, pr in zip(base["soc"][:m], base["prob"][:m]):
                        grouped[int(global_row)].append((int(soc), float(p_major[ridx] * pr)))

        for row in batch.tolist():
            ml_merged: Dict[int, float] = {}
            for soc, pr in grouped.get(int(row), []):
                ml_merged[int(soc)] = max(ml_merged.get(int(soc), 0.0), float(pr))
            ml_ordered = sorted(ml_merged.items(), key=lambda kv: kv[1], reverse=True)[:cfg.topk_soc]
            ml_codes = [soc for soc, _ in ml_ordered]
            ml_probs = [pr for _, pr in ml_ordered]
            codes, probs = merge_ranked_candidates(
                ml_codes=ml_codes,
                ml_probs=ml_probs,
                sem_codes=sem_codes_global[int(row)],
                sem_probs=sem_probs_global[int(row)],
                sem_conf=float(sem_conf_global[int(row)]),
                sem_sim=float(sem_sim_global[int(row)]),
                cfg=cfg,
                topk=cfg.topk_soc,
                code_kind="soc",
            )
            for rank in range(cfg.topk_soc):
                top_soc[int(row), rank] = int(codes[rank]) if codes[rank] is not None else -1
                top_prob[int(row), rank] = float(probs[rank])
        del xs
        gc.collect()

    pred = {"top_soc": top_soc, "top_prob": top_prob, "dict_used": d["used"], "title_present": title_present}
    return maybe_llm_rerank_soc(pred, score_df, cfg, text_cols)


# ---------------------------------------------------------------------
# SPSS helpers
# ---------------------------------------------------------------------

def read_sav_with_meta(path: str) -> Tuple[pd.DataFrame, Any]:
    return pyreadstat.read_sav(path, user_missing=True, apply_value_formats=False)


def write_sav_preserving_metadata(df: pd.DataFrame, meta: Any, path: str) -> None:
    cols = list(df.columns)
    pyreadstat.write_sav(
        df,
        path,
        file_label=getattr(meta, "file_label", "") or "",
        column_labels=filter_meta_dict(getattr(meta, "column_names_to_labels", None), cols),
        variable_value_labels=filter_meta_dict(getattr(meta, "variable_value_labels", None), cols),
        missing_ranges=filter_meta_dict(getattr(meta, "missing_ranges", None), cols),
        variable_display_width=filter_meta_dict(getattr(meta, "variable_display_width", None), cols),
        variable_measure=filter_meta_dict(getattr(meta, "variable_measure", None), cols),
        variable_format=filter_meta_dict(getattr(meta, "original_variable_types", None), cols),
        note=getattr(meta, "notes", None),
        compress=False,
    )


def assign_code_predictions(out_df: pd.DataFrame, col: str, apply_mask: np.ndarray, preds: Sequence[Optional[str]]) -> None:
    if col not in out_df.columns:
        return
    idx = np.where(apply_mask)[0]
    if len(idx) == 0:
        return
    if pd.api.types.is_numeric_dtype(out_df[col]):
        vals = pd.to_numeric(pd.Series([preds[i] for i in idx]), errors="coerce").to_numpy()
        out_df.loc[out_df.index[idx], col] = vals
    else:
        vals = ["" if preds[i] is None else str(preds[i]) for i in idx]
        out_df.loc[out_df.index[idx], col] = vals


def assign_soc_predictions(out_df: pd.DataFrame, col: str, apply_mask: np.ndarray, preds: np.ndarray) -> None:
    if col not in out_df.columns:
        return
    idx = np.where(apply_mask)[0]
    if len(idx) == 0:
        return
    out_df.loc[out_df.index[idx], col] = preds[idx].astype(float)


# ---------------------------------------------------------------------
# Column plan and logging
# ---------------------------------------------------------------------

def build_prefixes(opt: argparse.Namespace) -> Dict[str, str]:
    prefixes: Dict[str, str] = {
        opt.qe4_col: "JOBTITLE:",
        opt.qe5_col: "UNUSEDSKILL_TITLE:",
        "qa20": "EMPLOYER:",
        opt.naics_col: "NAICS:",
        "qe1oth": "INDUSTRY_OTHER:",
        "qe8aoth": "SELFEMP_OTHER:",
        "qe2b": "WORKCITY:",
        "qa9a": "ED_FIELD:",
        "qa9c1": "CERT:",
        "qa9c2": "CERT:",
        "qa9c3": "CERT:",
        "qa9c4": "CERT:",
        "qa9g": "LICENSE:",
        "qe13oth": "PAYTYPE_OTHER:",
    }
    return prefixes


def column_plan(opt: argparse.Namespace) -> Dict[str, List[str]]:
    common_text_cols = ["qa20", "qe1oth", "qe8aoth", "qe2b", "qa9a", "qa9c1", "qa9c2", "qa9c3", "qa9c4", "qa9g", "qe13oth"]
    plan = {
        "naics_text": ["qa20", opt.qe4_col, opt.qe5_col, "qe1oth", "qe8aoth", "qe2b"],
        "naics_cat": ["qe1", "qse1", "qe8a", "wh2", "qe8", "qa9", "qa9ar", "qe13", "qa6"],
        "naics_num": ["qe10"],
        "qe4_text": [opt.qe4_col] + common_text_cols + [opt.naics_col],
        "qe4_cat": [opt.naics_col, "qe1", "qse1", "qe8a", "wh2", "qe8", "qa9", "qa9ar", "qe13", "qa6"],
        "qe4_num": ["qe10"],
        "qe5_text": [opt.qe5_col] + common_text_cols + [opt.naics_col],
        "qe5_cat": [opt.naics_col, "qe1", "qa9", "qa9ar"],
        "qe5_num": ["qe10"],
    }
    return plan


def classifier_summary(clf: FittedClassifier) -> Dict[str, Any]:
    return {
        "stage": clf.stage_name,
        "models": clf.model_names,
        "weights": clf.weights,
        "classes": int(clf.n_classes),
        "rows": int(clf.n_rows),
        "features": int(clf.n_features),
        "cv_macro_f1": clf.cv_macro_f1,
        "cv_log_loss": clf.cv_log_loss,
        "candidate_scores": [asdict(s) for s in clf.cv_scores],
    }


def bundle_summary(bundle: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "version": bundle.get("version"),
        "created_at": bundle.get("created_at"),
        "platform": bundle.get("platform"),
        "package_availability": bundle.get("package_availability"),
        "config": bundle.get("config"),
    }
    if "naics_model" in bundle:
        out["naics"] = classifier_summary(bundle["naics_model"]["classifier"])
    for key in ["soc_qe4_model", "soc_qe5_model"]:
        if key in bundle:
            m = bundle[key]
            out[key] = {
                "major": classifier_summary(m["major_model"]),
                "within_model_count": len(m.get("within_models", {})),
                "within_baseline_count": len(m.get("within_baselines", {})),
            }
    return out


def write_log(
    path: str,
    opt: argparse.Namespace,
    cfg: TrainConfig,
    bundle: Dict[str, Any],
    score_df: pd.DataFrame,
    before_naics: pd.Series,
    after_naics: pd.Series,
    pred_naics: Dict[str, Any],
    apply_naics: np.ndarray,
    before4: pd.Series,
    after4: pd.Series,
    pred4: Dict[str, Any],
    apply4: np.ndarray,
    before5: pd.Series,
    after5: pd.Series,
    pred5: Dict[str, Any],
    apply5: np.ndarray,
) -> None:
    n = len(score_df)
    id_vec = score_df[opt.id_col].tolist() if opt.id_col in score_df.columns else list(range(1, n + 1))
    lines: List[str] = []
    lines.append("NAICS-first SOC Autocode Log (state-of-art ensemble)")
    lines.append("====================================================")
    lines.append(f"Created: {now_stamp()}")
    lines.append(f"Script version: {VERSION}")
    lines.append(f"Mode: {opt.mode}")
    lines.append(f"Profile: {cfg.profile}")
    lines.append(f"Package availability: {bundle.get('package_availability')}")
    lines.append(f"Output .sav: {opt.out_sav}")
    lines.append("")
    lines.append("Model summary:")
    lines.append(json.dumps(bundle_summary(bundle), indent=2, default=str))
    lines.append("")
    lines.append("Row-level predictions:")

    top_naics = pred_naics["top_code"]
    prob_naics = pred_naics["top_prob"]
    top4 = pred4["top_soc"]
    pr4 = pred4["top_prob"]
    top5 = pred5["top_soc"]
    pr5 = pred5["top_prob"]

    for i in range(n):
        naics_alts: List[str] = []
        for k in range(1, cfg.topk_naics):
            code = top_naics[i][k]
            if code is not None:
                naics_alts.append(f"{code} ({prob_naics[i, k] * 100:.1f}%)")
        soc4_alts: List[str] = []
        for k in range(1, cfg.topk_soc):
            if top4[i, k] > 0:
                soc4_alts.append(f"{top4[i, k]:06d} ({pr4[i, k] * 100:.1f}%)")
        soc5_alts: List[str] = []
        for k in range(1, cfg.topk_soc):
            if top5[i, k] > 0:
                soc5_alts.append(f"{top5[i, k]:06d} ({pr5[i, k] * 100:.1f}%)")

        line = (
            f"id={id_vec[i]} | "
            f"{opt.naics_col}: {format_code(before_naics.iloc[i])} -> {format_code(after_naics.iloc[i])} "
            f"(model_top={top_naics[i][0] or 'MISSING'} conf={prob_naics[i,0]*100:.1f}% applied={'YES' if apply_naics[i] else 'NO'}) "
            f"[alts: {', '.join(naics_alts)}] | "
            f"{opt.qe4ar_col}: {format_soc(before4.iloc[i])} -> {format_soc(after4.iloc[i])} "
            f"(model_top={format_soc(top4[i,0]) if top4[i,0] > 0 else 'MISSING'} conf={pr4[i,0]*100:.1f}% applied={'YES' if apply4[i] else 'NO'}) "
            f"[alts: {', '.join(soc4_alts)}] | "
            f"{opt.qe5ar_col}: {format_soc(before5.iloc[i])} -> {format_soc(after5.iloc[i])} "
            f"(model_top={format_soc(top5[i,0]) if top5[i,0] > 0 else 'MISSING'} conf={pr5[i,0]*100:.1f}% applied={'YES' if apply5[i] else 'NO'}) "
            f"[alts: {', '.join(soc5_alts)}]"
        )
        lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


# ---------------------------------------------------------------------
# Training and scoring orchestration
# ---------------------------------------------------------------------

def package_availability() -> Dict[str, bool]:
    return {
        "lightgbm": HAS_LIGHTGBM,
        "xgboost": HAS_XGBOOST,
        "catboost": HAS_CATBOOST,
        "optuna": HAS_OPTUNA,
        "sentence_transformers": HAS_SENTENCE_TRANSFORMERS,
        "rapidfuzz": HAS_RAPIDFUZZ,
        "faiss": HAS_FAISS,
        "llama_cpp": HAS_LLAMA_CPP,
        "psutil": HAS_PSUTIL,
    }


def train_full_bundle(train_df: pd.DataFrame, opt: argparse.Namespace, cfg: TrainConfig, parallel_cfg: ParallelConfig) -> Dict[str, Any]:
    prefixes = build_prefixes(opt)
    plan = column_plan(opt)

    eprint("\nTraining NAICS model first...")
    naics_model = train_naics_target(
        train_df=train_df,
        target_col=opt.naics_col,
        text_cols=plan["naics_text"],
        cat_cols=plan["naics_cat"],
        num_cols=plan["naics_num"],
        prefixes=prefixes,
        cfg=cfg,
        parallel_cfg=parallel_cfg,
    )

    # Fill missing NAICS in training data before SOC model training. This makes
    # SOC training mimic production scoring: NAICS is available first.
    train_for_soc = train_df.copy()
    # Do not spend local-LLM cycles on the internal training-data NAICS fill.
    _llm_flag = cfg.use_local_llm_fallback
    cfg.use_local_llm_fallback = False
    try:
        train_naics_pred = predict_naics_topk(naics_model, train_for_soc, cfg)
    finally:
        cfg.use_local_llm_fallback = _llm_flag
    before_train_naics = train_for_soc[opt.naics_col] if opt.naics_col in train_for_soc.columns else pd.Series([np.nan] * len(train_for_soc))
    train_naics_missing = is_missing_code_series(before_train_naics, cfg.missing_naics_values).to_numpy(dtype=bool)
    train_naics_top = [row[0] for row in train_naics_pred["top_code"]]
    train_naics_conf = train_naics_pred["top_prob"][:, 0]
    fill_train_naics = train_naics_missing & np.array([c is not None for c in train_naics_top]) & (train_naics_conf >= cfg.min_write_confidence_naics)
    assign_code_predictions(train_for_soc, opt.naics_col, fill_train_naics, train_naics_top)
    eprint(f"[NAICS] Filled {int(fill_train_naics.sum())} missing training NAICS values for SOC training context")

    eprint("\nTraining SOC model for qe4ar...")
    soc_qe4_model = train_hierarchical_soc_target(
        train_df=train_for_soc,
        y_col=opt.qe4ar_col,
        title_col=opt.qe4_col,
        naics_col=opt.naics_col,
        text_cols=plan["qe4_text"],
        cat_cols=plan["qe4_cat"],
        num_cols=plan["qe4_num"],
        prefixes=prefixes,
        cfg=cfg,
        parallel_cfg=parallel_cfg,
    )

    eprint("\nTraining SOC model for qe5ar...")
    soc_qe5_model = train_hierarchical_soc_target(
        train_df=train_for_soc,
        y_col=opt.qe5ar_col,
        title_col=opt.qe5_col,
        naics_col=opt.naics_col,
        text_cols=plan["qe5_text"],
        cat_cols=plan["qe5_cat"],
        num_cols=plan["qe5_num"],
        prefixes=prefixes,
        cfg=cfg,
        parallel_cfg=parallel_cfg,
    )

    bundle = {
        "version": VERSION,
        "created_at": now_stamp(),
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "package_availability": package_availability(),
        "config": asdict(cfg),
        "columns": vars(opt),
        "naics_model": naics_model,
        "soc_qe4_model": soc_qe4_model,
        "soc_qe5_model": soc_qe5_model,
    }
    return bundle


def score_with_bundle(score_df: pd.DataFrame, bundle: Dict[str, Any], opt: argparse.Namespace, cfg: TrainConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out_df = score_df.copy()
    n = len(out_df)

    eprint("\nScoring NAICS first...")
    pred_naics = predict_naics_topk(bundle["naics_model"], out_df, cfg)
    before_naics = out_df[opt.naics_col].copy() if opt.naics_col in out_df.columns else pd.Series([np.nan] * n)
    top_naics = [row[0] for row in pred_naics["top_code"]]
    conf_naics = pred_naics["top_prob"][:, 0]
    has_naics_pred = np.array([c is not None for c in top_naics], dtype=bool)
    if opt.overwrite_existing or opt.overwrite_existing_naics:
        apply_naics = has_naics_pred & (conf_naics >= cfg.min_write_confidence_naics)
    else:
        apply_naics = is_missing_code_series(before_naics, cfg.missing_naics_values).to_numpy(dtype=bool) & has_naics_pred & (conf_naics >= cfg.min_write_confidence_naics)
    assign_code_predictions(out_df, opt.naics_col, apply_naics, top_naics)
    after_naics = out_df[opt.naics_col].copy() if opt.naics_col in out_df.columns else before_naics
    eprint(f"[NAICS] Applied predictions to {int(apply_naics.sum())} rows")

    eprint("\nScoring qe4ar SOC using updated NAICS...")
    pred4 = predict_soc_target_topk(bundle["soc_qe4_model"], out_df, cfg)
    before4 = score_df[opt.qe4ar_col].copy() if opt.qe4ar_col in score_df.columns else pd.Series([np.nan] * n)
    pred4_top = pred4["top_soc"][:, 0]
    conf4_top = pred4["top_prob"][:, 0]
    has4 = pred4_top > 0
    if opt.overwrite_existing or opt.overwrite_existing_soc:
        apply4 = has4 & (conf4_top >= cfg.min_write_confidence_soc)
    else:
        apply4 = is_missing_soc_series(before4, cfg.missing_soc_values).to_numpy(dtype=bool) & has4 & (conf4_top >= cfg.min_write_confidence_soc)
    assign_soc_predictions(out_df, opt.qe4ar_col, apply4, pred4_top)
    after4 = out_df[opt.qe4ar_col].copy() if opt.qe4ar_col in out_df.columns else before4
    eprint(f"[{opt.qe4ar_col}] Applied predictions to {int(apply4.sum())} rows")

    eprint("\nScoring qe5ar SOC using updated NAICS...")
    pred5 = predict_soc_target_topk(bundle["soc_qe5_model"], out_df, cfg)
    before5 = score_df[opt.qe5ar_col].copy() if opt.qe5ar_col in score_df.columns else pd.Series([np.nan] * n)
    pred5_top = pred5["top_soc"][:, 0]
    conf5_top = pred5["top_prob"][:, 0]
    has5 = pred5_top > 0
    if opt.overwrite_existing or opt.overwrite_existing_soc:
        apply5 = has5 & (conf5_top >= cfg.min_write_confidence_soc)
    else:
        apply5 = is_missing_soc_series(before5, cfg.missing_soc_values).to_numpy(dtype=bool) & has5 & (conf5_top >= cfg.min_write_confidence_soc)
    assign_soc_predictions(out_df, opt.qe5ar_col, apply5, pred5_top)
    after5 = out_df[opt.qe5ar_col].copy() if opt.qe5ar_col in out_df.columns else before5
    eprint(f"[{opt.qe5ar_col}] Applied predictions to {int(apply5.sum())} rows")

    details = {
        "before_naics": before_naics,
        "after_naics": after_naics,
        "pred_naics": pred_naics,
        "apply_naics": apply_naics,
        "before4": before4,
        "after4": after4,
        "pred4": pred4,
        "apply4": apply4,
        "before5": before5,
        "after5": after5,
        "pred5": pred5,
        "apply5": apply5,
    }
    return out_df, details


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def apply_profile_defaults(opt: argparse.Namespace) -> argparse.Namespace:
    # Only fill unset values. argparse sets explicit defaults, so this function
    # is mostly used to harmonize profile-level choices after parsing.
    if opt.profile == "fast":
        opt.model_search = opt.model_search or "light"
        opt.use_embeddings = False if opt.use_embeddings is None else opt.use_embeddings
        opt.use_semantic_memory = False if opt.use_semantic_memory is None else opt.use_semantic_memory
        opt.svd_components = 0 if opt.svd_components is None else opt.svd_components
        opt.ensemble_size = min(opt.ensemble_size, 1)
        opt.optuna_trials_naics = 0
        opt.optuna_trials_major = 0
        opt.optuna_trials_within = 0
    elif opt.profile == "robust":
        opt.model_search = opt.model_search or "robust"
        if opt.svd_components is None:
            opt.svd_components = 64
        if opt.use_embeddings is None:
            opt.use_embeddings = False
        if opt.use_semantic_memory is None:
            opt.use_semantic_memory = True
    else:  # accuracy
        opt.model_search = opt.model_search or "max"
        if opt.svd_components is None:
            opt.svd_components = 128
        if opt.use_embeddings is None:
            opt.use_embeddings = True
        if opt.use_semantic_memory is None:
            opt.use_semantic_memory = True
        if opt.optuna_trials_naics is None:
            opt.optuna_trials_naics = 12
        if opt.optuna_trials_major is None:
            opt.optuna_trials_major = 8
        if opt.optuna_trials_within is None:
            # Usually too costly to tune every SOC major group. Set higher only if
            # the machine will be left to chew overnight.
            opt.optuna_trials_within = 0
    return opt


def build_config(opt: argparse.Namespace) -> Tuple[TrainConfig, ParallelConfig]:
    opt = apply_profile_defaults(opt)
    parallel_cfg = ParallelConfig(
        max_workers=recommend_max_workers(opt.max_workers),
        native_threads=max(1, int(opt.native_threads)),
    )
    cfg = TrainConfig(
        profile=str(opt.profile),
        random_seed=int(opt.random_seed),
        dict_purity=float(opt.dict_purity),
        dict_min_count=int(opt.dict_min_count),
        naics_dict_purity=float(opt.naics_dict_purity),
        naics_dict_min_count=int(opt.naics_dict_min_count),
        use_fuzzy_dictionary=bool(opt.use_fuzzy_dictionary),
        fuzzy_threshold=int(opt.fuzzy_threshold),
        fuzzy_max_choices_per_bucket=int(opt.fuzzy_max_choices_per_bucket),
        use_semantic_memory=bool(opt.use_semantic_memory),
        semantic_backend=str(opt.semantic_backend),
        semantic_k=int(opt.semantic_k),
        semantic_min_similarity=float(opt.semantic_min_similarity),
        semantic_min_confidence=float(opt.semantic_min_confidence),
        semantic_fusion_weight=float(opt.semantic_fusion_weight),
        ml_fusion_weight=float(opt.ml_fusion_weight),
        semantic_reference_weight=float(opt.semantic_reference_weight),
        naics_reference_csv=str(opt.naics_reference_csv or ""),
        soc_reference_csv=str(opt.soc_reference_csv or ""),
        use_local_llm_fallback=bool(opt.use_local_llm_fallback),
        llm_model_path=str(opt.llm_model_path or ""),
        llm_trigger_confidence_naics=float(opt.llm_trigger_confidence_naics),
        llm_trigger_confidence_soc=float(opt.llm_trigger_confidence_soc),
        llm_max_rows=int(opt.llm_max_rows),
        llm_threads=int(opt.llm_threads),
        llm_ctx_size=int(opt.llm_ctx_size),
        llm_temperature=float(opt.llm_temperature),
        llm_top_p=float(opt.llm_top_p),
        llm_max_tokens=int(opt.llm_max_tokens),
        llm_rerank_confidence=float(opt.llm_rerank_confidence),
        llm_allow_free_code=bool(opt.llm_allow_free_code),
        topk_naics=int(opt.topk_naics),
        topk_soc=int(opt.topk_soc),
        top_groups=int(opt.top_groups),
        min_naics_n=int(opt.min_naics_n),
        min_soc_n=int(opt.min_soc_n),
        nfolds_naics=int(opt.nfolds_naics),
        nfolds_major=int(opt.nfolds_major),
        nfolds_within=int(opt.nfolds_within),
        model_search=str(opt.model_search),
        ensemble_size=int(opt.ensemble_size),
        ensemble_power=float(opt.ensemble_power),
        optuna_trials_naics=int(opt.optuna_trials_naics or 0),
        optuna_trials_major=int(opt.optuna_trials_major or 0),
        optuna_trials_within=int(opt.optuna_trials_within or 0),
        optuna_timeout_seconds=opt.optuna_timeout_seconds,
        max_word_terms=int(opt.max_word_terms),
        max_char_terms=int(opt.max_char_terms),
        word_min_count=int(opt.word_min_count),
        char_min_count=int(opt.char_min_count),
        doc_prop_max=float(opt.doc_prop_max),
        use_embeddings=bool(opt.use_embeddings),
        embedding_model=str(opt.embedding_model),
        embedding_backend=str(opt.embedding_backend),
        embedding_batch_size=int(opt.embedding_batch_size),
        embedding_device=opt.embedding_device,
        svd_components=int(opt.svd_components or 0),
        logreg_C=float(opt.logreg_C),
        logreg_max_iter=int(opt.logreg_max_iter),
        sgd_alpha=float(opt.sgd_alpha),
        sgd_max_iter=int(opt.sgd_max_iter),
        sgd_tol=float(opt.sgd_tol),
        linearsvc_C=float(opt.linearsvc_C),
        boost_n_estimators=int(opt.boost_n_estimators),
        boost_max_classes=int(opt.boost_max_classes),
        boost_max_features=int(opt.boost_max_features),
        score_batch_size=int(opt.score_batch_size),
        min_write_confidence_naics=float(opt.min_write_confidence_naics),
        min_write_confidence_soc=float(opt.min_write_confidence_soc),
        missing_soc_values=parse_csv_ints(opt.missing_soc_values),
        missing_naics_values=parse_csv_strs(opt.missing_naics_values),
    )
    if cfg.use_embeddings and not HAS_SENTENCE_TRANSFORMERS:
        eprint("WARNING: --use_embeddings requested but sentence-transformers is unavailable. Disabling embeddings.")
        cfg.use_embeddings = False
    if cfg.use_semantic_memory and not HAS_SENTENCE_TRANSFORMERS:
        eprint("WARNING: --use_semantic_memory requested but sentence-transformers is unavailable. Disabling semantic memory.")
        cfg.use_semantic_memory = False
    if cfg.semantic_backend == "faiss" and not HAS_FAISS:
        eprint("WARNING: --semantic_backend faiss requested but faiss is unavailable. Falling back to sklearn cosine search.")
        cfg.semantic_backend = "sklearn"
    if cfg.use_local_llm_fallback and not HAS_LLAMA_CPP:
        eprint("WARNING: --use_local_llm_fallback requested but llama-cpp-python is unavailable. Disabling local LLM fallback.")
        cfg.use_local_llm_fallback = False
    if cfg.use_fuzzy_dictionary and not HAS_RAPIDFUZZ:
        eprint("WARNING: --use_fuzzy_dictionary requested but rapidfuzz is unavailable. Disabling fuzzy dictionary.")
        cfg.use_fuzzy_dictionary = False
    if cfg.model_search == "max" and not any([HAS_LIGHTGBM, HAS_XGBOOST, HAS_CATBOOST]):
        eprint("WARNING: model_search=max requested, but no gradient boosting packages were found. Sparse linear models will still be used.")
    return cfg, parallel_cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NAICS-first + SOC state-of-art autocoder for SPSS .sav files")
    p.add_argument("--mode", choices=["train", "score", "train_and_score"], default="train_and_score")
    p.add_argument("--train_sav", type=str, default="")
    p.add_argument("--score_sav", type=str, default="")
    p.add_argument("--out_sav", type=str, default="")
    p.add_argument("--log_txt", type=str, default="")
    p.add_argument("--model_out", type=str, default="")
    p.add_argument("--model_in", type=str, default="")
    p.add_argument("--save_model", action="store_true", default=False)

    # Columns
    p.add_argument("--id_col", type=str, default="id")
    p.add_argument("--naics_col", type=str, default="qa20a")
    p.add_argument("--qe4_col", type=str, default="qe4")
    p.add_argument("--qe4ar_col", type=str, default="qe4ar")
    p.add_argument("--qe5_col", type=str, default="qe5a")
    p.add_argument("--qe5ar_col", type=str, default="qe5ar")

    # Write behavior
    p.add_argument("--overwrite_existing", action="store_true", default=False)
    p.add_argument("--overwrite_existing_naics", action="store_true", default=False)
    p.add_argument("--overwrite_existing_soc", action="store_true", default=False)
    p.add_argument("--missing_soc_values", type=str, default="")
    p.add_argument("--missing_naics_values", type=str, default="")
    p.add_argument("--min_write_confidence_naics", type=float, default=0.0)
    p.add_argument("--min_write_confidence_soc", type=float, default=0.0)

    # Profiles/search
    p.add_argument("--profile", choices=["fast", "robust", "accuracy"], default="accuracy")
    p.add_argument("--model_search", choices=["none", "light", "robust", "max"], default=None)
    p.add_argument("--ensemble_size", type=int, default=3)
    p.add_argument("--ensemble_power", type=float, default=2.0)
    p.add_argument("--optuna_trials_naics", type=int, default=None)
    p.add_argument("--optuna_trials_major", type=int, default=None)
    p.add_argument("--optuna_trials_within", type=int, default=None)
    p.add_argument("--optuna_timeout_seconds", type=int, default=None)

    # Dictionaries
    p.add_argument("--dict_purity", type=float, default=0.90)
    p.add_argument("--dict_min_count", type=int, default=5)
    p.add_argument("--naics_dict_purity", type=float, default=0.90)
    p.add_argument("--naics_dict_min_count", type=int, default=5)
    p.add_argument("--use_fuzzy_dictionary", action="store_true", default=False)
    p.add_argument("--fuzzy_threshold", type=int, default=93)
    p.add_argument("--fuzzy_max_choices_per_bucket", type=int, default=25000)

    # Semantic vector memory / dense retrieval
    p.add_argument("--use_semantic_memory", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--semantic_backend", choices=["auto", "faiss", "sklearn"], default="auto")
    p.add_argument("--semantic_k", type=int, default=35)
    p.add_argument("--semantic_min_similarity", type=float, default=0.42)
    p.add_argument("--semantic_min_confidence", type=float, default=0.25)
    p.add_argument("--semantic_fusion_weight", type=float, default=0.85)
    p.add_argument("--ml_fusion_weight", type=float, default=1.0)
    p.add_argument("--semantic_reference_weight", type=float, default=0.60)
    p.add_argument("--naics_reference_csv", type=str, default="")
    p.add_argument("--soc_reference_csv", type=str, default="")

    # Optional local, air-gapped LLM candidate reranker via llama-cpp-python
    p.add_argument("--use_local_llm_fallback", action="store_true", default=False)
    p.add_argument("--llm_model_path", type=str, default="")
    p.add_argument("--llm_trigger_confidence_naics", type=float, default=0.55)
    p.add_argument("--llm_trigger_confidence_soc", type=float, default=0.55)
    p.add_argument("--llm_max_rows", type=int, default=250)
    p.add_argument("--llm_threads", type=int, default=6)
    p.add_argument("--llm_ctx_size", type=int, default=4096)
    p.add_argument("--llm_temperature", type=float, default=0.0)
    p.add_argument("--llm_top_p", type=float, default=0.90)
    p.add_argument("--llm_max_tokens", type=int, default=48)
    p.add_argument("--llm_rerank_confidence", type=float, default=0.72)
    p.add_argument("--llm_allow_free_code", action="store_true", default=False)

    # Top-k and rare class filters
    p.add_argument("--top_groups", type=int, default=3)
    p.add_argument("--topk_soc", type=int, default=4)
    p.add_argument("--topk_naics", type=int, default=4)
    p.add_argument("--min_soc_n", type=int, default=3)
    p.add_argument("--min_naics_n", type=int, default=3)
    p.add_argument("--nfolds_naics", type=int, default=5)
    p.add_argument("--nfolds_major", type=int, default=5)
    p.add_argument("--nfolds_within", type=int, default=3)

    # Feature settings
    p.add_argument("--max_word_terms", type=int, default=120000)
    p.add_argument("--max_char_terms", type=int, default=120000)
    p.add_argument("--word_min_count", type=int, default=2)
    p.add_argument("--char_min_count", type=int, default=5)
    p.add_argument("--doc_prop_max", type=float, default=0.80)
    p.add_argument("--use_embeddings", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    p.add_argument("--embedding_backend", choices=["torch", "onnx", "openvino"], default="torch")
    p.add_argument("--embedding_batch_size", type=int, default=32)
    p.add_argument("--embedding_device", type=str, default=None)
    p.add_argument("--svd_components", type=int, default=None)

    # Model settings
    p.add_argument("--logreg_C", type=float, default=1.0)
    p.add_argument("--logreg_max_iter", type=int, default=1500)
    p.add_argument("--sgd_alpha", type=float, default=1e-5)
    p.add_argument("--sgd_max_iter", type=int, default=2500)
    p.add_argument("--sgd_tol", type=float, default=1e-4)
    p.add_argument("--linearsvc_C", type=float, default=1.0)
    p.add_argument("--boost_n_estimators", type=int, default=300)
    p.add_argument("--boost_max_classes", type=int, default=120)
    p.add_argument("--boost_max_features", type=int, default=180000)

    # Hardware/scoring
    p.add_argument("--max_workers", type=str, default="auto")
    p.add_argument("--native_threads", type=int, default=1)
    p.add_argument("--score_batch_size", type=int, default=5000)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--joblib_compress", type=int, default=3)
    return p


def validate_args(opt: argparse.Namespace) -> None:
    if opt.mode in {"train", "train_and_score"} and not opt.train_sav:
        raise ValueError("--train_sav is required for train/train_and_score mode")
    if opt.mode in {"score", "train_and_score"} and not opt.score_sav:
        raise ValueError("--score_sav is required for score/train_and_score mode")
    if opt.mode == "score" and not opt.model_in:
        raise ValueError("--model_in is required for score mode")
    if opt.mode in {"score", "train_and_score"} and (not opt.out_sav or not opt.log_txt):
        raise ValueError("--out_sav and --log_txt are required when scoring")
    if opt.save_model and not opt.model_out:
        raise ValueError("--model_out is required when --save_model is used")
    if opt.mode == "train" and not opt.model_out:
        raise ValueError("--model_out is required for train mode")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    opt = parser.parse_args(argv)
    validate_args(opt)
    cfg, parallel_cfg = build_config(opt)

    eprint("NAICS-first SOC state-of-art autocoder")
    eprint(f"Version: {VERSION}")
    eprint(f"Profile={cfg.profile} | model_search={cfg.model_search} | max_workers={parallel_cfg.max_workers} | native_threads={parallel_cfg.native_threads}")
    eprint(f"Optional package availability: {package_availability()}")

    bundle: Optional[Dict[str, Any]] = None
    train_meta = None
    score_meta = None

    with threadpool_limits(limits=parallel_cfg.native_threads):
        if opt.mode in {"train", "train_and_score"}:
            eprint("\nReading training .sav...")
            train_df, train_meta = read_sav_with_meta(opt.train_sav)
            bundle = train_full_bundle(train_df, opt, cfg, parallel_cfg)
            if opt.save_model or opt.mode == "train":
                eprint(f"\nSaving model bundle to: {opt.model_out}")
                # joblib compression saves disk space. Set --joblib_compress 0 for fastest load and memmap-friendly arrays.
                joblib.dump(bundle, opt.model_out, compress=int(opt.joblib_compress))
                summary_path = opt.model_out + ".summary.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(bundle_summary(bundle), f, indent=2, default=str)
                eprint(f"Saved model summary to: {summary_path}")

        if opt.mode == "score":
            eprint(f"\nLoading model bundle from: {opt.model_in}")
            bundle = joblib.load(opt.model_in)
            # Use saved config unless explicit runtime write thresholds/overwrite settings changed.
            saved_cfg = bundle.get("config")
            if isinstance(saved_cfg, dict):
                # Keep current write/parallel CLI decisions but use the training feature/model settings.
                runtime_write = {
                    "min_write_confidence_naics": cfg.min_write_confidence_naics,
                    "min_write_confidence_soc": cfg.min_write_confidence_soc,
                    "missing_soc_values": cfg.missing_soc_values,
                    "missing_naics_values": cfg.missing_naics_values,
                    "score_batch_size": cfg.score_batch_size,
                }
                merged_cfg = asdict(cfg)
                merged_cfg.update(saved_cfg)
                merged_cfg.update(runtime_write)
                cfg = TrainConfig(**merged_cfg)

        if opt.mode in {"score", "train_and_score"}:
            if bundle is None:
                raise RuntimeError("No model bundle is available for scoring")
            eprint("\nReading scoring .sav...")
            score_df, score_meta = read_sav_with_meta(opt.score_sav)
            out_df, details = score_with_bundle(score_df, bundle, opt, cfg)
            eprint(f"\nWriting scored .sav to: {opt.out_sav}")
            write_sav_preserving_metadata(out_df, score_meta, opt.out_sav)
            eprint(f"Writing log to: {opt.log_txt}")
            write_log(
                path=opt.log_txt,
                opt=opt,
                cfg=cfg,
                bundle=bundle,
                score_df=score_df,
                before_naics=details["before_naics"],
                after_naics=details["after_naics"],
                pred_naics=details["pred_naics"],
                apply_naics=details["apply_naics"],
                before4=details["before4"],
                after4=details["after4"],
                pred4=details["pred4"],
                apply4=details["apply4"],
                before5=details["before5"],
                after5=details["after5"],
                pred5=details["pred5"],
                apply5=details["apply5"],
            )

    eprint("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        eprint("Interrupted by user.")
        raise SystemExit(130)
