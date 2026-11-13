import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "scripts" / "train.py"

spec = importlib.util.spec_from_file_location("train_module", TRAIN_PATH)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)


def test_skops_trusted_types_include_randomforest_tree_type():
    assert "numpy.dtype" in train_module.SKOPS_TRUSTED_TYPES
    assert "sklearn.tree._tree.Tree" in train_module.SKOPS_TRUSTED_TYPES
