import numpy as np

from othello_rl.utils.config import Config, dump_config, load_config
from othello_rl.utils.logging import MetricLogger
from othello_rl.utils.seed import seed_everything, spawn_seed
import random


def test_config_attr_access_and_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\nnested:\n  x: 2\n  y: 3\nlist: [1, 2]\n")
    cfg = load_config(p, nested={"y": 30})
    assert cfg.a == 1
    assert cfg.nested.x == 2
    assert cfg.nested.y == 30  # deep-merged override
    assert cfg["list"] == [1, 2]
    assert cfg.get_path("nested.x") == 2
    assert cfg.get_path("nested.missing", "d") == "d"

    out = tmp_path / "o.yaml"
    dump_config(cfg, out)
    assert load_config(out).nested.x == 2


def test_seed_everything_reproducible():
    seed_everything(123)
    a = [random.random(), float(np.random.rand())]
    seed_everything(123)
    b = [random.random(), float(np.random.rand())]
    assert a == b
    assert seed_everything(None) is None


def test_spawn_seed_stream():
    r1 = random.Random(0)
    r2 = random.Random(0)
    assert [spawn_seed(r1) for _ in range(5)] == [spawn_seed(r2) for _ in range(5)]


def test_metric_logger_jsonl_and_csv(tmp_path):
    log = MetricLogger(tmp_path / "m.jsonl")
    log.log(step=1, loss=0.5, arr=np.array([1, 2]))
    log.log(step=2, loss=0.25)
    loaded = MetricLogger.load(tmp_path / "m.jsonl")
    assert len(loaded) == 2 and loaded[0]["loss"] == 0.5
    xs, ys = log.series("step", "loss")
    assert xs == [1, 2] and ys == [0.5, 0.25]
    csv_path = log.to_csv(tmp_path / "m.csv")
    assert csv_path.exists()
    assert "loss" in csv_path.read_text().splitlines()[0]
