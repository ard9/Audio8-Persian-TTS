"""Hub storage: data and checkpoints survive losing *all* local state (new machine / account)."""

import shutil

import fa_tools as fa
from fake_hub import FakeHfApi


def test_hub_storage_round_trip(fake_env, audio8_repo, tmp_path, monkeypatch):
    base, work, _cache = fake_env
    api = FakeHfApi(tmp_path / "hub")
    hub = fa.HubStore("me/codes", "me/training", api=api)
    plan = {"mana": 2, "gptinformal": 1}
    options = fa.PrepOptions(max_seconds=10)

    cache = tmp_path / "cache"
    fa.prepare_all(base, work, cache, plan, options, hub=hub)
    data_files = set(api.list_repo_files("me/codes", "dataset"))
    assert {"README.md", "refs.json", "shards/mana__dataset_part_001.tar"} <= data_files
    assert any(f.startswith("refs/") for f in data_files)

    # A brand-new machine: nothing local at all. Encoding must not run again.
    shutil.rmtree(work)
    shutil.rmtree(cache)
    work.mkdir()
    monkeypatch.setattr(fa, "_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-encoded")))
    hub = fa.HubStore("me/codes", "me/training", api=api)
    fa.prepare_all(base, work, cache, plan, options, hub=hub)
    train_jsonl, eval_jsonl = fa.build_manifests(
        base, audio8_repo, work, cache, plan, max_length=600, eval_size=4,
        manifest_dir=tmp_path / "run" / "manifests", hub=hub, run="r1")
    assert "runs/r1/manifests/train.jsonl" in api.list_repo_files("me/training", "model")

    refs = fa.load_refs(cache)
    clone = next(r for r in refs if r["dataset"] == "gptinformal")
    out = tmp_path / "run" / "checkpoints"
    cfg = fa.TrainConfig(
        base_dir=str(base), repo_dir=str(audio8_repo), train_jsonl=str(train_jsonl), eval_jsonl=str(eval_jsonl),
        output_dir=str(out), samples_dir=str(tmp_path / "run" / "samples"),
        batch_size=2, grad_accum=2, epochs=1, save_steps=3, keep_checkpoints=2, max_length=600,
        test_sentences=("سلام",), clone_reference=clone, logging_steps=1, num_workers=0, learning_rate=1e-3,
        hub=hub, hub_run="r1", hub_keep=1,
    )
    fa.train(cfg)
    remote = hub.remote_checkpoints("r1")
    assert len(remote) == 1 and api.squashed >= 1  # only the newest is kept, history squashed
    assert any(f.startswith("runs/r1/samples/") for f in api.list_repo_files("me/training", "model"))

    # Another machine again: local checkpoints gone -> resume from the Hub copy.
    shutil.rmtree(out)
    cfg.epochs = 2
    fa.train(cfg)
    assert hub.remote_checkpoints("r1")[-1] == 2 * remote[-1]

    # Manifests come back from the Hub too (same order on every machine).
    manifests = tmp_path / "run" / "manifests"
    first = (manifests / "train.jsonl").read_text("utf-8")
    shutil.rmtree(manifests)
    fa.build_manifests(base, audio8_repo, work, cache, plan, max_length=600, eval_size=4,
                       manifest_dir=manifests, hub=hub, run="r1")
    assert (manifests / "train.jsonl").read_text("utf-8") == first
