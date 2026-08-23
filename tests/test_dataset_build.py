from backend.services.dataset.build import build_dataset


def test_build_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = build_dataset(
        count=100,
        seed=42,
    )

    assert result["total"] == 100
    assert result["train"] == 80
    assert result["validation"] == 10
    assert result["test"] == 10

    assert sum(result["actions"].values()) == 100
    assert sum(result["risks"].values()) == 100
    assert sum(result["categories"].values()) == 100

    assert (tmp_path / "data" / "train.jsonl").exists()
    assert (tmp_path / "data" / "validation.jsonl").exists()
    assert (tmp_path / "data" / "test.jsonl").exists()