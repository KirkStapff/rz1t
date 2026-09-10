from rz1t.screen import ARMS, arm_config, chunk_text, write_jsonl


def test_chunk_text_packs_short_paragraphs(tmp_path):
    text = "aaa\n\nbbb\n\n" + ("c" * 200)
    docs = chunk_text(text, target=50, minimum=10)
    assert all(len(d) >= 10 for d in docs)
    path = write_jsonl(docs, tmp_path / "docs.jsonl")
    assert path.read_text().count("\n") == len(docs)


def test_screen_arms_have_expected_depths():
    applied = []
    for spec in ARMS:
        cfg = arm_config(
            arm=spec["arm"], vocab=80, sequence=128, n_embed=64, aft_heads=4,
            aft_ksize=4, linear_fan_in=4, steps=10, batch_size=2, eval_every=5,
            checkpoint_every=5, lr=3e-4, warmup=1, seed=0, manifest="manifest.json",
            p=spec["p"], m=spec["m"], k=spec["k"], q=spec["q"], injection=spec["injection"],
        )
        from rz1t.config import ModelConfig
        model = ModelConfig.from_dict(cfg["model"])
        applied.append((spec["arm"], model.applied_depth, model.unique_depth, model.injection))
    assert applied[0] == ("U12", 12, 12, "none")
    assert applied[1] == ("R6x2", 12, 6, "none")
    assert applied[2] == ("R6x2-residual", 12, 6, "residual")
    assert applied[3] == ("R2x6-residual", 12, 2, "residual")
    assert applied[4] == ("P1M2K4Q1-residual", 10, 4, "residual")
