from integrated_architecture import LeonardoEngine


def test_ingest_and_hypothesis():
    engine = LeonardoEngine()
    engine.ingest("A", "source-a", 0.9)
    h = engine.generate_hypothesis("B")
    result = engine.evaluate(h)
    assert result["statement"] == "B"
    assert 0.0 <= result["score"] <= 1.0


def test_contradiction_detection():
    engine = LeonardoEngine()
    engine.ingest("A", "positive", 0.8)
    engine.ingest("not A", "negative", 0.8)
    assert len(engine.find_contradictions()) == 1
