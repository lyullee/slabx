from slabx.source_ledger import SourceLedger, SourceState


def test_source_ledger_round_trip(tmp_path):
    ledger = SourceLedger(
        substance="hydrogen", stage="gas_handoff", duration_s=30.0,
        states=(SourceState(10.0, 0.2, 1.0, 300.0, 1.0, 0.01),),
    )
    path = tmp_path / "source.json"
    ledger.write_json(path)
    assert SourceLedger.read_json(path).to_dict() == ledger.to_dict()


def test_unresolved_liquid_is_rejected():
    ledger = SourceLedger(
        substance="hydrogen", stage="gas_handoff", duration_s=30.0,
        states=(SourceState(10.0, 0.2, 1.0, 80.0, 1.0, 0.01, liquid_fraction=0.1),),
    )
    try:
        ledger.to_horizontal_jet(substance=object())
    except ValueError as exc:
        assert "liquid" in str(exc)
    else:
        raise AssertionError("unresolved liquid state was accepted")
