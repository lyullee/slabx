"""
Contract tests for the Trajectory schema.

The schema is the interface every add-on depends on, so these tests exist
to make a breaking change loud.  They check the contract, not physics.
"""

import numpy as np
import pytest

from slabx.core.trajectory import (
    ARRAY_FIELDS,
    SCHEMA_VERSION,
    Mode,
    Trajectory,
)

# Every field the reference implementation stores (SLAB_store, L2957) must
# have a home here.  Left column is the JS name, right column ours.
JS_FIELD_MAP = {
    "xp": "x", "timp": "t",
    "bbp": "b_half", "bp": "b_shape", "betap": "beta",
    "bbxp": "b_half_x", "bxp": "b_shape_x", "betaxp": "beta_x",
    "hp": "h", "zcp": "z_c",
    "up": "u", "tp": "T", "rhop": "rho",
    "cmp": "mass_frac", "cvp": "vol_frac",
    "vgp": "v_g", "ugp": "u_g", "wcp": "w_c", "uap": "u_ambient_mean",
    "cmdap": "mass_frac_dry_air", "cmwp": "mass_frac_water",
    "cmwvp": "mass_frac_water_vapour", "cmevp": "mass_frac_emission_vapour",
    "wp": "w_entrain", "vp": "v_entrain", "vxp": "v_x_entrain",
    "qintp": "mass_in_cloud",
}


def rows(n=20, mode=Mode.PLUME, x0=0.0):
    out = []
    for i in range(n):
        x = x0 + 10.0 * i
        out.append(
            dict(
                x=x, t=x / 2.0, mode=int(mode),
                h=1.0 + 0.05 * i, z_c=0.0,
                b_half=10.0 + i, b_shape=8.0 + i, beta=1.0,
                b_half_x=1.0, b_shape_x=1.0, beta_x=0.01,
                sigma_z=(1.0 + 0.05 * i) / np.sqrt(3.0),
                u=2.0, T=280.0, rho=1.3, cp=1010.0,
                mass_frac=0.3 * np.exp(-0.01 * i),
                vol_frac=0.4 * np.exp(-0.01 * i),
                v_g=0.5, u_g=0.0, w_c=0.0, u_ambient_mean=2.5,
                mass_frac_dry_air=0.7, mass_frac_water=1e-3,
                mass_frac_water_vapour=1e-3,
                mass_frac_emission_vapour=0.3, mass_frac_water_liquid=0.0,
                mass_frac_emission_liquid=0.0,
                w_entrain=0.01, v_entrain=0.05, v_x_entrain=0.06,
                u_star_cloud=0.15, phi_h=5.0, inv_L_cloud=0.4,
                f_u=0.1, f_v=-0.2, f_w=0.0, f_t=100.0,
                mass_in_cloud=min(1000.0, 60.0 * (i + 1)),
                R_flux=50.0,
            )
        )
    return out


def traj(**kw):
    return Trajectory.from_rows(rows(**kw), meta={"case": "unit-test"})


# ===========================================================================
# the contract itself
# ===========================================================================
def test_every_reference_output_has_a_home():
    """All 26 arrays of SLAB_store must be representable."""
    missing = [js for js, ours in JS_FIELD_MAP.items() if ours not in ARRAY_FIELDS]
    assert missing == [], f"reference fields with no schema slot: {missing}"


def test_schema_version_recorded():
    assert traj().meta["schema_version"] == SCHEMA_VERSION


def test_provenance_is_preserved():
    t = Trajectory.from_rows(rows(), meta={"coeffs": "ermak90", "thermo": "legacy"})
    assert t.meta["coeffs"] == "ermak90"
    assert t.meta["thermo"] == "legacy"


def test_trajectory_is_immutable():
    t = traj()
    with pytest.raises(Exception):
        t.x = np.zeros(3)


def test_all_arrays_same_length():
    t = traj(n=7)
    assert len(t) == 7
    for name in ARRAY_FIELDS:
        assert len(getattr(t, name)) == 7
    assert len(t.mode) == 7


# ===========================================================================
# validation
# ===========================================================================
def test_rejects_length_mismatch():
    r = rows(5)
    r[-1].pop("h")
    d = {k: np.array([row.get(k, np.nan) for row in r]) for k in ARRAY_FIELDS}
    d["h"] = d["h"][:-1]
    d["mode"] = np.zeros(5, dtype=np.int8)
    with pytest.raises(ValueError, match="length"):
        Trajectory(**d)


def test_rejects_empty():
    with pytest.raises(ValueError):
        Trajectory.from_rows([])


def test_rejects_missing_required_field():
    r = rows(3)
    for row in r:
        row.pop("vol_frac")
    with pytest.raises(KeyError, match="vol_frac"):
        Trajectory.from_rows(r)


def test_optional_fields_default_to_nan():
    r = rows(3)
    for row in r:
        row.pop("phi_h")
    t = Trajectory.from_rows(r)
    assert np.all(np.isnan(t.phi_h))
    assert np.all(np.isfinite(t.vol_frac))


def test_rejects_non_monotonic_coordinates():
    r = rows(4)
    r[2]["x"] = -5.0
    with pytest.raises(ValueError, match="non-decreasing"):
        Trajectory.from_rows(r)
    r = rows(4)
    r[2]["t"] = -1.0
    with pytest.raises(ValueError, match="non-decreasing"):
        Trajectory.from_rows(r)


# ===========================================================================
# mode handling
# ===========================================================================
def test_single_mode_has_no_transition():
    assert traj().transition_index is None


def test_mixed_mode_transition_and_slicing():
    r = rows(10, mode=Mode.PLUME) + rows(6, mode=Mode.PUFF, x0=100.0)
    for i, row in enumerate(r):
        row["t"] = float(i)
    t = Trajectory.from_rows(r)
    assert t.transition_index == 10
    assert t.is_plume.sum() == 10 and t.is_puff.sum() == 6
    assert len(t.slice(Mode.PLUME)) == 10
    assert len(t.slice(Mode.PUFF)) == 6
    assert t.slice(Mode.PUFF).meta["schema_version"] == SCHEMA_VERSION


def test_slice_of_absent_mode_raises():
    with pytest.raises(ValueError):
        traj().slice(Mode.PUFF)


def test_both_coordinates_always_present():
    """Add-ons may index by x or by t regardless of the integration variable."""
    t = traj(mode=Mode.PUFF)
    assert np.all(np.isfinite(t.x)) and np.all(np.isfinite(t.t))


# ===========================================================================
# interpolation
# ===========================================================================
def test_at_interpolates_linearly():
    t = traj()
    assert t.at(0.0)["b_half"] == pytest.approx(10.0)
    assert t.at(5.0)["b_half"] == pytest.approx(10.5)


def test_at_can_restrict_fields():
    assert set(traj().at(5.0, fields_=("h", "u"))) == {"h", "u"}


def test_at_rejects_extrapolation():
    t = traj()
    with pytest.raises(ValueError):
        t.at(-1.0)
    with pytest.raises(ValueError):
        t.at(t.x[-1] + 1.0)


def test_at_log_beats_linear_for_exponential_decay():
    """vol_frac decays exponentially, so log interpolation is exact-er."""
    t = traj(n=40)
    xq = 0.5 * (t.x[10] + t.x[11])
    exact = 0.4 * np.exp(-0.01 * (xq / 10.0))
    lin = t.at(xq, fields_=("vol_frac",))["vol_frac"]
    log = t.at_log(xq, "vol_frac")
    assert abs(log - exact) <= abs(lin - exact)


def test_at_log_returns_nan_for_all_nan_field():
    r = rows(4)
    for row in r:
        row.pop("phi_h")
    assert np.isnan(Trajectory.from_rows(r).at_log(5.0, "phi_h"))


# ===========================================================================
# export
# ===========================================================================
def test_to_dict_round_trip():
    t = traj()
    d = t.to_dict()
    assert set(d) == set(ARRAY_FIELDS) | {"mode"}
    assert np.array_equal(d["vol_frac"], t.vol_frac)


def test_to_dataframe_carries_metadata():
    pd = pytest.importorskip("pandas")
    df = traj().to_dataframe()
    assert df.attrs["case"] == "unit-test"
    assert df.index.name == "x"
    assert "vol_frac" in df.columns


# ===========================================================================
# built-in Tier-0 check
# ===========================================================================
def test_mass_conservation_scalar_form_checks_the_plateau():
    t = traj(n=40)                      # mass_in_cloud saturates at 1000
    good = t.check_mass_conservation(1000.0)
    assert good["passes"]
    assert "plateau" in good["checked_over"]
    bad = t.check_mass_conservation(1200.0)
    assert not bad["passes"]
    assert bad["max_rel_error"] > 0.1


def test_mass_conservation_array_form_checks_every_row():
    """
    The array form is the meaningful one: it tests the source region, where
    mass errors are actually made, instead of only the plateau.
    """
    t = traj(n=40)
    exact = t.mass_in_cloud.copy()
    assert t.check_mass_conservation(exact)["passes"]

    perturbed = exact.copy()
    perturbed[3] *= 1.05                    # 5 % error early on, inside the ramp
    r = t.check_mass_conservation(perturbed)
    assert not r["passes"]
    assert r["max_rel_error"] == pytest.approx(0.05 / 1.05, rel=1e-6)
    assert r["worst_at_x"] == pytest.approx(t.x[3])
    # the scalar form is blind to it
    assert t.check_mass_conservation(1000.0)["passes"]


def test_mass_conservation_rejects_bad_arguments():
    t = traj(n=10)
    with pytest.raises(ValueError):
        t.check_mass_conservation(np.ones(3))
    with pytest.raises(ValueError):
        t.check_mass_conservation(1000.0, tail_frac=0.0)


def test_mass_conservation_reports_unavailable():
    r = rows(6)
    for row in r:
        row.pop("mass_in_cloud")
    assert Trajectory.from_rows(r).check_mass_conservation(1.0) == {"available": False}


def test_repr_is_informative():
    s = repr(traj())
    assert "Trajectory" in s and "PLUME" in s
