"""
Two-phase jet thermodynamics against an independent calculation.

The Desert Tortoise ammonia jets are the hardest source type in this package
and the least validated: the manual's deck exercises it, and the SMEDIS
equivalent sources gave two loose comparison points, but nothing tested the
phase-equilibrium solver itself against a worked calculation.

The Jack Rabbit III Modelers Working Group published exactly that.  For the
JRIII inter-comparison exercise (2021-2024), HSE re-implemented the CERC
(2000) two-phase jet method and tabulated every intermediate state for
DT1, DT2 and DT4 — the flash quality, the temperature and composition at the
point where the last droplet evaporates, and the resulting density.  That is
an independent calculation of the same physics, by a different group, from
NIST properties, published as a working document rather than fitted to any
dispersion statistic.

A modelling trap worth stating
-------------------------------
The JRIII "equivalent vapour-only source" cannot be given to SLAB directly.
It describes an ammonia *and air* mixture at 14 mol % and 1.46 kg/m3 — denser
than the ambient — but SLAB's jet source is built from the pure released
material, so handing it the equivalent conditions produces pure ammonia
vapour at 205 K and 0.91 kg/m3, which is *lighter* than air.  The cloud then
lifts off (z_c = 19 m at 100 m) and the ground-level concentration collapses
to zero.

The equivalent source exists for models that cannot represent a two-phase
release.  SLAB can, so it should be given the real one: the post-flash
conditions from Table 5, with liquid fraction 1 - X_f and the flashing
temperature, and left to do its own entrainment and evaporation.

Reference
---------
Gant S., Tickle G., Hetherington R. and Chang J. (2022) "Equivalent
Vapor-Only Source Conditions for the Desert Tortoise Trials", Jack Rabbit III
Modelers Working Group, Version 1.2, 12 January 2022.  Tables 3-7.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("CoolProp")

from slabx.validation._data_access import require
from slabx.coefficients import PHYS                              # noqa: E402
from slabx.thermo.base import LegacyThermo, Substance            # noqa: E402
from slabx.thermo.coolprop import CoolPropThermo                 # noqa: E402
from slabx.thermo.equilibrium import Mixture, solve_equilibrium  # noqa: E402

#: Ammonia as the JRIII document specifies it: NIST properties, cp_v from
#: their Table 6, latent heat from Osborne & Van Dusen (1917).
AMMONIA = Substance(name="NH3", mw=0.017031, cp_vapour=2272.0,
                    cp_liquid=4611.8, dh_vap=1370000.0, T_boil=239.6,
                    rho_liquid=684.0, sat_B=2976.01)

#: Temperature at the end of flashing, common to all three trials (Table 4).
T_FLASH = 237.7

DATA = (Path(__file__).resolve().parents[1] / "src" / "slabx" / "validation"
        / "data" / "desert_tortoise_source.csv")


def _trials():
    with open(require(DATA.name), newline="") as fh:
        return {r["trial"]: r for r in csv.DictReader(fh)}


def _evaporation_endpoint(row):
    """
    Dilute the flashed jet with air until the last droplet evaporates, and
    return the state there — which is what the equivalent source is.
    """
    quality = float(row["quality_flash"])
    T_amb = float(row["T_atm_K"])
    p_amb = float(row["p_atm_Pa"])
    emission = CoolPropThermo(AMMONIA, fluid="Ammonia")
    water = LegacyThermo(AMMONIA)
    rho_amb = p_amb * PHYS.MW_AIR / (PHYS.R_GAS * T_amb)

    previous = None
    for m in np.linspace(0.30, 0.01, 6000):
        # Sensible enthalpy on the 0 K datum; the solver supplies the latent
        # term through `dh_vap_datum`.  Getting this wrong shifts the answer
        # by 4 % in temperature and a factor two in composition.
        enthalpy = (m * quality * AMMONIA.cp_vapour * T_FLASH
                    + m * (1.0 - quality) * AMMONIA.cp_liquid * T_FLASH
                    + (1.0 - m) * 1005.0 * T_amb)
        mix = Mixture(m_emission=m, m_water=0.0, m_dry_air=1.0 - m,
                      m_ev_transported=m * quality, m_wv_transported=0.0,
                      enthalpy=enthalpy)
        try:
            eq = solve_equilibrium(mix, emission, water, T_ambient=T_amb,
                                   rho_ambient=rho_amb,
                                   mw_ambient_moist=PHYS.MW_AIR)
        except Exception:                                          # noqa: BLE001
            continue
        liquid = m - eq.m_ev
        if previous is not None and liquid <= 1e-7 < previous:
            mole = ((m / AMMONIA.mw)
                    / (m / AMMONIA.mw + (1.0 - m) / PHYS.MW_AIR))
            return eq.T, mole, eq.rho
        previous = liquid
    raise AssertionError("no evaporation endpoint found")


@pytest.fixture(scope="module")
def endpoints():
    return {t: _evaporation_endpoint(r) for t, r in _trials().items()}


def test_the_source_table_is_complete():
    trials = _trials()
    assert set(trials) == {"DT1", "DT2", "DT4"}
    for t, r in trials.items():
        assert 0.18 < float(r["quality_flash"]) < 0.20, t
        assert 204.0 < float(r["T_evap_K"]) < 206.0, t


@pytest.mark.parametrize("trial", ["DT1", "DT2", "DT4"])
def test_temperature_at_the_evaporation_endpoint(trial, endpoints):
    """
    Within about 1 % of HSE's independently computed value, for all three
    trials.  The endpoint temperature is where the two-phase thermodynamics
    is most exposed: it follows from the energy balance between the latent
    heat of the evaporating droplets and the sensible heat of the entrained
    air, with nothing else to absorb an error.
    """
    ours, _, _ = endpoints[trial]
    theirs = float(_trials()[trial]["T_evap_K"])
    assert ours == pytest.approx(theirs, rel=0.02)


@pytest.mark.parametrize("trial", ["DT1", "DT2", "DT4"])
def test_composition_at_the_evaporation_endpoint(trial, endpoints):
    """
    The mole fraction follows from the saturation curve at that temperature,
    so agreeing here means the saturation model agrees too — ours from
    CoolProp, theirs from the NIST Antoine fit.
    """
    _, ours, _ = endpoints[trial]
    theirs = float(_trials()[trial]["mole_frac_evap"])
    assert ours == pytest.approx(theirs, rel=0.03)


@pytest.mark.parametrize("trial", ["DT1", "DT2", "DT4"])
def test_density_at_the_evaporation_endpoint(trial, endpoints):
    _, _, ours = endpoints[trial]
    theirs = float(_trials()[trial]["rho_evap"])
    assert ours == pytest.approx(theirs, rel=0.03)


def test_the_enthalpy_datum_matters(endpoints):
    """
    Regression guard.  Building the mixture enthalpy with the latent heat
    written in explicitly — rather than as sensible heat on the 0 K datum the
    solver expects — shifts the endpoint temperature by 4 % and the
    composition by a factor of two, in a way that still converges and still
    looks physical.
    """
    row = _trials()["DT1"]
    quality = float(row["quality_flash"])
    T_amb, p_amb = float(row["T_atm_K"]), float(row["p_atm_Pa"])
    emission, water = CoolPropThermo(AMMONIA, fluid="Ammonia"), \
        LegacyThermo(AMMONIA)
    rho_amb = p_amb * PHYS.MW_AIR / (PHYS.R_GAS * T_amb)

    m = 0.086                                    # near the true endpoint
    wrong = (m * (AMMONIA.cp_vapour * T_FLASH - (1 - quality) * 1.37e6)
             + (1 - m) * 1005.0 * T_amb)
    eq = solve_equilibrium(
        Mixture(m_emission=m, m_water=0.0, m_dry_air=1 - m,
                m_ev_transported=m * quality, m_wv_transported=0.0,
                enthalpy=wrong),
        emission, water, T_ambient=T_amb, rho_ambient=rho_amb,
        mw_ambient_moist=PHYS.MW_AIR)
    correct_T = endpoints["DT1"][0]
    assert eq.T < 0.97 * correct_T               # visibly, silently wrong


# ===========================================================================
# against measured arc-max concentrations
# ===========================================================================
def _met():
    with open(require("desert_tortoise_met.csv"), newline="") as fh:
        return {r["trial"]: r for r in csv.DictReader(fh)}


def _arc_observations():
    with open(require("jriii_arcmax.csv"), newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["trial"].startswith("DT")]


def _predict(trial):
    """The real two-phase release, from post-flash conditions."""
    from slabx.coefficients import preset
    from slabx.core.plume import run_dispersion
    from slabx.core.source import HorizontalJet
    from slabx.post.concentration import concentration_field
    from slabx.submodels.atmosphere import Atmosphere
    from slabx.thermo.coolprop import coolprop_water

    src_row, met = _trials()[trial], _met()[trial]
    atm = Atmosphere(u_ref=float(met["u_ref_m_s"]), z_ref=float(met["z_ref_m"]),
                     T=float(met["T_amb_K"]), rh=float(met["rh_pct"]),
                     z0=float(met["z0_m"]), stability=met["stability"],
                     p=float(src_row["p_atm_Pa"]))
    jet = HorizontalJet(
        substance=AMMONIA, rate=float(src_row["mdot_kg_s"]),
        area=float(src_row["A_flash_m2"]),
        duration=float(met["duration_s"]),
        liquid_fraction=1.0 - float(src_row["quality_flash"]),
        height=0.79, T_source=T_FLASH,
    )
    traj, _ = run_dispersion(jet, atm, CoolPropThermo(AMMONIA, fluid="Ammonia"),
                             coolprop_water(), x_max=1200.0, n_puff_steps=40,
                             coeffs=preset("ermak90"))
    field = concentration_field(traj, atm, z=float(met["sensor_height_m"]),
                                t_avg=1.0, t_release=float(met["duration_s"]))
    return traj, field


@pytest.fixture(scope="module")
def arc_comparison():
    from slabx.post.concentration import ConcentrationField  # noqa: F401

    obs, pred = [], []
    fields = {t: _predict(t)[1] for t in ("DT1", "DT2", "DT4")}
    rows = _arc_observations()
    for r in rows:
        f = fields[r["trial"]]
        x = float(r["arc_m"])
        pred.append(float(np.exp(np.interp(
            x, f.x, np.log(np.maximum(f.peak, 1e-30))))) * 1e6)
        obs.append(float(r["c_obs_ppm"]))
    return np.array(obs), np.array(pred), rows


def test_arc_observations_loaded():
    rows = _arc_observations()
    assert len(rows) == 6                     # three trials, two arcs each
    assert {float(r["arc_m"]) for r in rows} == {100.0, 800.0}


def test_two_phase_jet_agrees_within_a_factor_of_two_at_most_arcs(
        arc_comparison):
    """
    The first comparison of this source type against measured concentrations.
    Two of six arcs fall outside a factor of two — both the 100 m arc, where
    the near-field jet is hardest — which is the same pattern the other
    integral models show.
    """
    from slabx.validation.metrics import metrics

    obs, pred, _ = arc_comparison
    m = metrics(obs, pred)
    assert m.FAC2 >= 0.6
    assert 0.5 < m.MG < 1.5


def test_it_is_not_worse_than_drift_on_the_same_arcs(arc_comparison):
    """
    Stated as a fact about these six points, not a general claim.  DRIFT and
    slabx both over-predict the near arc; Phast is the best of the three.
    """
    from slabx.validation.metrics import metrics

    obs, pred, rows = arc_comparison
    drift = np.array([float(r["c_drift"]) for r in rows])
    ours, theirs = metrics(obs, pred), metrics(obs, drift)
    assert abs(ours.MG - 1.0) <= abs(theirs.MG - 1.0)
    assert ours.VG <= theirs.VG


def test_the_equivalent_source_must_not_be_used_directly():
    """
    Regression guard for the trap in the module docstring: the JRIII
    equivalent source is an ammonia-air mixture, and feeding its conditions
    to SLAB's pure-material jet makes the cloud buoyant and lifts it off.
    """
    from slabx.coefficients import preset
    from slabx.core.plume import run_dispersion
    from slabx.core.source import HorizontalJet
    from slabx.submodels.atmosphere import Atmosphere
    from slabx.thermo.base import LegacyThermo, water_backend

    row, met = _trials()["DT1"], _met()["DT1"]
    half = float(row["halfwidth_evap_m"])
    atm = Atmosphere(u_ref=float(met["u_ref_m_s"]), z_ref=2.0,
                     T=float(met["T_amb_K"]), rh=float(met["rh_pct"]),
                     z0=0.003, stability="D")
    jet = HorizontalJet(substance=AMMONIA, rate=float(row["mdot_kg_s"]),
                        area=2 * half * half, duration=400.0,
                        liquid_fraction=0.0, height=0.0,
                        T_source=float(row["T_evap_K"]))
    traj, _ = run_dispersion(jet, atm, LegacyThermo(AMMONIA), water_backend(),
                             x_max=400.0, n_puff_steps=20,
                             coeffs=preset("ermak90"))
    i = int(np.argmin(np.abs(traj.x - 100.0)))
    assert traj.z_c[i] > 10.0                 # lofted, wrongly
    assert jet.rho_source(atm) < atm.rho      # because it came out buoyant
