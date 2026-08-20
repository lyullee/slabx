"""
Physical constants and externalized empirical coefficients.

Everything here that Ermak (1990) hard-coded is exposed as a field of a
frozen dataclass so that it can be (a) swapped in ablation studies,
(b) perturbed in uncertainty quantification, and (c) re-fitted against
validation data.  No numeric literal from the original model should ever
appear inline in the physics modules.

Provenance is recorded per field.  `EQ` refers to equation numbers in
UCRL-MA-105607 (Ermak 1990); `TNO` to CPR 14E chapter 4 (Yellow Book).
"""

from __future__ import annotations

from dataclasses import dataclass, replace, asdict, field
from typing import Any


# ===========================================================================
# Physical constants — not tunable
# ===========================================================================
@dataclass(frozen=True)
class Physical:
    """Universal constants and fixed fluid properties (SLAB.FOR L325-406)."""

    R_GAS: float = 8.31431          # J/(mol*K)   gas constant
    GRAVITY: float = 9.80665        # m/s^2
    VON_KARMAN: float = 0.41        # -           EQ 32
    P_ATM: float = 101325.0         # Pa          SLAB assumes 1 atm throughout

    # dry air
    MW_AIR: float = 0.02896         # kg/mol
    CP_AIR: float = 1005.87         # J/(kg*K)

    # water (built into SLAB; not user input)
    MW_WATER: float = 0.01802       # kg/mol
    RHO_WATER_LIQ: float = 1000.0   # kg/m^3
    CP_WATER_VAP: float = 1846.0    # J/(kg*K)
    CP_WATER_LIQ: float = 4178.0    # J/(kg*K)
    DH_WATER: float = 2441000.0     # J/kg        heat of vaporisation
    SAT_A_WATER: float = 15.08      # -           EQ 43a constants for water
    SAT_B_WATER: float = 5514.0     # K


PHYS = Physical()


# ===========================================================================
# Empirical coefficients — tunable
# ===========================================================================
@dataclass(frozen=True)
class Coefficients:
    """
    Empirically fitted constants of the SLAB submodels.

    Defaults reproduce Ermak (1990) exactly.  Named variants live in
    `PRESETS`.  Use `.perturb(**kw)` or `.variant(name)` to obtain
    modified sets; instances are frozen so a Trajectory can safely record
    the exact set that produced it.
    """

    # -- entrainment -------------------------------------------------------
    a_entrain: float = 1.5
    """Vertical entrainment constant `a`, EQ 35a. Zeman (1982)."""

    h_entrain_ref: float = 4.0

    #: Vertical entrainment closure. ``"slab"`` is EQ 35a as Ermak wrote it,
    #: with the fixed reference height above. ``"robins"`` and ``"nielsen"``
    #: replace the ``urf/uah`` factor with a published Richardson-number form
    #: that carries no absolute length; see `submodels/entrainment.py`.
    #: Variants only -- the default reproduces the original.
    entrainment_closure: str = "slab"

    #: The other absolute length in the entrainment closures: EQ 36a
    #: references its ambient-stability correction to the wind at a fixed
    #: 2 m.  Externalised for the Froude-similarity test.
    h_horiz_ref: float = 2.0

    #: Added-mass coefficient for a cloud rising broadside, ``C_A`` in
    #: ``k = C_A (rho_a/rho)(B/h)``.  The default is the potential-flow value
    #: for an oblate spheroid, 2/pi, which is not fitted.  Externalised so
    #: that the Hall & Walker comparison can vary it; see
    #: `docs/PREREG_added_mass.md`.
    c_added_mass: float = 0.6366197723675814

    #: Weight on the *ambient* term of the in-cloud Monin-Obukhov length,
    #: EQ 35d.  The default 1.0 is what SLAB does; setting it to zero leaves
    #: only the cloud's own stratification, which is the controlled
    #: intervention that turns the observed stability correlation into a
    #: causal statement.  See `docs/PREREG_stability.md`.
    w_ambient_stratification: float = 1.0

    #: Robins et al. (2001), Atmos. Environ. 35, 2243-2252:
    #: ``W_E / u* = c_robins / (1 + b_robins Ri*)`` for Ri* < 15, measured
    #: directly in a wind tunnel over a rough surface.
    c_robins: float = 0.65
    b_robins: float = 0.2

    #: Nielsen (1998), Risoe-R-1030(EN):
    #: ``u_e / e = c_nielsen / (a_nielsen + Ri)`` with
    #: ``e = (u*^3 + 0.1 w*^3)^(1/3)``, constructed to meet all four
    #: canonical-flow limits at once.
    #:
    #: The numerator is fixed by Nielsen's own passive limit rather than
    #: taken on trust: at Ri = 0 in neutral air ``e = u*``, so ``u_e/u*``
    #: must equal ``lambda_1 = 0.75``, giving ``c = 0.75 * 3.3 = 2.5``.
    #: The form was first transcribed here with 0.25, which is a factor ten
    #: below its own stated limit; the inconsistency is what found it.
    c_nielsen: float = 2.5
    a_nielsen: float = 3.3
    """
    Reference height in the vertical entrainment, EQ 35a [m].

    ``W_e`` is scaled by ``U_r / U_a(h_top)`` with ``U_r`` the dimensionless
    wind profile at this height.  It is an *absolute* length in an otherwise
    similarity-based closure, and it is the reason the model is not
    Froude-similar: scaling a wind-tunnel release up by a factor 1000 in
    length, with velocity and rate scaled to match, changes the predicted
    concentration by 2.4 rather than leaving it fixed.

    That matters because it means wind-tunnel and field results cannot be
    transferred through the model, which is why the MEP requires wind-tunnel
    trials to be run at tunnel scale rather than at their equivalent full
    scale.  Exposed here so the dependence can be measured; the default is
    Ermak's hard-coded value.
    """

    weber_critical: float = 12.0
    """
    Critical Weber number for aerodynamic droplet break-up (rainout).

    Sets the largest stable droplet, ``d = We sigma / (rho_g u^2)``, and
    through it the settling velocity.  Values of 10-20 appear in the
    break-up literature depending on how break-up is defined; 12 is the usual
    choice for the largest stable droplet.

    It lives here rather than as a module constant because a module-level
    default argument binds at import and cannot be swept — which silently
    made an entire sensitivity study return the same answer five times.
    """

    rainout_efficiency: float = 1.0
    """
    Scaling on the rainout removal rate, ``v_t / (U h)``.

    The droplet size comes from a measured break-up criterion and the
    settling velocity from a drag balance, so neither is free.  The removal
    rate is the one modelling choice: it assumes a droplet has to fall the
    full cloud depth and that the liquid is distributed uniformly through it,
    both of which overstate how quickly droplets reach the ground.

    Calibrated against the SMEDIS equivalent sources for Desert Tortoise —
    the distance at which no liquid remains, 51 m for DT1 and 48 m for DT2 —
    which is a droplet-lifetime measurement and carries no dispersion
    statistics.  Sweeping it, DT1 reaches its target at about 0.13 and DT2 at
    about 0.45 — a factor three apart, which for a first calibration on two
    trials at the resolution of the output grid is reasonable agreement, and
    far tighter than the factor 200 that the surface roughness needed.  Both
    say the unscaled closure removes droplets several times too fast.

    The default leaves the closure unscaled so that the calibration is
    opt-in; `PRESETS["rainout_smedis"]` carries 0.25.
    """

    alpha_front: float = 0.0
    """
    Gravity-front entrainment coefficient (EQ 36a extension, not in Ermak).

    SLAB's horizontal entrainment has two contributions — ambient turbulence
    and jet shear — and no term driven by the gravity-spreading front itself.
    Every other integral model of the same generation has one: DEGADIS takes
    it from van Ulden (1979, 1983), HEGADAS folds it into its cross-wind
    spreading equation, and DRIFT reports its "edge" entrainment dominating
    over top entrainment in F2 conditions.

    The front velocity V_g is 5-30 times SLAB's whole horizontal entrainment
    velocity in stable, low-wind conditions, so even a small coefficient
    changes the answer.  The default is **zero** — Ermak's model exactly —
    because the slab-averaged value is not established in the literature and
    has to be calibrated; `PRESETS["frontal"]` carries a nominal value for
    sensitivity work.
    """

    a2_horiz: float = 0.0004
    """Horizontal entrainment damping `a2`, EQ 36a."""

    c_10: float = 0.086
    """Reference friction-velocity ratio `C10` in S(La), EQ 36a (JS `cf00`)."""

    L_a_ref: float = 10.0
    """Reference Monin-Obukhov length `La` [m], EQ 36a."""

    p_meander: float = 0.2
    """Exponent `p` in the averaging-time function F_theta, EQ 36a."""

    tau_min: float = 10.0
    """Minimum averaging time `tau` [s], EQ 36a / EQ 49b."""

    t_ref_avg: float = 900.0
    """Normalisation averaging time `t0` [s] (15 min), EQ 36a."""

    c_shear_x: float = 0.6
    """Down-wind shear coefficient, EQ 36b / TNO 4.128."""

    c_shear_y: float = 0.209
    """Cross-wind shear coefficient (= a*k), TNO 4.127."""

    c3_stability: float = 34.1
    """Constant `C3` [m] in the stability function fs(L), TNO 4.130."""

    # -- friction velocity / fluxes ---------------------------------------
    c_drag_top: float = 0.02
    """
    Top-of-cloud drag coefficient `Cf`, EQ 35e / TNO 4.132-4.133.

    TNO's write-up implies 0.0195; Ermak's code uses 0.02
    (SLAB.FOR L381).  The default follows the code.
    TNO 4.127 independently confirms 0.0195: its cross-wind shear constant
    0.209 equals sqrt(0.0195)*1.5 to four figures, whereas sqrt(0.02)*1.5
    gives 0.2121.  We therefore treat 0.02 as a rounding defect in the
    reference implementation, reproducible via `Coefficients.LEGACY_JS`.
    """

    c_mu_strat: float = 0.025
    """
    Stratification coefficient in the in-cloud Monin-Obukhov length, EQ 35d.

    Together with `phi_stable` it fixes the *shape* of the entrainment
    damping.  In the two shear-driven limits SLAB gives

        u_e/u_*        -> lambda_1                 as Ri -> 0
        u_e Ri / u_*   -> lambda_2 = lambda_1 / (phi_stable c_mu_strat)

    so the ratio ``lambda_2/lambda_1 = 1/(phi_stable c_mu_strat)`` is a pure
    shape number: it is independent of the box-height convention and of how
    the turbulence velocity scale is defined, because both limits are
    normalised the same way.  That makes it directly comparable with
    laboratory measurements.

    Nielsen & Jensen (Risoe) collect the canonical values
    ``lambda_1 = 0.75`` (Sutton 1953, analytic) and ``lambda_2 = 2.5``
    (Kato & Phillips 1969, annular tank), giving ``lambda_2/lambda_1 = 3.33``.
    With ``phi_stable = 5`` that requires ``c_mu_strat = 0.060``; Ermak's
    value is 0.025, so SLAB's stratification damping is a factor 2.4 too
    weak and it over-entrains strongly stratified clouds.

    Field data contradicts the laboratory value
    -------------------------------------------
    Sweeping the damping strength ``D = phi_stable * c_mu_strat`` against the
    Burro measurements gives, at the standard roughness:

        D        lambda_2/lambda_1   MG      VG     FAC2   NMSE
        0.031        32.0           0.820   1.248   0.83   0.279  passes
        0.0625       16.0           0.710   1.376   0.75   0.461
        0.125         8.0           0.606   1.616   0.67   0.750  Ermak
        0.300         3.33          0.535   1.978   0.50   0.982  laboratory

    Monotone: the value the canonical flows require is the *worst* on every
    measure, and the field prefers four times less damping than Ermak already
    uses.  The between-trial scatter tells the same story with a genuine
    interior minimum — the standard deviation of log MG across the four
    trials falls from 0.416 at the laboratory value to 0.241 at D = 0.0625,
    so this is not simply a bias knob.

    The reading is that this coefficient is standing in for mixing that field
    releases experience and laboratory gravity currents do not — ambient
    turbulence over a range of scales that a tank or an annular flume cannot
    reproduce.  Ermak's 0.125 is already well below the laboratory value, so
    the calibration was evidently against field data too; the field simply
    wants to go further.

    This is why `PRESETS["canonical"]` fails its pre-registered prediction:
    it moves *towards* the laboratory value.  The preset is kept because the
    laboratory limits are a real constraint and the contradiction between
    them and the field is the finding, not an error to be tuned away.

    The same coefficient also controls the *convective* branch, through the
    same damping function but with the thermal velocity scale in place of
    the friction velocity.  Setting it from the shear limits alone moves the
    convective knee from 2.5 to 1.0 against a reference of 0.72 — an
    independent check, since the convective data were not used to choose it.

    The default stays at Ermak's value — `PRESETS["canonical"]` carries the
    laboratory-consistent one.  Note that none of the four reference values
    comes from a dispersion trial, so using field data to test the change is
    not circular.
    """
    """Stratification damping `C_mu` in the in-cloud MO length, EQ 35d."""

    c_thermal: float = 0.14
    """
    Convective-turbulence coefficient `C_t` in EQ 35e.

    It also fixes the relation between SLAB's thermal velocity and the
    standard convective velocity scale.  With ``phi = rho V_H cp dT`` the
    definition ``w_*^3 = g h phi/(rho cp T)`` reduces to
    ``w_*^3 = g h V_H dT / T``, so EQ 35e's
    ``U_t^3 = C_t g dT V_H h / T`` gives ``U_t = C_t^(1/3) w_* = 0.519 w_*``.
    Any comparison of SLAB's convective entrainment with measured values has
    to apply that factor, or it overstates the entrainment by 1.93.
    """
    """Thermal convection constant `Ct`, EQ 35e."""

    c_drag_lofted: float = 0.039
    """Drag coefficient for lofted plumes, TNO 4.134-4.135."""

    # -- gravity spreading -------------------------------------------------
    alpha_g: float = 0.25
    """Gravity-flow pressure coefficient `alpha_g` in EQ 4 (JS `alfg`).
    Set to 0 for lofted clouds."""

    alpha_gv: float = 0.75
    """Gravity-flow coefficient in the cross-wind momentum EQ 5a (JS `alfgv`)."""

    c_source_shear: float = 0.05
    """
    Source-jet shear contribution to the in-cloud friction velocity,
    U_ss^2 = c * W_s * Ua (JS `cws`).  EQ 35e as scanned reads 0.5; the
    reference implementation uses 0.05 and is taken as authoritative.
    """

    x_relax_source: float = 1.0
    """Scale of the exponential relaxation of U_bs^2 away from the source
    region (JS `xstr`); set per-source, not a free constant."""

    # -- plume rise --------------------------------------------------------
    hmp_rise: float = 1.32
    """Dense-jet max plume rise prefactor, EQ 45a. Hoot-Meroney-Peterka (1973)."""

    hmp_conc: float = 1.69
    """Dense-jet peak concentration prefactor, EQ 45c."""

    hmp_conc_exp: float = 1.85
    """Dense-jet peak concentration exponent, EQ 45c."""

    hmp_xpr: float = 0.435
    """Down-wind location of max rise, EQ 45b."""

    briggs_beta0: float = 0.4
    """Briggs entrainment `beta = beta0 + beta1*Ua/Ws`, EQ 46b."""

    briggs_beta1: float = 1.2
    """See above, EQ 46b."""

    briggs_buoyant: float = 1.2
    """Buoyant-jet plume rise prefactor, EQ 46c."""

    aspect_ratio: float = 0.6
    """Assumed cloud height / width after plume rise, EQ 45e-45f."""

    # -- atmosphere --------------------------------------------------------
    mix_height_ref: float = 130.0
    """Mixing-layer height prefactor [m]: H = 130 * 2^(7-s), section 2.5.1."""

    phi_stable: float = 5.0
    """Coefficient in the stable MO function Phi_m = 1 + 5 z/L, EQ 33a."""

    phi_unstable: float = 16.0
    """Coefficient in the unstable MO function, EQ 33b."""

    z_transition: float = 2.71828182845904523536
    """Near-ground matching height in units of z0 (= e), EQ 34c."""

    # -- numerics (not physics, but must be recorded) ----------------------
    newton_tol: float = 1.0e-3
    """Convergence tolerance of the thermodynamic Newton loop."""

    newton_max_iter: int = 15
    """Iteration cap of the thermodynamic Newton loop."""

    eval_tol: float = 1.0e-3
    """Convergence tolerance of the outer U-h-Uab fixed-point loop."""

    eval_max_iter: int = 11
    """Iteration cap of the outer fixed-point loop."""

    # -- provenance --------------------------------------------------------
    name: str = "ermak90"
    """Identifier recorded in the Trajectory metadata."""

    # ------------------------------------------------------------------
    def perturb(self, **kw: Any) -> "Coefficients":
        """Return a copy with selected fields replaced (for UQ / ablation)."""
        unknown = set(kw) - set(asdict(self))
        if unknown:
            raise KeyError(f"unknown coefficient(s): {sorted(unknown)}")
        kw.setdefault("name", f"{self.name}+mod")
        return replace(self, **kw)

    def scaled(self, factor: float, *fields: str) -> "Coefficients":
        """Return a copy with `fields` multiplied by `factor` (sensitivity runs)."""
        d = asdict(self)
        return self.perturb(
            **{f: d[f] * factor for f in fields},
            name=f"{self.name}*{factor:g}({','.join(fields)})",
        )

    def as_dict(self) -> dict:
        return asdict(self)


COEFFS = Coefficients()

#: Named coefficient sets used by the ablation study.
PRESETS: dict[str, Coefficients] = {
    # Manual / TNO values.  Default.
    "ermak90": COEFFS,
    # The value TNO's Yellow Book implies for the same coefficient.  Its
    # cross-wind shear constant is 0.209, and 0.209 = sqrt(0.0195) * 1.5 to
    # four figures, against sqrt(0.02) * 1.5 = 0.2121.
    #
    # This was originally taken for a rounding defect in the code and the
    # default was set to 0.0195.  Checking against Ermak's Fortran
    # (SLAB.FOR line 381) shows the code has always used 0.02,
    # so the discrepancy is between the code and the TNO documentation of it,
    # not an error in either.  The default now follows the code; this preset
    # keeps the documented value available.
    "tno": COEFFS.perturb(c_drag_top=0.0195, name="tno"),
    # Nominal gravity-front entrainment, for sensitivity work only.  The
    # value is a placeholder chosen to make the new term comparable to the
    # ambient-turbulence term at mid-range, NOT a calibrated result.
    "frontal": COEFFS.perturb(alpha_front=0.05, name="frontal"),
    # Stratification damping set by the canonical shear-flow limits rather
    # than by Ermak's choice; see `c_mu_strat`.  Determined without any
    # dispersion data — and refuted by it.
    "canonical": COEFFS.perturb(c_mu_strat=0.060, name="canonical"),
    # The other end of the same axis: the damping strength at which the
    # between-trial scatter on Burro is smallest, D = 0.0625, half Ermak's.
    # Fitted to dispersion data and labelled as such.
    "field_damping": COEFFS.perturb(c_mu_strat=0.0125, name="field_damping"),
    # Rainout removal rate scaled to the SMEDIS droplet-lifetime target for
    # Desert Tortoise.  Calibrated against a droplet measurement, not against
    # dispersion statistics, so field concentrations remain a test of it.
    "rainout_smedis": COEFFS.perturb(rainout_efficiency=0.25,
                                     name="rainout_smedis"),
}


def preset(name: str) -> Coefficients:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; have {sorted(PRESETS)}") from None
