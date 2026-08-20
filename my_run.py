"""
내 시나리오 실행 파일
=====================

slabx 루트에서:
    python my_run.py

Node.js 불필요 — 원본 SLAB 대조에만 쓰입니다.

아래 SCENARIO 딕셔너리만 고치면 됩니다.
"""

import numpy as np

from slabx.coefficients import COEFFS, preset
from slabx.core.plume import run_dispersion
from slabx.core.source import (
    EvaporatingPool, HorizontalJet, InstantaneousRelease,
)
from slabx.core.vertical_jet import VerticalJet
from slabx.post.concentration import concentration_field
from slabx.scope import describe_scope
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import LegacyThermo, Substance, water_backend

# ===========================================================================
# 물질 라이브러리 — 필요한 것만 쓰세요
# ===========================================================================
LNG = Substance(name="LNG", mw=0.016043, cp_vapour=2238.0, cp_liquid=3348.5,
                dh_vap=509900.0, T_boil=111.7, rho_liquid=424.1)

AMMONIA = Substance(name="NH3", mw=0.017031, cp_vapour=2045.9,
                    cp_liquid=4611.8, dh_vap=1170000.0, T_boil=239.57,
                    rho_liquid=603.0, sat_B=2976.01)

CHLORINE = Substance(name="Cl2", mw=0.070906, cp_vapour=498.1,
                     cp_liquid=926.3, dh_vap=287840.0, T_boil=239.1,
                     rho_liquid=1574.0, sat_B=1978.34, sat_C=-27.01)

PROPANE = Substance(name="Propane", mw=0.044096, cp_vapour=1670.0,
                    cp_liquid=2520.0, dh_vap=425700.0, T_boil=231.1,
                    rho_liquid=580.0)


# ===========================================================================
# ▼▼▼ 여기만 고치세요 ▼▼▼
# ===========================================================================
SCENARIO = dict(
    name="Burro 8 (LNG 증발풀, 저풍속·안정)",
    # 조건은 시험 기록 기준입니다. SLAB 매뉴얼 §4.1 예제는 시연용 기상
    # (T=306 K, RH=4.6 %) 을 쓰므로 검증값과 다른 답이 나옵니다.

    # --- 기상 --------------------------------------------------------
    substance=LNG,
    u_ref=1.94,        # 풍속 [m/s]
    z_ref=3.0,         # 그 풍속을 잰 높이 [m]
    T=290.0,           # 기온 [K]   (섭씨 + 273.15)
    rh=50.0,           # 상대습도 [%]
    z0=2e-4,           # 지표 조도 [m]  물 2e-4 / 사막 3e-3 / 초지 0.03
    stability="E",     # "A"~"F".  대신 inv_L=0.0665 처럼 1/L 직접 지정 가능
    inv_L=None,        # 쓰려면 stability=None 으로

    # --- 소스 (아래 넷 중 하나만 채우세요) ---------------------------
    source="pool",     # "pool" | "hjet" | "vjet" | "puff"

    rate=116.93,       # 방출률 [kg/s]   (puff 는 무시)
    # 웅덩이 면적 또는 팽창 후 제트 단면적 [m2].
    # 증발 웅덩이는 방출률을 증발플럭스로 나눈 값이 맞습니다 --
    # 매뉴얼이 적어둔 657 m2 를 그대로 쓰면 LFL 거리가 60 m 짧아집니다.
    area=116.93 / 0.167,
    duration=107.0,    # 지속시간 [s]    (puff 는 무시)
    mass=0.0,          # puff 전용: 총 방출량 [kg]
    liquid_fraction=0.0,   # 제트 전용: 액체 질량분율 0~1
    height=0.0,        # 방출 높이 [m]
    T_source=None,     # 소스 온도 [K]. None 이면 비점

    # --- 계산 범위 ---------------------------------------------------
    x_max=1000.0,      # 계산 거리 [m]
    sensor_z=1.0,      # 농도를 평가할 높이 [m]  ("max" 도 가능)
    t_avg=80.0,        # 평균화 시간 [s]
    levels=(0.05, 0.025),   # 관심 농도 (부피분율).  LNG LFL=0.05

    # --- 추가 물리 (전부 기본 꺼짐) ----------------------------------
    real_properties=False,   # CoolProp 실물성
    rainout=False,           # 레인아웃 (2상에서만 효과)
    kinetic_evaporation=False,   # 유한속도 액적증발
    added_mass=False,        # 부가질량 (부력 구름에서만)
    substrate=None,          # "dry_soil" | "concrete" | "water" | None

    # --- 계수 --------------------------------------------------------
    coeffs="ermak90",  # "ermak90" | "canonical" | "field_damping"
                       # | "rainout_smedis" | "legacy_js"
)
# ===========================================================================
# ▲▲▲ 여기까지 ▲▲▲
# ===========================================================================


def build(cfg):
    """설정 딕셔너리를 실제 객체로."""
    sub = cfg["substance"]

    atm = Atmosphere(
        u_ref=cfg["u_ref"], z_ref=cfg["z_ref"], T=cfg["T"], rh=cfg["rh"],
        z0=cfg["z0"],
        stability=cfg.get("stability"),
        inv_L=cfg.get("inv_L") or 0.0,
    )

    kind = cfg["source"]
    common = dict(substance=sub, rate=cfg["rate"], area=cfg["area"],
                  duration=cfg["duration"], height=cfg["height"])
    if cfg.get("T_source") is not None:
        common["T_source"] = cfg["T_source"]

    if kind == "pool":
        src = EvaporatingPool(**{k: v for k, v in common.items()
                                 if k != "height"})
    elif kind == "hjet":
        src = HorizontalJet(liquid_fraction=cfg["liquid_fraction"], **common)
    elif kind == "vjet":
        src = VerticalJet(liquid_fraction=cfg["liquid_fraction"], **common)
    elif kind == "puff":
        src = InstantaneousRelease(substance=sub, rate=0.0, duration=0.0,
                                   area=cfg["area"], mass=cfg["mass"],
                                   height=cfg["height"])
    else:
        raise ValueError(f"source 는 pool/hjet/vjet/puff 중 하나: {kind!r}")

    if cfg["real_properties"]:
        from slabx.thermo.coolprop import COOLPROP_NAMES, CoolPropThermo, \
            coolprop_water
        fluid = COOLPROP_NAMES.get(sub.name)
        em, wt = CoolPropThermo(sub, fluid=fluid), coolprop_water()
    else:
        em, wt = LegacyThermo(sub), water_backend()

    sub_map = {"dry_soil": "DRY_SOIL", "concrete": "CONCRETE",
               "water": "WATER_SUBSTRATE"}
    substrate = None
    if cfg.get("substrate"):
        import slabx.submodels.ground as G
        substrate = getattr(G, sub_map[cfg["substrate"]])

    return atm, src, em, wt, substrate


def main():
    cfg = SCENARIO
    atm, src, em, wt, substrate = build(cfg)

    traj, used = run_dispersion(
        src, atm, em, wt,
        x_max=cfg["x_max"], n_puff_steps=40,
        coeffs=preset(cfg["coeffs"]),
        rainout=cfg["rainout"],
        kinetic_evaporation=cfg["kinetic_evaporation"],
        added_mass=cfg["added_mass"],
        substrate=substrate,
    )

    field = concentration_field(
        traj, atm, z=cfg["sensor_z"], t_avg=cfg["t_avg"],
        t_release=cfg["duration"] if cfg["source"] != "puff" else 0.0,
    )

    # -- 요약 ------------------------------------------------------------
    print(f"\n{'=' * 66}\n{cfg['name']}\n{'=' * 66}")
    print(f"  대기      u*={atm.u_star:.3f} m/s   안정도 s={atm.s:.2f}   "
          f"혼합층={atm.h_mix:.0f} m   rho_a={atm.rho:.4f}")
    print(f"  소스      반폭 {src.half_width_geometric:.2f} → "
          f"{used.half_width:.2f} m")
    print(f"  모드      {sorted(set(traj.mode.tolist()))}   "
          f"전이 t={traj.meta.get('transition_t', '—')}")

    print(f"\n  관심 농도까지의 거리")
    for lv in cfg["levels"]:
        print(f"    {lv * 100:5.2f} %   {field.distance_to(lv):8.0f} m")

    # -- 프로파일 --------------------------------------------------------
    lo = max(traj.x[traj.x > 0].min(), 1.0)
    xs = np.geomspace(lo, traj.x.max(), 10)
    print(f"\n{'x [m]':>9}{'C [%]':>9}{'T [K]':>8}{'rho/rhoa':>10}"
          f"{'B [m]':>9}{'h [m]':>8}{'z_c [m]':>9}")
    for x in xs:
        c = float(np.exp(np.interp(x, field.x,
                                   np.log(np.maximum(field.peak, 1e-30)))))
        i = int(np.argmin(np.abs(traj.x - x)))
        print(f"{x:9.1f}{c * 100:9.3f}{traj.T[i]:8.1f}"
              f"{traj.rho[i] / atm.rho:10.4f}{traj.b_half[i]:9.1f}"
              f"{traj.h[i]:8.2f}{traj.z_c[i]:9.2f}")

    # -- CSV 저장 --------------------------------------------------------
    out = "my_run_output.csv"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("x_m,C_volfrac,C_pct,T_K,rho,B_m,h_m,z_c_m,u_m_s\n")
        for i in range(len(traj)):
            if traj.x[i] <= 0:
                continue
            c = float(np.exp(np.interp(traj.x[i], field.x,
                                       np.log(np.maximum(field.peak, 1e-30)))))
            fh.write(f"{traj.x[i]:.4f},{c:.6e},{c * 100:.6f},{traj.T[i]:.3f},"
                     f"{traj.rho[i]:.5f},{traj.b_half[i]:.4f},"
                     f"{traj.h[i]:.4f},{traj.z_c[i]:.4f},{traj.u[i]:.4f}\n")
    print(f"\n  → {out} 저장 ({len(traj)} 행)")


if __name__ == "__main__":
    import warnings

    from slabx.scope import ScopeWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        main()
        scope_warnings = [w for w in caught
                          if issubclass(w.category, ScopeWarning)]

    if scope_warnings:
        print(f"\n{'!' * 66}")
        print("검증 범위 밖 입력입니다 — 결과는 나오지만 검증되지 않았습니다:")
        for w in scope_warnings:
            print(f"  · {w.message}")
        print(f"{'!' * 66}")
        print(describe_scope())
