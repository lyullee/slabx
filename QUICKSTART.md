# 직접 돌려보기

압축 파일 하나(`slabx.zip`, 558 KB)만 받으면 됩니다. 아티팩트를 개별로
받을 필요는 없습니다 — 그건 대화 중 변경사항을 보여준 것이고, 이 압축본에
전부 들어 있습니다.

---

## 1. 준비 (5분)

모델만 쓰실 거면 한 줄입니다.

```bash
pip install slabx                  # numpy, scipy
pip install "slabx[thermo]"        # 실물성(CoolProp)까지
```

시험군까지 돌리시려면 저장소를 받으십시오 — `tests/` 는 패키지에
들어가지 않습니다.

```bash
git clone https://github.com/lyullee/slabx.git
cd slabx

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[thermo,dev]"
```

**원본 SLAB 대조는 선택**입니다. 없어도 나머지는 다 돌아갑니다 (관련
테스트만 skip). 하려면 `gfortran` 과 원본 소스가 필요한데, 원본은 비상업
조건이 붙어 있어 이 배포본에 포함되지 않습니다 —
`golden/fortran/README.md` 에 구하는 법과 빌드 명령이 있습니다.

---

## 2. 30초 확인

```bash
PYTHONPATH=src python3 -m pytest -q
```

기대 결과: 실패 0, `xfailed` 1, 그리고 다수의 `skipped`.

건너뛴 시험은 원본 Fortran, 선택 의존성(CoolProp), 또는 제3자 관측자료가
없어서입니다. 무엇을 갖추었는지에 따라 개수가 달라지며, **없다고 실패하지는
않습니다.**

`xfailed` 1건은 **의도된 것**입니다 — 알려진 실패(고밀도 저속 제트)를
재현 데크와 함께 남겨둔 것입니다.

CoolProp이 없으면 `skipped` 가 늘어납니다. 정상입니다.

---

## 3. 원본 대조 (선택)

```bash
PYTHONPATH=src python3 examples/fortran_reference.py
```

```
burro8  —  LNG  (idspl=1)
    변수 established  near-source   최대   원본 격자   판정
     cv      0.55%           —    2.45%     0.27%     OK
... (5개 전부)
```

**5개 전부 OK 여야 합니다.** 이게 무너지면 나머지 검증이 전부 무효입니다.

(`golden/fortran/slab` 빌드 필요)

---

## 4. 검증 결과 보기

```bash
# 실측 대조 + 사전등록 판정 (약 3분)
PYTHONPATH=src python3 examples/validate_burro.py

# 가설 배제 5단계 — 소스면적·미앤더·확장·분배·중력감쇠
PYTHONPATH=src python3 examples/diagnose_burro.py

# 정준흐름 극한 4종 (실험실 vs 현장 모순)
PYTHONPATH=src python3 examples/entrainment_limits.py

# 무작위 차등검정 — 원본과 임의 입력 대조 (Fortran 필요, 느림)
PYTHONPATH=src python3 examples/fuzz_fortran.py 20 21

# 실물성 ablation
PYTHONPATH=src python3 examples/ablation_coolprop.py
```

---

## 5. 적용범위 확인

```bash
PYTHONPATH=src python3 -c "from slabx.scope import describe_scope; print(describe_scope())"
```

검증된 범위와, 표현 자체가 없는 조건(장애물·지형·축척변환)을 출력합니다.

---

## 6. 직접 써보기

```python
import sys; sys.path.insert(0, "src")

from slabx.submodels.atmosphere import Atmosphere
from slabx.core.source import EvaporatingPool
from slabx.core.plume import run_dispersion
from slabx.post.concentration import concentration_field
from slabx.thermo.base import Substance, LegacyThermo, water_backend

LNG = Substance(name="LNG", mw=0.016043, cp_vapour=2238.0,
                cp_liquid=3348.5, dh_vap=509900.0,
                T_boil=111.7, rho_liquid=424.1)

# Burro 8: 저풍속·안정 — 규제상 가장 중요한 조건
atm = Atmosphere(u_ref=1.94, z_ref=3.0, T=290.0, rh=50.0,
                 z0=2e-4, stability="E")
src = EvaporatingPool(substance=LNG, rate=116.93,
                      area=116.93 / 0.167, duration=107.0)

traj, used = run_dispersion(src, atm, LegacyThermo(LNG), water_backend(),
                            x_max=1000.0, n_puff_steps=40)

field = concentration_field(traj, atm, z=1.0, t_avg=80.0, t_release=107.0)

print(f"LFL(5%) 거리 = {field.distance_to(0.05):.0f} m   (실측 455 m)")
print(f"소스 반폭  {src.half_width_geometric:.1f} → {used.half_width:.1f} m")
```

### 추가 물리 켜기 (전부 기본 꺼짐)

```python
from slabx.thermo.coolprop import CoolPropThermo, coolprop_water
from slabx.submodels.ground import DRY_SOIL

traj, used = run_dispersion(
    src, atm,
    CoolPropThermo(LNG, fluid="Methane"), coolprop_water(),   # 실물성
    x_max=1000.0,
    rainout=True,              # 레인아웃
    kinetic_evaporation=True,  # 유한속도 액적증발
    added_mass=True,           # 부가질량
    substrate=DRY_SOIL,        # 지반 열전달
)
```

### 계수 바꿔보기

```python
from slabx.coefficients import COEFFS, preset

print(preset("canonical").c_mu_strat)      # 0.060 — 실험실 정합값
print(preset("field_damping").c_mu_strat)  # 0.0125 — 현장 최적

my = COEFFS.perturb(c_mu_strat=0.04, name="mine")
traj, _ = run_dispersion(src, atm, LegacyThermo(LNG), water_backend(),
                         x_max=1000.0, coeffs=my)
```

---

## 7. 문서

```
README.md           ← 여기부터
LIMITATIONS.md      못 하는 것, 근거 포함
VALIDATION.md       38시험 결과 + 재현 명령
docs/00_INDEX.md    상세 핸드오프
docs/01_THEORY.md            이론 구조
docs/02_VALIDATION_REFERENCE.md   원본 대조
docs/03_VALIDATION_FIELD.md       실측 검증
docs/04_DEVELOPMENT.md            개발 과정
docs/05_PAPER_BRIEF.md            논문 명령서
docs/06_RESULTS_TABLES.md         숫자 + 참고문헌
```

---

## 문제가 생기면

| 증상 | 원인 |
|---|---|
| `ModuleNotFoundError: slabx` | `PYTHONPATH=src` 를 빼먹음 |
| CoolProp 테스트가 skip | `pip install CoolProp` |
| 원본 대조가 skip됨 | `golden/fortran/slab` 미빌드. 정상입니다 |
| `ScopeWarning` | 검증범위 밖 입력. 정상 동작 |
| 테스트가 4분 넘게 걸림 | 정상입니다 (실측 검증이 모델을 여러 번 돌립니다) |
