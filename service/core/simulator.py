"""5대 밀링 기계의 센서 값을 실시간으로 만들어내는 시뮬레이터.

ai4i2020 데이터의 분포(주변온도 ~300K, 공정온도 = 주변온도+10K,
회전속도 ~1500rpm, 토크 ~40Nm, 공구마모 0~250분)를 따르는 랜덤 워크에
간헐적인 '고부하 이벤트'를 섞어 고장 징후가 나타나도록 만든다.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

MACHINE_NAMES = ["MILL-01", "MILL-02", "MILL-03", "MILL-04", "MILL-05"]
GRADE_BY_MACHINE = {"MILL-01": 1, "MILL-02": 2, "MILL-03": 3, "MILL-04": 1, "MILL-05": 2}

# 제품등급별 1스텝당 공구마모 증가량 (H 등급일수록 가공 강도가 높다)
WEAR_PER_STEP = {1: 2.0, 2: 3.0, 3: 5.0}
WEAR_LIMIT = 240.0  # 이 값을 넘으면 공구 교체


@dataclass
class Machine:
    name: str
    제품등급: int
    주변온도_K: float = 300.0
    공정온도_K: float = 310.0
    회전속도_rpm: float = 1500.0
    토크_Nm: float = 40.0
    공구마모_분: float = 0.0
    stress_steps: int = 0        # 남은 고부하 이벤트 스텝 수
    tool_changes: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=120))

    def _drift(self, value, target, rate, noise, lo, hi):
        value += (target - value) * rate + random.gauss(0, noise)
        return min(max(value, lo), hi)

    def step(self, stress_rate: float = 0.06, load_scale: float = 1.0) -> dict:
        """센서 값을 한 틱 진행시키고 현재 상태를 dict 로 반환."""
        if self.stress_steps == 0 and random.random() < stress_rate:
            self.stress_steps = random.randint(3, 8)   # 고부하 구간 진입

        under_stress = self.stress_steps > 0
        if under_stress:
            self.stress_steps -= 1

        torque_target = (58.0 if under_stress else 40.0) * load_scale
        rpm_target = 1250.0 if under_stress else 1520.0
        temp_gap_target = 12.0 if under_stress else 10.0

        self.주변온도_K = self._drift(self.주변온도_K, 300.0, 0.15, 0.25, 295.0, 304.5)
        gap = self.공정온도_K - self.주변온도_K
        gap = self._drift(gap, temp_gap_target, 0.25, 0.15, 7.6, 13.0)
        self.공정온도_K = self.주변온도_K + gap

        self.회전속도_rpm = self._drift(self.회전속도_rpm, rpm_target, 0.3, 60.0, 1160.0, 2880.0)
        self.토크_Nm = self._drift(self.토크_Nm, torque_target, 0.3, 2.5, 3.8, 78.0)

        self.공구마모_분 += WEAR_PER_STEP[self.제품등급] * (1.4 if under_stress else 1.0)
        replaced = False
        if self.공구마모_분 >= WEAR_LIMIT:
            self.공구마모_분 = 0.0
            self.tool_changes += 1
            replaced = True

        return {
            "설비명": self.name,
            "시각": datetime.now(),
            "제품등급": self.제품등급,
            "주변온도_K": round(self.주변온도_K, 2),
            "공정온도_K": round(self.공정온도_K, 2),
            "회전속도_rpm": round(self.회전속도_rpm, 1),
            "토크_Nm": round(self.토크_Nm, 2),
            "공구마모_분": round(self.공구마모_분, 1),
            "고부하": int(under_stress),
            "공구교체": int(replaced),
        }

    def replace_tool(self):
        self.공구마모_분 = 0.0
        self.tool_changes += 1


class MachineFleet:
    """5대 설비를 한꺼번에 진행시키고 이력을 관리한다."""

    def __init__(self, names: list[str] | None = None, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        names = names or MACHINE_NAMES
        self.machines: dict[str, Machine] = {}
        for name in names:
            grade = GRADE_BY_MACHINE.get(name, 1)
            self.machines[name] = Machine(
                name=name,
                제품등급=grade,
                공구마모_분=float(random.randint(0, 180)),
                토크_Nm=random.uniform(34, 46),
                회전속도_rpm=random.uniform(1400, 1650),
            )
        self.tick = 0
        self.alerts: deque = deque(maxlen=200)

    def step(self, stress_rate: float = 0.06, load_scale: float = 1.0) -> pd.DataFrame:
        self.tick += 1
        rows = [m.step(stress_rate, load_scale) for m in self.machines.values()]
        return pd.DataFrame(rows)

    def record(self, df: pd.DataFrame):
        """예측 결과가 붙은 행을 설비별 이력에 저장하고 경보를 남긴다."""
        for row in df.to_dict("records"):
            machine = self.machines[row["설비명"]]
            machine.history.append(row)
            if row.get("고장예측") == 1:
                self.alerts.appendleft(
                    {
                        "시각": row["시각"],
                        "설비명": row["설비명"],
                        "고장확률": row["고장확률"],
                        "토크_Nm": row["토크_Nm"],
                        "회전속도_rpm": row["회전속도_rpm"],
                        "공구마모_분": row["공구마모_분"],
                    }
                )

    def history_df(self, name: str | None = None) -> pd.DataFrame:
        if name:
            return pd.DataFrame(list(self.machines[name].history))
        frames = [pd.DataFrame(list(m.history)) for m in self.machines.values()]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def alerts_df(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.alerts))
