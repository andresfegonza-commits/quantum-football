from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from .domain import BetSide


@dataclass(frozen=True)
class HandicapLeg:
    line: float
    stake_fraction: float
    outcome: str
    profit_fraction: float


@dataclass(frozen=True)
class HandicapSettlement:
    side: BetSide
    line: float
    odds: float
    stake: float
    profit: float
    return_amount: float
    legs: tuple[HandicapLeg, ...]

    @property
    def roi(self) -> float:
        return self.profit / self.stake if self.stake else 0.0


def split_quarter_line(line: float) -> tuple[float, ...]:
    """Split quarter-goal lines into their two half-stake components."""
    scaled = Decimal(str(line)) * Decimal("4")
    if scaled != scaled.to_integral_value():
        raise ValueError(f"Asian handicap line must move in quarter goals: {line}")

    line_decimal = Decimal(str(line))
    quarter_remainder = abs(int(scaled)) % 2
    if quarter_remainder == 0:
        return (float(line_decimal),)

    lower = (line_decimal - Decimal("0.25")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    upper = (line_decimal + Decimal("0.25")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return (float(lower), float(upper))


def settle_leg(goal_difference: int, line: float, odds: float) -> tuple[str, float]:
    adjusted_margin = goal_difference + line
    if adjusted_margin > 0:
        return "win", odds - 1.0
    if adjusted_margin == 0:
        return "push", 0.0
    return "loss", -1.0


def settle_asian_handicap(
    *,
    side: BetSide,
    line: float,
    odds: float,
    home_goals: int,
    away_goals: int,
    stake: float = 1.0,
) -> HandicapSettlement:
    if odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    if stake <= 0:
        raise ValueError("Stake must be positive")

    raw_margin = home_goals - away_goals
    goal_difference = raw_margin if side == "home" else -raw_margin
    legs = split_quarter_line(line)
    stake_fraction = 1.0 / len(legs)

    settled_legs: list[HandicapLeg] = []
    profit_ratio = 0.0
    for leg_line in legs:
        outcome, leg_profit = settle_leg(goal_difference, leg_line, odds)
        profit_ratio += leg_profit * stake_fraction
        settled_legs.append(
            HandicapLeg(
                line=leg_line,
                stake_fraction=stake_fraction,
                outcome=outcome,
                profit_fraction=leg_profit * stake_fraction,
            )
        )

    profit = stake * profit_ratio
    return HandicapSettlement(
        side=side,
        line=line,
        odds=odds,
        stake=stake,
        profit=profit,
        return_amount=stake + profit,
        legs=tuple(settled_legs),
    )
