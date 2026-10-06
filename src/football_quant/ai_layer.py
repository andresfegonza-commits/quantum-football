from __future__ import annotations

from .domain import Edge, MatchSnapshot, RiskDecision


class QuantExplanationLayer:
    def explain(self, match: MatchSnapshot, edge: Edge, risk: RiskDecision) -> str:
        side_name = match.home.name if edge.quote.side == "home" else match.away.name
        opposing_name = match.away.name if edge.quote.side == "home" else match.home.name
        inputs = edge.estimate.inputs

        drivers = [
            f"Elo adjusted difference: {inputs.get('elo_diff', 0.0):.1f}",
            f"recent form difference: {inputs.get('form_diff', 0.0):.2f}",
            f"market implied probability: {edge.implied_probability:.2%}",
            f"model fair probability: {edge.estimate.fair_probability:.2%}",
            f"expected value: {edge.expected_value:.2%}",
        ]
        risk_text = "; ".join(risk.reasons)
        return (
            f"{side_name} {edge.quote.line:+.2f} vs {opposing_name} is classified as "
            f"{risk.status.value}. Main drivers: {', '.join(drivers)}. "
            f"Risk decision: stake {risk.stake:.2f}, max loss {risk.max_loss:.2f}. {risk_text}"
        )
