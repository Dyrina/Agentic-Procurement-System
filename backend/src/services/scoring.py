from __future__ import annotations

PAYMENT_TERMS_SCORES: dict[str, int] = {
    "net-60": 100,
    "net-45": 75,
    "net-30": 50,
    "net-15": 25,
    "immediate": 0,
}


def min_max_invert(values: list[float], value: float) -> float:
    """Normalize value where lower is better → higher score."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return 100.0
    return (hi - value) / (hi - lo) * 100.0


def payment_terms_score(terms: str) -> int:
    normalized = terms.lower().strip()
    for key, score in PAYMENT_TERMS_SCORES.items():
        if key in normalized:
            return score
    return 50  # unknown


def score_suppliers(
    quotes: list[dict],
    avg_historical_price_sen: float,
    avg_historical_delivery_days: float,
) -> list[dict]:
    """
    Score each supplier. Returns enriched list with score fields + risk_flags + is_recommended.
    Weights: price 55%, delivery 30%, payment_terms 15%.
    """
    prices = [float(q["unit_price_sen"]) for q in quotes]
    deliveries = [float(q["quoted_delivery_days"]) for q in quotes]

    results = []
    for q in quotes:
        p_score = min_max_invert(prices, float(q["unit_price_sen"]))
        d_score = min_max_invert(deliveries, float(q["quoted_delivery_days"]))
        pt_score = float(payment_terms_score(q.get("payment_terms", "Unknown")))
        total = p_score * 0.55 + d_score * 0.30 + pt_score * 0.15

        risk_flags: list[str] = []
        if avg_historical_price_sen > 0:
            ratio = q["unit_price_sen"] / avg_historical_price_sen
            if ratio > 1.10:
                pct = (ratio - 1.0) * 100
                risk_flags.append(f"Price {pct:.0f}% above historical average")
        if avg_historical_delivery_days > 0:
            ratio = q["quoted_delivery_days"] / avg_historical_delivery_days
            if ratio > 1.10:
                pct = (ratio - 1.0) * 100
                risk_flags.append(f"Delivery {pct:.0f}% above historical average")

        results.append({
            **q,
            "price_score": round(p_score, 2),
            "delivery_score": round(d_score, 2),
            "payment_terms_score": round(pt_score, 2),
            "total_score": round(total, 2),
            "risk_flags": risk_flags,
            "is_recommended": False,  # set below
        })

    best = max(results, key=lambda r: r["total_score"])
    for r in results:
        r["is_recommended"] = r["supplier_id"] == best["supplier_id"]

    return results
