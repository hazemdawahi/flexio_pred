# smartpay_api.py
# ---------------------------------------------------------------------
# FastAPI micro-service – prédit le nombre de mois (terms) idéal
# pour rembourser un achat, à partir des préférences SmartPay.
# ---------------------------------------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from decimal import Decimal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import math                        # NEW

app = FastAPI(title="SmartPay Term Advisor")

# ---------- Pydantic --------------------------------------------------
class AdviceRequest(BaseModel):
    price: float = Field(..., gt=0, description="Purchase price")
    preference: Dict[str, Any]           # Preferences document


class AdviceResponse(BaseModel):
    terms: int            # durée en MOIS
    model_type: str       # rule | interp | linear | polynomial
    mae: float            # modèle MAE (0 for rule / interp)
    model_config = {"protected_namespaces": ()}

# ---------- Plages de prix -------------------------------------------
PRICE_BANDS = {
    "under150":          (0, 150),
    "range300to500":     (300, 500),
    "range1000to2000":   (1000, 2000),
    "range5000to10000":  (5000, 10000),
}

# ---------- Contraintes globales -------------------------------------
MIN_TERMS: int = 1       # 1 mois
MAX_TERMS: int = 60      # 60 mois (5 ans)

# ---------- Utilitaires ----------------------------------------------
def _f(x):                        # Decimal → float
    return float(x) if isinstance(x, Decimal) else x


def band_for(price: float):
    for b, (lo, hi) in PRICE_BANDS.items():
        if lo <= price <= hi:
            return b
    return None


def clamp(price: float, pref: Dict[str, Any], months: float) -> int:
    """
    Limite la prédiction :
      • aux min/max déclarés pour la bande de prix (si dispo)
      • puis aux bornes globales MIN_TERMS / MAX_TERMS
      • ENFIN, arrondit *vers le bas* (floor) pour rester conservateur
    """
    b = band_for(price)
    if b:
        mn, mx = f"{b}DurationMin", f"{b}DurationMax"
        if mn in pref and mx in pref:
            dmin, dmax = _f(pref[mn]), _f(pref[mx])
            months = max(dmin, min(dmax, months))

    # Arrondi conservateur
    months_int = int(math.floor(months + 1e-9))
    return max(MIN_TERMS, min(MAX_TERMS, months_int))


def build_dataset(pref: Dict[str, Any], samples: int = 5) -> pd.DataFrame:
    """
    Génére un jeu de données synthétique (montant ↔ durée) pour entraîner
    une régression quand aucune règle simple n'est applicable.
    """
    rows: List[Dict[str, Any]] = []
    for band, (lo, hi) in PRICE_BANDS.items():
        mn, mx = f"{band}DurationMin", f"{band}DurationMax"
        if mn not in pref or mx not in pref:
            continue

        dmin, dmax = _f(pref[mn]), _f(pref[mx])
        for i in range(samples):
            p = (i + 1) / (samples + 1)          # échantillonnage uniforme
            rows.append({
                "amount":   lo + p * (hi - lo),
                "duration": dmin + p * (dmax - dmin),
            })
  
    if not rows:
        raise ValueError("Les préférences ne contiennent aucune bande exploitable.")
    return pd.DataFrame(rows)


def best_duration_pipe(X: np.ndarray, y: np.ndarray):
    """
    Sélectionne le meilleur modèle (linéaire ou polynômial d'ordre 2)
    en fonction de la MAE.
    """
    candidates = {
        "linear": Pipeline([
            ("poly",  PolynomialFeatures(degree=1, include_bias=False)),
            ("scale", StandardScaler()),
            ("reg",   LinearRegression())
        ]),
        "polynomial": Pipeline([
            ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("reg",   LinearRegression())
        ]),
    }
    maes = {name: mean_absolute_error(y, pipe.fit(X, y).predict(X))
            for name, pipe in candidates.items()}
    best_name = min(maes, key=maes.get)
    return candidates[best_name], maes[best_name], best_name

# ---------- Endpoint /advise -----------------------------------------
@app.post("/advise", response_model=AdviceResponse)
def advise(req: AdviceRequest):
    price, pref = req.price, req.preference

    # 1) Raccourci « règle fixe » : durée égale min = max
    b = band_for(price)
    if b:
        mn, mx = f"{b}DurationMin", f"{b}DurationMax"
        if mn in pref and mx in pref and pref[mn] == pref[mx]:
            months = _f(pref[mn])
            months = clamp(price, pref, months)
            return AdviceResponse(
                terms=months,
                model_type="rule",
                mae=0.0,
            )

    # 2) Interpolation linéaire *à l'intérieur* de la bande
    if b:
        mn, mx = f"{b}DurationMin", f"{b}DurationMax"  
        lo, hi  = PRICE_BANDS[b]
        if mn in pref and mx in pref:
            dmin, dmax = _f(pref[mn]), _f(pref[mx])
            # Si les bornes sont distinctes, on interpole
            if dmin != dmax:
                frac    = (price - lo) / (hi - lo)
                months  = dmin + frac * (dmax - dmin)
                months  = clamp(price, pref, months)
                return AdviceResponse(
                    terms=months,
                    model_type="interp",
                    mae=0.0,
                )

    # 3) Régression synthétique (fallback général)
    df              = build_dataset(pref)
    X               = df[["amount"]].values
    y_months        = df["duration"].values
    dpipe, mae, typ = best_duration_pipe(X, y_months)

    raw_months = float(dpipe.predict([[price]])[0])
    months     = clamp(price, pref, raw_months)

    return AdviceResponse(
        terms=months,
        model_type=typ,
        mae=mae,
    )
