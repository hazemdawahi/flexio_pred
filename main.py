# smartpay_api.py
# ---------------------------------------------------------------------
# FastAPI micro-service – prédit le nombre de mois (terms) idéal
# pour rembourser un achat, à partir des préférences SmartPay.
# ---------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal
import numpy as np
from decimal import Decimal
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
import math

app = FastAPI(title="SmartPay Term Advisor")


# ---------- Pydantic --------------------------------------------------
class AdviceRequest(BaseModel):
    price: float = Field(..., gt=0, description="Purchase price")
    preference: Dict[str, Any]  # Preferences document
    rounding_mode: Literal["floor", "round", "ceil"] = Field(
        default="floor",
        description="How to round predicted months"
    )
    include_debug: bool = Field(
        default=False,
        description="Include debug info in response"
    )


class AdviceResponse(BaseModel):
    terms: int  # durée en MOIS
    model_type: str  # rule | interp | piecewise | isotonic
    confidence: float  # 0.0-1.0: how reliable is this prediction
    mae_synthetic: float  # modèle MAE on synthetic breakpoints (0 for rule/interp)
    debug_info: Optional[Dict[str, Any]] = None  # Optional debug information
    model_config = {"protected_namespaces": ()}


# ---------- Plages de prix (boundaries: [lo, hi) except last) --------
# All bands are [lo, hi) (inclusive lo, exclusive hi)
# EXCEPT the last band which includes both endpoints
PRICE_BANDS = {
    "under150": (0, 150),
    "range150to300": (150, 300),
    "range300to500": (300, 500),
    "range500to1000": (500, 1000),
    "range1000to2000": (1000, 2000),
    "range2000to5000": (2000, 5000),
    "range5000to10000": (5000, 10000),
    "over10000": (10000, float("inf")),  # No upper cap for very high prices
}

# ---------- Contraintes globales -------------------------------------
MIN_TERMS: int = 1  # 1 mois
MAX_TERMS: int = 60  # 60 mois (5 ans)
MIN_BREAKPOINT_PRICE: float = 1.0  # Minimum price for breakpoints and predictions


# ---------- Utilitaires ----------------------------------------------
def _f(x):  # Decimal → float
    return float(x) if isinstance(x, Decimal) else x


def band_for(price: float):
    """
    Returns band name if price is inside a band, None otherwise.
    Bands are [lo, hi) except the last one which is [lo, hi].
    """
    band_list = list(PRICE_BANDS.items())

    for i, (band_name, (lo, hi)) in enumerate(band_list):
        is_last_band = (i == len(band_list) - 1)

        if is_last_band:
            # Last band: [lo, hi] (inclusive on both ends)
            if lo <= price <= hi:
                return band_name
        else:
            # Other bands: [lo, hi) (inclusive lo, exclusive hi)
            if lo <= price < hi:
                return band_name

    return None


def apply_rounding(months: float, mode: str) -> int:
    """Apply rounding mode: floor (conservative), round (neutral), ceil (lower monthly)"""
    if mode == "floor":
        return int(math.floor(months + 1e-9))
    elif mode == "ceil":
        return int(math.ceil(months - 1e-9))
    else:  # round
        return int(round(months))


def clamp(price: float, pref: Dict[str, Any], months: float, rounding_mode: str,
          use_band_limits: bool = True) -> int:
    """
    Limite la prédiction :
      • Si use_band_limits=True et price dans une bande: utilise min/max de cette bande
      • Si use_band_limits=False (gap prices): utilise seulement MIN_TERMS/MAX_TERMS
      • ENFIN, arrondit selon le mode (floor/round/ceil)
    """
    if use_band_limits:
        b = band_for(price)
        if b:
            mn, mx = f"{b}DurationMin", f"{b}DurationMax"
            if mn in pref and mx in pref:
                dmin, dmax = _f(pref[mn]), _f(pref[mx])
                months = max(dmin, min(dmax, months))

    # Arrondi selon mode
    months_int = apply_rounding(months, rounding_mode)
    return max(MIN_TERMS, min(MAX_TERMS, months_int))


def calculate_confidence(
        price: float,
        model_type: str,
        amounts: Optional[np.ndarray] = None,
        durations: Optional[np.ndarray] = None,
        band: Optional[str] = None,
        pref: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate confidence score (0.0-1.0) for the prediction.

    High confidence (1.0):
    - Fixed rule within configured band
    - Interpolation within configured band
    - Piecewise/isotonic near breakpoints

    Lower confidence:
    - Extrapolation beyond configured range
    - Large gaps between breakpoints
    - No band configuration for price range
    """
    # Rule and interp within band: highest confidence
    if model_type in ["rule", "interp"] and band:
        return 1.0

    # For piecewise/isotonic, calculate based on proximity to breakpoints
    if model_type in ["piecewise", "isotonic"] and amounts is not None:
        # Check if within configured range
        min_amount, max_amount = amounts[0], amounts[-1]

        if price < min_amount:
            # Extrapolating below range - lower confidence
            distance_ratio = (min_amount - price) / min_amount
            return max(0.3, 1.0 - distance_ratio)

        if price > max_amount:
            # Extrapolating above range - lower confidence
            if max_amount > 0:
                distance_ratio = (price - max_amount) / max_amount
                return max(0.3, 1.0 - min(distance_ratio, 0.7))
            return 0.5

        # Within range: check proximity to nearest breakpoints
        nearest_idx = np.argmin(np.abs(amounts - price))
        distance = abs(amounts[nearest_idx] - price)

        # Check gap size around this point
        if nearest_idx > 0:
            left_gap = amounts[nearest_idx] - amounts[nearest_idx - 1]
        else:
            left_gap = amounts[1] - amounts[0] if len(amounts) > 1 else 0

        if nearest_idx < len(amounts) - 1:
            right_gap = amounts[nearest_idx + 1] - amounts[nearest_idx]
        else:
            right_gap = left_gap

        max_gap = max(left_gap, right_gap)

        # High confidence if close to breakpoint and small gaps
        if max_gap > 0:
            gap_penalty = min(max_gap / 5000, 0.3)  # Penalize large gaps
            distance_penalty = min(distance / max_gap, 0.2)  # Penalize far from breakpoints
            return max(0.6, 1.0 - gap_penalty - distance_penalty)

        return 0.9

    # Default: moderate confidence
    return 0.7


def validate_preferences(pref: Dict[str, Any]):
    """Validate preference consistency"""
    errors = []
    warnings = []

    # Check each band
    for band in PRICE_BANDS.keys():
        mn, mx = f"{band}DurationMin", f"{band}DurationMax"
        if mn in pref and mx in pref:
            dmin, dmax = _f(pref[mn]), _f(pref[mx])
            if dmin > dmax:
                errors.append(f"{band}: DurationMin ({dmin}) > DurationMax ({dmax})")
            if dmin < MIN_TERMS or dmax > MAX_TERMS:
                errors.append(f"{band}: Duration out of global bounds [{MIN_TERMS}, {MAX_TERMS}]")
        elif mn in pref or mx in pref:
            warnings.append(f"{band}: Incomplete configuration (only one of Min/Max set)")

    # Check for large discontinuities between adjacent bands
    sorted_bands = sorted(
        [(band, lo, hi) for band, (lo, hi) in PRICE_BANDS.items()],
        key=lambda x: x[1]
    )

    for i in range(len(sorted_bands) - 1):
        b1, _, hi1 = sorted_bands[i]
        b2, lo2, _ = sorted_bands[i + 1]

        mx1 = f"{b1}DurationMax"
        mn2 = f"{b2}DurationMin"

        if mx1 in pref and mn2 in pref:
            dmax1 = _f(pref[mx1])
            dmin2 = _f(pref[mn2])
            gap = abs(dmax1 - dmin2)
            if gap > 12:  # More than 1 year difference
                warnings.append(
                    f"Large discontinuity between {b1} and {b2}: "
                    f"{dmax1} months vs {dmin2} months (gap: {gap} months)"
                )

    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors, "warnings": warnings})

    return warnings


def build_breakpoints(pref: Dict[str, Any]) -> tuple:
    """
    Build breakpoints (amount, duration) from preference bands.
    For bands with infinite upper bound, use the lower bound only.
    Replaces 0 with MIN_BREAKPOINT_PRICE to avoid odd slope behavior.
    Returns sorted arrays for piecewise interpolation.
    """
    points = []

    for band in sorted(PRICE_BANDS.keys(), key=lambda b: PRICE_BANDS[b][0]):
        lo, hi = PRICE_BANDS[band]
        mn, mx = f"{band}DurationMin", f"{band}DurationMax"

        if mn not in pref or mx not in pref:
            continue

        dmin, dmax = _f(pref[mn]), _f(pref[mx])

        # Replace 0 with MIN_BREAKPOINT_PRICE to avoid odd extrapolation slopes
        if lo == 0:
            lo = MIN_BREAKPOINT_PRICE

        # Add lower breakpoint
        points.append((lo, dmin))

        # Only add upper breakpoint if not infinity
        if not math.isinf(hi):
            points.append((hi, dmax))
        else:
            # For infinite upper bound, add a "far out" point for extrapolation
            # Use a high price with the max duration
            points.append((lo * 10 if lo > 0 else 100000, dmax))

    if not points:
        raise ValueError("No valid preference bands found")

    # Remove duplicates and sort
    points = sorted(set(points), key=lambda x: x[0])

    amounts = np.array([p[0] for p in points])
    durations = np.array([p[1] for p in points])

    return amounts, durations


def piecewise_predict(price: float, amounts: np.ndarray, durations: np.ndarray) -> float:
    """
    Piecewise-linear interpolation between breakpoints.
    Extrapolates linearly outside the range using nearest segment slope.
    """
    if price <= amounts[0]:
        # Extrapolate using first segment slope
        if len(amounts) > 1:
            slope = (durations[1] - durations[0]) / (amounts[1] - amounts[0])
            return float(durations[0] + slope * (price - amounts[0]))
        return float(durations[0])

    if price >= amounts[-1]:
        # Extrapolate using last segment slope
        if len(amounts) > 1:
            slope = (durations[-1] - durations[-2]) / (amounts[-1] - amounts[-2])
            return float(durations[-1] + slope * (price - amounts[-1]))
        return float(durations[-1])

    # Normal interpolation
    interpolator = interp1d(amounts, durations, kind='linear')
    return float(interpolator(price))


def isotonic_predict(price: float, pref: Dict[str, Any]) -> tuple:
    """
    Monotonic regression to ensure higher price never gives shorter terms.
    Returns (predicted_months, mae_synthetic).
    """
    amounts, durations = build_breakpoints(pref)

    # Fit isotonic regression (enforces monotonicity)
    iso = IsotonicRegression(increasing=True)
    iso.fit(amounts, durations)

    # Predict
    predicted = float(iso.predict([price])[0])

    # Compute MAE on breakpoints (synthetic)
    mae_synthetic = np.mean(np.abs(iso.predict(amounts) - durations))

    return predicted, mae_synthetic


# ---------- Endpoints ------------------------------------------------
@app.post("/validate", response_model=Dict[str, Any])
def validate_only(preference: Dict[str, Any]):
    """Validate preferences without making a prediction"""
    warnings = validate_preferences(preference)

    try:
        amounts, durations = build_breakpoints(preference)

        # Identify unconfigured price ranges
        configured_bands = []
        for band, (lo, hi) in PRICE_BANDS.items():
            mn, mx = f"{band}DurationMin", f"{band}DurationMax"
            if mn in preference and mx in preference:
                configured_bands.append(band)

        unconfigured_bands = [b for b in PRICE_BANDS.keys() if b not in configured_bands]

        return {
            "valid": True,
            "warnings": warnings,
            "configured_bands": configured_bands,
            "unconfigured_bands": unconfigured_bands,
            "breakpoints": {
                "amounts": amounts.tolist(),
                "durations": durations.tolist()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})


@app.post("/advise", response_model=AdviceResponse)
def advise(req: AdviceRequest):
    price, pref = req.price, req.preference
    rounding_mode = req.rounding_mode
    include_debug = req.include_debug

    # Reject prices below minimum breakpoint price
    if price < MIN_BREAKPOINT_PRICE:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Price must be at least ${MIN_BREAKPOINT_PRICE:.2f}",
                "price": price,
                "minimum": MIN_BREAKPOINT_PRICE
            }
        )

    # Validate preferences first
    warnings = validate_preferences(pref)

    debug_info = {
        "warnings": warnings,
        "price": price,
        "band": None,
        "raw_prediction": None,
        "used_band_limits": False
    } if include_debug else None

    # 1) Check if price is inside a band
    b = band_for(price)

    if debug_info:
        debug_info["band"] = b

    # 2) If inside band with fixed rule: min = max
    if b:
        mn, mx = f"{b}DurationMin", f"{b}DurationMax"
        if mn in pref and mx in pref and pref[mn] == pref[mx]:
            months = _f(pref[mn])
            if debug_info:
                debug_info["raw_prediction"] = months
                debug_info["used_band_limits"] = True
            months = clamp(price, pref, months, rounding_mode, use_band_limits=True)
            confidence = calculate_confidence(price, "rule", band=b, pref=pref)
            return AdviceResponse(
                terms=months,
                model_type="rule",
                confidence=confidence,
                mae_synthetic=0.0,
                debug_info=debug_info
            )

    # 3) If inside band with different min/max: interpolate within band
    if b:
        mn, mx = f"{b}DurationMin", f"{b}DurationMax"
        lo, hi = PRICE_BANDS[b]
        if mn in pref and mx in pref:
            dmin, dmax = _f(pref[mn]), _f(pref[mx])
            if dmin != dmax:
                if not math.isinf(hi):
                    frac = (price - lo) / (hi - lo) if hi != lo else 0
                else:
                    # For infinite upper bound, use a decay function
                    frac = min(1.0, (price - lo) / (lo * 9)) if lo > 0 else 0

                raw_months = dmin + frac * (dmax - dmin)
                if debug_info:
                    debug_info["raw_prediction"] = raw_months
                    debug_info["used_band_limits"] = True
                months = clamp(price, pref, raw_months, rounding_mode, use_band_limits=True)
                confidence = calculate_confidence(price, "interp", band=b, pref=pref)
                return AdviceResponse(
                    terms=months,
                    model_type="interp",
                    confidence=confidence,
                    mae_synthetic=0.0,
                    debug_info=debug_info
                )

    # 4) Price is in a gap OR band has no config: use piecewise/isotonic across all breakpoints
    try:
        amounts, durations = build_breakpoints(pref)

        if debug_info:
            debug_info["breakpoints"] = {
                "amounts": amounts.tolist(),
                "durations": durations.tolist()
            }

        # Check if monotonicity is violated
        needs_isotonic = not all(durations[i] <= durations[i + 1]
                                 for i in range(len(durations) - 1))

        if needs_isotonic:
            # Use isotonic regression for monotonicity
            raw_months, mae_synthetic = isotonic_predict(price, pref)
            if debug_info:
                debug_info["raw_prediction"] = raw_months
                debug_info["used_band_limits"] = False
                debug_info["monotonicity_enforced"] = True
            months = clamp(price, pref, raw_months, rounding_mode, use_band_limits=False)
            confidence = calculate_confidence(
                price, "isotonic", amounts, durations, band=b, pref=pref
            )
            return AdviceResponse(
                terms=months,
                model_type="isotonic",
                confidence=confidence,
                mae_synthetic=mae_synthetic,
                debug_info=debug_info
            )
        else:
            # Use simple piecewise-linear
            raw_months = piecewise_predict(price, amounts, durations)
            # MAE on breakpoints
            predicted_all = np.array([piecewise_predict(a, amounts, durations)
                                      for a in amounts])
            mae_synthetic = float(np.mean(np.abs(predicted_all - durations)))

            if debug_info:
                debug_info["raw_prediction"] = raw_months
                debug_info["used_band_limits"] = False
                debug_info["monotonicity_enforced"] = False

            months = clamp(price, pref, raw_months, rounding_mode, use_band_limits=False)
            confidence = calculate_confidence(
                price, "piecewise", amounts, durations, band=b, pref=pref
            )
            return AdviceResponse(
                terms=months,
                model_type="piecewise",
                confidence=confidence,
                mae_synthetic=mae_synthetic,
                debug_info=debug_info
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "SmartPay Term Advisor"}