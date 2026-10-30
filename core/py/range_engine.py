"""Calorie Lens range engine (reference implementation). See spec/range_engine.md.

Stdlib only. Triples are dicts {"min","typ","max"}. All results are full precision;
rounding is display-only (format_range).
"""
import math

Z90 = 1.645          # one-sided z for the 5th/95th percentile
SPAN90 = 3.29        # 2 * Z90 (rounded as in PLAN.md section 5)
FLOOR_LO = 0.85
FLOOR_HI = 1.15
LOW_CONF_SIGMA = 0.35
SUGAR_VARIANCE = 0.25
KCAL_PER_G_SUGAR = 4.0
KCAL_PER_G_OIL = 9.0


def _num(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError("%s must be a finite number" % what)
    return float(x)


def _triple(t, what, positive=True):
    if not isinstance(t, dict):
        raise ValueError("%s missing" % what)
    mn, ty, mx = (_num(t.get(k), "%s.%s" % (what, k)) for k in ("min", "typ", "max"))
    if not (mn <= ty <= mx):
        raise ValueError("%s must satisfy min<=typ<=max" % what)
    if positive and mn <= 0:
        raise ValueError("%s.min must be > 0" % what)
    if not positive and mn < 0:
        raise ValueError("%s.min must be >= 0" % what)
    return mn, ty, mx


def sigma_of(mn, mx):
    """Log-space sigma from a 90% interval: ln(max/min)/3.29 (0 if min==max)."""
    if mn <= 0:
        raise ValueError("min must be > 0")
    if mn == mx:
        return 0.0
    return math.log(mx / mn) / SPAN90


def _floor(lo, med, hi):
    return min(lo, FLOOR_LO * med), max(hi, FLOOR_HI * med)


def estimate_solid(dish, volume_ml):
    """dish: DishEntry dict. volume_ml: Triple or None. Returns {id,lo,med,hi,flags}."""
    flags = []
    e_min, e_typ, e_max = _triple(dish.get("kcal_100g"), "kcal_100g")
    s_e = sigma_of(e_min, e_max)
    if volume_ml is None:
        p_min, p_typ, p_max = _triple(dish.get("default_portion_g"), "default_portion_g")
        mass_med = p_typ
        flags.append("portion_default")
        sig2 = sigma_of(p_min, p_max) ** 2 + s_e ** 2
    else:
        v_min, v_typ, v_max = _triple(volume_ml, "volumeMl")
        d_min, d_typ, d_max = _triple(dish.get("density_g_ml"), "density_g_ml")
        mass_med = v_typ * d_typ
        sig2 = sigma_of(v_min, v_max) ** 2 + sigma_of(d_min, d_max) ** 2 + s_e ** 2
    med = mass_med * e_typ / 100.0
    sigma = math.sqrt(sig2)
    lo = med * math.exp(-Z90 * sigma)
    hi = med * math.exp(Z90 * sigma)
    oil = dish.get("oil_g_per_100g")
    if oil is not None:
        o_min = _num(oil.get("min"), "oil.min")
        o_max = _num(oil.get("max"), "oil.max")
        if o_min < 0 or o_max < o_min:
            raise ValueError("oil must satisfy 0<=min<=max")
        lo += o_min * mass_med * KCAL_PER_G_OIL / 100.0
        hi += o_max * mass_med * KCAL_PER_G_OIL / 100.0
        med += (o_min + o_max) / 2.0 * mass_med * KCAL_PER_G_OIL / 100.0
    lo, hi = _floor(lo, med, hi)
    if sigma > LOW_CONF_SIGMA:
        flags.append("low_confidence")
    return {"id": dish.get("id"), "lo": lo, "med": med, "hi": hi, "flags": flags}


def estimate_drink(drink, size_ml, sugar_level, full_sugar_g_per_700, sugar_fraction):
    """Drink kcal = sugar*4 + base*(size/700) + toppings, combined by interval."""
    size = _num(size_ml, "sizeMl")
    if size <= 0:
        raise ValueError("sizeMl must be > 0")
    full = _num(full_sugar_g_per_700, "fullSugarGPer700")
    if full < 0:
        raise ValueError("fullSugarGPer700 must be >= 0")
    if not isinstance(sugar_fraction, dict) or sugar_level not in sugar_fraction:
        raise ValueError("unknown sugar level")
    frac = _num(sugar_fraction[sugar_level], "sugar fraction")
    if frac < 0:
        raise ValueError("sugar fraction must be >= 0")
    b = _triple(drink.get("base_kcal_per_700ml"), "base_kcal_per_700ml", positive=False)
    t = _triple(drink.get("toppings_kcal"), "toppings_kcal", positive=False)
    sugar = full * frac * size / 700.0
    s_kcal = sugar * KCAL_PER_G_SUGAR
    k = size / 700.0
    lo = s_kcal * (1 - SUGAR_VARIANCE) + b[0] * k + t[0]
    med = s_kcal + b[1] * k + t[1]
    hi = s_kcal * (1 + SUGAR_VARIANCE) + b[2] * k + t[2]
    lo, hi = _floor(lo, med, hi)
    return {"id": drink.get("id"), "lo": lo, "med": med, "hi": hi,
            "flags": ["drink_sugar_assumed"]}


def combine_meal(items):
    """Sum medians; combine lower/upper half-widths in quadrature."""
    med = sum(i["med"] for i in items)
    dl = math.sqrt(sum((i["med"] - i["lo"]) ** 2 for i in items))
    du = math.sqrt(sum((i["hi"] - i["med"]) ** 2 for i in items))
    return {"lo": med - dl, "med": med, "hi": med + du}


def round5(x):
    """Round to nearest 5, halves up."""
    return math.floor(x / 5.0 + 0.5) * 5


def format_range(lo, hi):
    """Display string like '450-600' (en dash), values rounded to 5 kcal."""
    return "%d–%d" % (round5(lo), round5(hi))
