#!/usr/bin/env python3
# Onn Azulay – Apex | Modular Isp sweep at multiple environment points
# Modified: plot only finite segments (do not draw across NaNs) + robust card aliases + zero->NaN

import os, sys, math, csv, io
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

# Optional libraries
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
    from matplotlib import colormaps as cmaps
except Exception:
    plt = None

try:
    from rocketcea.cea_obj import CEA_Obj
except Exception as e:
    raise SystemExit(
        "rocketcea not importable. Install with:\n"
        "    python -m pip install rocketcea\n"
        f"{e}"
    )

# ------------------------------------------------------------
# --- Global config (common to all environments)
# ------------------------------------------------------------
PC_PSIA      = 1000.0           # chamber pressure
OF_MIN, OF_MAX, N_OF = 0.05, 8.0, 260
TOP_N        = 20
BASE_OUTDIR  = "isp_results"
os.makedirs(BASE_OUTDIR, exist_ok=True)

# --- Gentle spike sanitization (tunable) ---------------------
ISP_SPIKE_CAP_S          = 450.0   # treat samples above this as spurious spikes, numerical issues
MAX_SPIKE_FRACTION_KEEP  = 0.01    # if >1% samples are spikes -> discard that pair
# -------------------------------------------------------------

# ------------------------------------------------------------
# --- Environment test points
# ------------------------------------------------------------
env_points = [
    {"label": "A_ignition_0m",  "alt_m": 0,     "ambient_pa": 101325.0, "speed_m_s": 0.0,   "notes": "Sea-level ignition"},
    {"label": "B_maneuver_9km", "alt_m": 9000,  "ambient_pa": 30727.0,  "speed_m_s": 200.0, "notes": "Maneuver segment"},
    {"label": "C_intercept_8km","alt_m": 8000,  "ambient_pa": 35609.0,  "speed_m_s": 300.0, "notes": "High-speed interception"},
]

# ------------------------------------------------------------
# --- RocketCEA cards and synonyms (robust)
# ------------------------------------------------------------
OX_CARDS = {
    "LOX": "LOX",
    "GOX": "GOX",
    "N2O": "N2O",          # nitrous oxide
    "N2O4": "N2O4",
    "H2O2": "H2O2",
    "F2": "F2",
    "OF2": "OF2",
    "IRFNA": "IRFNA"
}

FUEL_CARDS = {
    "LH2":"LH2", "GH2":"GH2",
    "RP-1":"RP-1", "RP1":"RP1_NASA",
    "CH4":"CH4", "C2H6":"C2H6",
    # Do NOT hardcode C2H4 card name (varies by build); we alias to ETHYLENE and try options.
    "ETHANOL":"ETHANOL",
    "MMH":"MMH", "UDMH":"UDMH", "N2H4":"N2H4", "NH3":"NH3", "JP10":"JP10",
    "ETHYLENE":"ETHYLENE"  # if your build exposes this, alias tries will catch it
}

OX_SYNONYMS = {
    "O2":"LOX","LO2":"LOX","O2(L)":"LOX",
    "GO2":"GOX","O2(G)":"GOX",
    "MON25":"N2O4","MON-25":"N2O4",
    "N2O(G)":"N2O","N2O(L)":"N2O"
}

FUEL_SYNONYMS = {
    "H2":"LH2","H2(L)":"LH2",
    "RP1":"RP1",
    "METHANE":"CH4",
    "ETHANE":"C2H6",
    # Normalize all ethylene spellings to ETHYLENE (alias tries will handle actual card name)
    "C2H4":"ETHYLENE","ETHYLENE":"ETHYLENE",
    # Ethanol
    "C2H5OH":"ETHANOL",
    # Others
    "Hydrazine":"N2H4",
    "JP-10":"JP10"
}

CARD_TEMPS_K = {
    "LOX":90.18,"GOX":298.15,"N2O":298.15,"N2O4":298.15,"H2O2":298.15,
    "F2":82.02,"OF2":167.0,"IRFNA":298.15,
    "LH2":20.27,"GH2":298.15,"CH4":111.66,"C2H6":184.56,
    "ETHYLENE":169.0,  # approximate if present
    "ETHANOL":298.15,"MMH":298.15,"UDMH":298.15,"N2H4":298.15,"NH3":298.15,"JP10":298.15,
    "RP-1":298.15,"RP1_NASA":298.15
}

# For cards that vary by RocketCEA build, try these alternates in order:
OX_ALIASES_TRY = {
    "N2O": ["N2O", "N2O(G)", "N2O(L)"],
}
FUEL_ALIASES_TRY = {
    # Ethylene is the tricky one: different builds expose different names.
    "ETHYLENE": ["ETHYLENE", "C2H4", "C2H4_NASA"],
}

# ------------------------------------------------------------
# --- Helpers
# ------------------------------------------------------------
def has_method(obj, name):
    return hasattr(obj, name) and callable(getattr(obj, name))

def unpack_numeric(x):
    if x is None: return float("nan")
    if isinstance(x, (list, tuple, np.ndarray)): return float(x[0])
    try: return float(x)
    except Exception: return float("nan")

def canon_ox(name: str) -> str:
    return OX_SYNONYMS.get(name, name)

def canon_fuel(name: str) -> str:
    return FUEL_SYNONYMS.get(name, name)

def pick_cards(ox, fuel):
    return OX_CARDS.get(ox, ox), FUEL_CARDS.get(fuel, fuel)

def _try_build_card_names(ox_name, fuel_name):
    """
    Attempt to build CEA_Obj with several card aliases while suppressing
    RocketCEA 'bad fuel/oxidizer name' prints. Returns (obj, None) on success,
    otherwise (None, last_error_message).
    """
    ox_list = OX_ALIASES_TRY.get(ox_name, [ox_name])
    fu_list = FUEL_ALIASES_TRY.get(fuel_name, [fuel_name])

    last_err = None
    for oc in ox_list:
        for fc in fu_list:
            buf_out, buf_err = io.StringIO(), io.StringIO()
            try:
                with redirect_stdout(buf_out), redirect_stderr(buf_err):
                    obj = CEA_Obj(oxName=oc, fuelName=fc)
                return obj, None
            except Exception as ex:
                last_err = str(ex) or buf_err.getvalue() or buf_out.getvalue()
    return None, last_err

def safe_make(ox_card, fuel_card):
    """
    Robust instantiation with:
      - normalization via OX_CARDS/FUEL_CARDS then synonyms,
      - alias-tries for tricky cards (e.g., ETHYLENE),
      - muted RocketCEA error prints on failed tries.
    """
    ox_norm = OX_CARDS.get(ox_card, OX_SYNONYMS.get(ox_card, ox_card))
    fu_norm = FUEL_CARDS.get(fuel_card, FUEL_SYNONYMS.get(fuel_card, fuel_card))
    obj, err = _try_build_card_names(ox_norm, fu_norm)
    if obj is not None:
        return obj, None
    return None, err

def find_eps_for_pcovpe(cea, Pc, MR, target_pcovpe):
    if has_method(cea, "get_eps_at_PcOvPe"):
        try: return float(cea.get_eps_at_PcOvPe(Pc=Pc, MR=MR, PcOvPe=target_pcovpe))
        except Exception: pass
    def pcovpe_from_eps(eps):
        if has_method(cea,"get_PcOvPe"):
            return float(cea.get_PcOvPe(Pc=Pc, MR=MR, eps=eps))
        if has_method(cea,"get_Pexit"):
            Pe=float(cea.get_Pexit(Pc=Pc, MR=MR, eps=eps))
            return Pc/Pe if Pe>0 else float("inf")
        raise RuntimeError("No PcOvPe/Pexit method.")
    lo, hi = 1.1, 400.0
    for _ in range(80):
        mid = 0.5*(lo+hi)
        val = pcovpe_from_eps(mid)
        if abs(val - target_pcovpe) / max(target_pcovpe, 1e-9) < 5e-4:
            return mid
        if val < target_pcovpe: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

def get_isp_equil(cea, Pc, MR, eps, pamb_psia):
    for kw in ({"Pc":Pc,"MR":MR,"eps":eps,"Pamb":pamb_psia},
               {"Pc":Pc,"MR":MR,"eps":eps,"Pa":pamb_psia},
               {"Pc":Pc,"MR":MR,"eps":eps}):
        try:
            val = unpack_numeric(cea.get_Isp(**kw))
            if val <= 0:  # zero or negative -> treat as invalid/no-solution
                return float("nan")
            return val
        except TypeError:
            continue
        except Exception:
            return float("nan")
    return float("nan")

# --- gentle per-curve sanitizer ----------------------------------------
def sanitize_curve(of_vec, isp_arr,
                   isp_cap=ISP_SPIKE_CAP_S,
                   max_outlier_fraction=MAX_SPIKE_FRACTION_KEEP):
    """
    1) Turn non-physical values (<=0) into NaN (erase those O/F regions).
    2) Replace spurious spikes (> isp_cap) with NaN unless they dominate the curve.
    """
    y = np.array(isp_arr, dtype=float)

    # (1) Non-physical or zero -> NaN (your CSV zeros become gaps on the plot)
    y[~np.isfinite(y)] = np.nan
    y[y <= 0.0] = np.nan

    # (2) Spike handling
    outliers = y > isp_cap
    if not np.any(outliers):
        return y
    frac = np.nanmean(outliers.astype(float))
    if frac > max_outlier_fraction:
        return None  # dominated by spikes -> discard pair entirely
    y[outliers] = np.nan
    return y

# ------------------------------------------------------------
# --- Candidate propellant families
# ------------------------------------------------------------
def candidate_pairs():
    ox_fams = ["LOX","GOX","N2O","N2O4","H2O2","F2","OF2","IRFNA"]
    fu_fams = ["LH2","GH2","RP-1","CH4","C2H6","ETHYLENE","ETHANOL",
               "MMH","UDMH","N2H4","NH3","JP10"]
    pairs = []
    for oc in ox_fams:
        for fc in fu_fams:
            ocard, fcard = pick_cards(oc, fc)
            pairs.append((oc, fc, ocard, fcard))
    return pairs

# ------------------------------------------------------------
# --- “Book-listed” pairs to highlight in legend
# ------------------------------------------------------------
BOOK_LISTED = {
    ("LOX","LH2"), ("LOX","CH4"), ("LOX","C2H6"), ("LOX","ETHYLENE"),
    ("LOX","RP-1"), ("LOX","N2H4"), ("LOX","NH3"), ("LOX","ETHANOL"),
    ("GOX","GH2"),
    ("F2","LH2"), ("F2","CH4"), ("F2","C2H6"), ("F2","ETHYLENE"),
    ("F2","RP-1"), ("F2","MMH"), ("F2","UDMH"), ("F2","N2H4"), ("F2","NH3"), ("F2","ETHANOL"),
    ("OF2","LH2"), ("OF2","CH4"), ("OF2","C2H6"), ("OF2","ETHYLENE"),
    ("OF2","RP-1"), ("OF2","MMH"), ("OF2","N2H4"), ("OF2","ETHANOL"),
    ("N2O4","MMH"), ("N2O4","UDMH"), ("N2O4","N2H4"),
    ("H2O2","RP-1"), ("H2O2","MMH"), ("H2O2","N2H4"), ("H2O2","ETHANOL"),
    ("IRFNA","MMH"), ("IRFNA","UDMH"),
    ("N2O","ETHANOL"), ("N2O","RP-1"), ("N2O","CH4")
}

def is_book_listed(ox_name: str, fuel_name: str) -> bool:
    oc = canon_ox(ox_name)
    fc = canon_fuel(fuel_name)
    return (oc, fc) in BOOK_LISTED

# ------------------------------------------------------------
# --- Core sweep per environment
# ------------------------------------------------------------
def sweep_all_pairs(pairs, Pc, Pe, of_min, of_max, n_of):
    of_vec = np.linspace(of_min, of_max, n_of)
    results, failed = [], []
    for oc, fc, ox_card, fuel_card in pairs:
        label = f"{oc}/{fc}"
        cea, err = safe_make(ox_card, fuel_card)
        if cea is None:
            failed.append((label, f"instantiation failed: {err}"))
            continue

        isp_vals, eps_fail = [], 0
        for of in of_vec:
            try:
                eps = find_eps_for_pcovpe(cea, Pc, of, Pc/Pe)
                isp = get_isp_equil(cea, Pc, of, eps, Pe)
            except Exception:
                isp = float("nan"); eps_fail += 1
            isp_vals.append(isp)

        arr = np.array(isp_vals, dtype=float)
        cleaned = sanitize_curve(of_vec, arr)
        if cleaned is None:
            failed.append((label, "discarded: excessive spikes > cap"))
            continue
        arr = cleaned

        if np.all(np.isnan(arr)):
            failed.append((label, f"All NaN (eps_fail={eps_fail})"))
            continue

        peak = float(np.nanmax(arr))
        results.append({
            "pair": label,
            "ox": oc, "fuel": fc,
            "ox_card": ox_card, "fuel_card": fuel_card,
            "of": of_vec.copy(), "isp": arr, "peak": peak,
            "book_listed": is_book_listed(oc, fc)
        })
        print(f"OK: {label:>15s}  peak(eq) ≈ {peak:.2f} s  {'[book]' if is_book_listed(oc, fc) else ''}")
    return results, failed

# ------------------------------------------------------------
# --- Plot helpers: draw only finite segments (do not connect NaNs)
# ------------------------------------------------------------
def iter_finite_segments(x, y):
    """
    Yield (x_seg, y_seg) pairs for each contiguous finite segment in y.
    x and y are 1D arrays of same length.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    finite = np.isfinite(y)
    if not np.any(finite):
        return
    idx = np.where(np.diff(finite.astype(int)) != 0)[0] + 1
    bounds = np.concatenate(([0], idx, [len(finite)]))
    for i in range(len(bounds)-1):
        s, e = bounds[i], bounds[i+1]
        if finite[s]:
            yield x[s:e], y[s:e]

def plot_top_segments(results, top_n, path, env_label):
    if plt is None or not results:
        print("Plot skipped (matplotlib unavailable or no results).")
        return
    results_sorted = sorted(results, key=lambda r: r["peak"], reverse=True)
    top = results_sorted[:top_n]

    plt.figure(figsize=(12,8))
    cmap = cmaps.get_cmap("tab20")
    legend_handles = []
    legend_labels = []

    for k, r in enumerate(top):
        color = cmap(k % 20)
        lw = 1.2
        if r.get("book_listed", False):
            lw = 2.4  # thicker for book-listed
        first_handle = None
        for (x_seg, y_seg) in iter_finite_segments(r["of"], r["isp"]):
            if x_seg.size < 2:
                continue
            line, = plt.plot(x_seg, y_seg, color=color, linewidth=lw, alpha=1.0)
            if first_handle is None:
                first_handle = line
        if first_handle is not None:
            legend_handles.append(first_handle)
            lbl = f"{r['pair']} [{r['ox_card']}/{r['fuel_card']}]"
            legend_labels.append(lbl)

    leg = plt.legend(legend_handles, legend_labels, fontsize="small",
                     bbox_to_anchor=(1.02,1), loc="upper left")
    if leg is not None:
        label_to_book = {f"{r['pair']} [{r['ox_card']}/{r['fuel_card']}]": r.get("book_listed", False) for r in top}
        for txt in leg.get_texts():
            txt_text = txt.get_text()
            if label_to_book.get(txt_text, False):
                txt.set_color("goldenrod")
                txt.set_fontweight("bold")

    plt.xlabel("O/F (mass ratio)")
    plt.ylabel("Isp (s)")
    plt.title(f"{env_label}: Isp vs O/F @ Pc={PC_PSIA:.0f} psia (equilibrium)")
    plt.grid(True, ls="--", lw=0.4)
    plt.tight_layout(rect=(0,0,0.78,1))
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Plot saved: {path}")

# ------------------------------------------------------------
# --- Output helpers (CSV/Excel)
# ------------------------------------------------------------
def write_full_csv(results, out_csv):
    if not results: return
    of_vec = results[0]["of"]
    with open(out_csv,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["O/F"]+[r["pair"] for r in results])
        for i,of in enumerate(of_vec):
            w.writerow([of]+[("" if math.isnan(r["isp"][i]) else r["isp"][i]) for r in results])
    print(f"CSV written: {out_csv}")

def write_excel(results, top_n, path):
    if pd is None or not results: return
    of_vec = results[0]["of"]
    df_all = pd.DataFrame({"O/F": of_vec})
    for r in results:
        df_all[r["pair"]] = r["isp"]
    df_meta = pd.DataFrame([{
        "pair": r["pair"], "book_listed": r["book_listed"], "peak_Isp_s": r["peak"],
        "ox_card": r["ox_card"], "fuel_card": r["fuel_card"],
        "T_ox_K": CARD_TEMPS_K.get(r["ox_card"], ""), "T_fuel_K": CARD_TEMPS_K.get(r["fuel_card"], "")
    } for r in results])
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df_all.to_excel(w, "All_pairs", index=False)
        df_meta.to_excel(w, "Metadata", index=False)
    print(f"Excel written: {path}")

# ------------------------------------------------------------
# --- One environment runner
# ------------------------------------------------------------
def run_environment_case(env):
    label = env["label"]
    alt = env["alt_m"]
    Pe_pa = env["ambient_pa"]
    Pe_psia = Pe_pa / 6894.757
    env_dir = os.path.join(BASE_OUTDIR, label)
    os.makedirs(env_dir, exist_ok=True)
    print(f"\n=== Running {label} (alt={alt} m, Pe={Pe_psia:.3f} psia) ===")

    pairs = candidate_pairs()
    results, failed = sweep_all_pairs(pairs, PC_PSIA, Pe_psia, OF_MIN, OF_MAX, N_OF)

    if failed:
        print("Failed pairs:", failed)
    if not results:
        print("No results for", label)
        return

    csv_path = os.path.join(env_dir, "isp_allpairs.csv")
    xlsx_path = os.path.join(env_dir, "isp_results.xlsx")
    png_path = os.path.join(env_dir, "isp_top.png")

    write_full_csv(results, csv_path)
    write_excel(results, TOP_N, xlsx_path)
    plot_top_segments(results, TOP_N, png_path, label)

# ------------------------------------------------------------
# --- Main driver
# ------------------------------------------------------------
def main():
    for env in env_points:
        run_environment_case(env)
    print("\nAll environment points completed.")

if __name__ == "__main__":
    main()