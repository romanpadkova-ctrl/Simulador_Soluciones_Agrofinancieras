import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Simulador de Soluciones Agrofinancieras", layout="wide")

# ========= Estilos =========
st.markdown("""
<style>
div[data-testid="stSidebar"] > div:first-child { background: linear-gradient(180deg,#0f1a12,#0b0f0c); }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ========= Helpers =========
EPS = 1e-6  # tolerancia numérica

def r05(x: float) -> float:
    # redondeo a múltiplos de 0.5 y 1 decimal (para primas y strikes)
    return round(round(float(x) * 2) / 2, 1)

def r1(x: float) -> float:
    return round(float(x), 1)

def df_round1(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float).round(1)
    return df

def fwd_from_matba(matba: float) -> float:
    # Regla: forward = MATBA - 3
    return r05(matba - 3.0)

def log(msg: str):
    st.session_state.logs.append(msg)

# ========= Curvas =========
def demo_rounds(start_month=9, rounds=9):
    """
    Curva con movimientos más amplios para testear performance:
    arranca 290; sube fuerte a 315; amaga; cae a 280 al final.
    """
    pattern = [290, 300, 315, 312, 295, 290, 305, 292, 280]
    rows = []
    for i in range(rounds):
        rows.append({
            "round_id": i+1,
            "date": f"2025-{((start_month-1+i)%12)+1:02d}-01",
            "matba_price": r05(pattern[i])
        })
    return pd.DataFrame(rows)

def rounds_from_file(file, start_month=9, rounds=9):
    name = getattr(file, "name", "").lower()
    df = pd.read_excel(file) if name.endswith((".xlsx",".xls")) else pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}
    if "date" not in cols or "matba_price" not in cols:
        raise ValueError("El archivo debe tener columnas 'date' y 'matba_price'.")
    df = df.rename(columns={cols["date"]:"date", cols["matba_price"]:"matba_price"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date","matba_price"])
    df["month"] = df["date"].dt.month
    monthly = df.groupby("month")["matba_price"].mean().to_dict()
    rows=[]
    for i in range(rounds):
        m = ((start_month-1+i)%12)+1
        val = float(monthly.get(m, df["matba_price"].mean()))
        rows.append({"round_id":i+1,"date":f"2025-{m:02d}-01","matba_price":r05(val)})
    return pd.DataFrame(rows)

# ========= Pricing (time-decay visible ronda a ronda) =========
VOL_MULT = {"baja":0.8,"media":1.0,"alta":1.3}

def tenor_months(curr_idx, last_idx):
    # meses restantes (≥1)
    return max(1, last_idx - curr_idx + 1)

def premium_option_base(matba_price: float, months_left: int, vol_level="media") -> float:
    """
    Prima base con sensibilidad a tiempo y subyacente.
    Elegimos exponente 0.7 y coef 0.020 para que el cambio por ronda sea ~0.5–1.5.
    """
    mult = VOL_MULT.get(vol_level, 1.0)
    time_factor = (months_left / 12.0) ** 0.7
    price_factor = (matba_price / 300.0) ** 0.9
    base = 0.020 * matba_price * time_factor * price_factor * mult
    return r05(base)

def piso_variants(matba, prem_base):
    # ATM caro; -5 y -10 más baratos
    atm = r05(max(10.0, prem_base * 1.30))
    m5  = r05(max( 6.0, atm * 0.60))
    m10 = r05(max( 3.0, atm * 0.35))
    return [
        {"label": f"Strike {r05(matba)} / Prima {atm}",   "strike": r05(matba),   "prima": atm},
        {"label": f"Strike {r05(matba-5)} / Prima {m5}",  "strike": r05(matba-5), "prima": m5},
        {"label": f"Strike {r05(matba-10)} / Prima {m10}", "strike": r05(matba-10), "prima": m10},
    ]

def ultra_piso_variants_from_piso(piso_opts):
    """
    Ultra Piso = mismos strikes que Piso, pero ~25% más barato.
    """
    opts=[]
    for o in piso_opts:
        strike = r05(o["strike"])
        prima  = r05(max(1.0, o["prima"] * 0.75))  # 25% más barato
        opts.append({"label": f"Strike {strike} / Prima {prima}", "strike": strike, "prima": prima})
    return opts

def ultra_banda_variants_from_ultrapiso(up_opts):
    """
    ULTRABANDA = mismos strikes que Ultra Piso, techo = strike + 15.
    Prima ~30% más barata que Ultra Piso.
    """
    opts=[]
    for o in up_opts:
        piso = r05(o["strike"])
        techo = r05(piso + 15)
        prima = r05(max(1.0, o["prima"] * 0.70))  # 30% más barato que UP
        opts.append({"piso": piso, "techo": techo, "prima": prima,
                     "label": f"Piso {piso} / Techo {techo} / Prima {prima}"})
    return opts

def cargill_plus_variants(matba):
    bonus=r05(10.0)
    return [
        {"techo":r05(matba),   "bonificacion":bonus,          "label":f"Techo {r05(matba)} / Bonificación {bonus}"},
        {"techo":r05(matba+5), "bonificacion":r05(bonus*0.8), "label":f"Techo {r05(matba+5)} / Bonificación {r05(bonus*0.8)}"},
        {"techo":r05(matba+10),"bonificacion":r05(bonus*0.65),"label":f"Techo {r05(matba+10)} / Bonificación {r05(bonus*0.65)}"},
    ]

def duplo_variants(matba):
    """
    Regla:
      triggers = MATBA - {3,5,10}
      accumulation levels = MATBA + {10,7,5}
    """
    t1=r05(matba-3);  a1=r05(matba+10)
    t2=r05(matba-5);  a2=r05(matba+7)
    t3=r05(matba-10); a3=r05(matba+5)
    opts=[{"trigger":t1,"accum":a1},{"trigger":t2,"accum":a2},{"trigger":t3,"accum":a3}]
    for o in opts: o["label"]=f"Disparador {o['trigger']} / Acumulación {o['accum']}"
    return opts

def build_catalog(mkt_row, curr_idx, last_i):
    matba=float(mkt_row["matba_price"])
    forward=fwd_from_matba(matba)
    # Primero piso -> luego derivamos UP y UB con reglas de descuento y techo
    prem = premium_option_base(matba, tenor_months(curr_idx,last_i), "media")
    piso_opts = piso_variants(matba, prem)
    up_opts   = ultra_piso_variants_from_piso(piso_opts)
    ub_opts   = ultra_banda_variants_from_ultrapiso(up_opts)
    return pd.DataFrame([
        {"tool_id":"forward","nombre":"Venta Forward","variants":[{"label": f"Forward soja mayo: {forward}", "price": forward}]},
        {"tool_id":"piso","nombre":"Piso Asegurado","variants":piso_opts},
        {"tool_id":"ultra_piso","nombre":"Ultra Piso (Piso Promediando)","variants":up_opts},
        {"tool_id":"ultra_banda","nombre":"ULTRABANDA (Piso + Techo Promediando)","variants":ub_opts},
        {"tool_id":"cargill_plus","nombre":"Cargill Plus (Potenciador)","variants":cargill_plus_variants(matba)},
        {"tool_id":"duplo","nombre":"Duplos (Acumuladores)","variants":duplo_variants(matba)},
    ])

# ========= Estado =========
def init_state():
    if "data_source" not in st.session_state: st.session_state.data_source="Simulada v0.7"
    if "uploaded_snapshot" not in st.session_state: st.session_state.uploaded_snapshot=None
    if "rounds_df" not in st.session_state: st.session_state.rounds_df=demo_rounds()
    if "round_idx" not in st.session_state: st.session_state.round_idx=0
    if "total_volume" not in st.session_state: st.session_state.total_volume=1000
    if "forwards" not in st.session_state: st.session_state.forwards=[]
    if "pisos" not in st.session_state: st.session_state.pisos=[]
    if "ultra_pisos" not in st.session_state: st.session_state.ultra_pisos=[]
    if "bandas" not in st.session_state: st.session_state.bandas=[]
    if "cplus" not in st.session_state: st.session_state.cplus=[]
    if "duplos" not in st.session_state: st.session_state.duplos=[]
    if "sel_tool" not in st.session_state: st.session_state.sel_tool=None
    if "sel_variant" not in st.session_state: st.session_state.sel_variant=None
    if "already_finalized" not in st.session_state: st.session_state.already_finalized=False
    if "logs" not in st.session_state: st.session_state.logs=[]

def reset_position():
    st.session_state.round_idx=0
    st.session_state.forwards=[]
    st.session_state.pisos=[]; st.session_state.ultra_pisos=[]
    st.session_state.bandas=[]; st.session_state.cplus=[]; st.session_state.duplos=[]
    st.session_state.sel_tool=None; st.session_state.sel_variant=None
    st.session_state.already_finalized=False
    st.session_state.logs=[]

def apply_data_source(source, uploaded_file):
    if source=="Simulada v0.7":
        st.session_state.rounds_df=demo_rounds(); st.session_state.uploaded_snapshot=None
        reset_position(); return True,"Usando curva simulada."
    if uploaded_file is None: return False,"Elegiste 'Archivo' pero no subiste ningún CSV/Excel."
    try:
        st.session_state.rounds_df=rounds_from_file(uploaded_file)
        st.session_state.uploaded_snapshot=uploaded_file.name
        reset_position(); return True,f"Usando archivo: {uploaded_file.name}"
    except Exception as e:
        return False,f"No pude leer el archivo: {e}"

def cur_market(): return st.session_state.rounds_df.iloc[st.session_state.round_idx].to_dict()
def last_idx(): return len(st.session_state.rounds_df)-1

def add_forward(qty, price, round_id, tag):
    st.session_state.forwards.append({"qty": r1(qty), "price": r1(price), "round": int(round_id), "tool": tag})

# ========= Compromiso / capacidad =========
def committed_breakdown():
    sold_qty = sum(f["qty"] for f in st.session_state.forwards)

    # Pisos
    p_early = sum(p.get("early_fixed_qty", 0.0) for p in st.session_state.pisos)
    p_total = sum(p["qty_total"] for p in st.session_state.pisos)
    p_commit = max(0.0, p_total - p_early)

    # Ultra pisos
    up_early = sum(up.get("early_fixed_qty", 0.0) for up in st.session_state.ultra_pisos)
    up_total = sum(up["qty_total"] for up in st.session_state.ultra_pisos)
    up_commit = max(0.0, up_total - up_early)

    # ULTRABANDA
    ub_commit = sum(b["qty_total"] for b in st.session_state.bandas)

    # Duplos
    dp_commit = sum(d["qty_total"] for d in st.session_state.duplos)

    # Cargill Plus: el leg del techo
    cp_commit = sum(c["qty"] for c in st.session_state.cplus)

    committed_wo = p_commit + up_commit + ub_commit + dp_commit + cp_commit
    committed_total = r1(sold_qty + committed_wo)
    return r1(sold_qty), r1(committed_wo), r1(committed_total)

def capacity_left():
    _, _, tot = committed_breakdown()
    return max(0.0, float(st.session_state.total_volume) - float(tot))

# ========= Alta de decisiones =========
def select_tool_variant(tool_id,variant):
    st.session_state.sel_tool=tool_id; st.session_state.sel_variant=variant

def add_decision(qty:int):
    if not st.session_state.sel_tool or not st.session_state.sel_variant:
        return
    left = capacity_left()
    if qty > left + EPS:
        msg=f"No se puede suscribir: capacidad libre {r1(left)} tn, intentaste {r1(qty)} tn."
        log(f"⚠️ {msg}")
        return

    mkt=cur_market(); rid=int(mkt["round_id"]); matba=float(mkt["matba_price"]); forward=fwd_from_matba(matba)
    total_rounds=len(st.session_state.rounds_df); rounds_left=total_rounds-(rid-1)
    t=st.session_state.sel_tool; v=st.session_state.sel_variant

    if t=="forward":
        add_forward(qty, forward, rid, "Forward")
        log(f"✅ Venta Forward {r1(qty)} tn a {r1(forward)} (ronda {rid}).")

    elif t=="piso":
        st.session_state.pisos.append({
            "id":f"piso_{rid}_{len(st.session_state.pisos)+1}",
            "qty_total":r1(qty),"qty_rem":r1(qty),"strike":r05(v["strike"]),"prima":r05(v["prima"]),
            "round_open":rid,"fixed_qty":0.0,"fixed_avg":0.0,"early_fixed_qty":0.0
        })
        log(f"🛡️ Piso Asegurado suscrito: {r1(qty)} tn | strike {r1(v['strike'])} | prima {r1(v['prima'])} (ronda {rid}).")

    elif t=="ultra_piso":
        dq=float(qty)/rounds_left
        st.session_state.ultra_pisos.append({
            "id":f"up_{rid}_{len(st.session_state.ultra_pisos)+1}",
            "qty_total":float(qty),"qty_rem":float(qty),"strike":r05(v["strike"]),"prima":r05(v["prima"]),
            "round_open":rid,"daily_qty":dq,"fixed_qty":0.0,"fixed_avg":0.0,
            "acum_last_round": None,"early_fixed_qty": 0.0
        })
        log(f"🧮 Ultra Piso suscrito: {r1(qty)} tn | strike {r1(v['strike'])} | prima {r1(v['prima'])} (ronda {rid}).")

    elif t=="ultra_banda":
        dq=float(qty)/rounds_left
        st.session_state.bandas.append({
            "id":f"ub_{rid}_{len(st.session_state.bandas)+1}",
            "qty_total":float(qty),"qty_rem":float(qty),
            "piso":r05(v["piso"]),"techo":r05(v["techo"]),"prima":r05(v["prima"]),
            "round_open":rid,"daily_qty":dq,"fixed_qty":0.0,"fixed_avg":0.0,
            "acum_last_round": None
        })
        log(f"📦 ULTRABANDA suscrita: {r1(qty)} tn | piso {r1(v['piso'])} / techo {r1(v['techo'])} | prima {r1(v['prima'])} (ronda {rid}).")

    elif t=="cargill_plus":
        add_forward(qty, forward+float(v["bonificacion"]), rid,
                    f"Cargill Plus — Forward ({r1(forward)} + {r1(v['bonificacion'])})")
        st.session_state.cplus.append({"qty":r1(qty),"techo":r05(v["techo"]),
                                       "bonificacion":r05(v["bonificacion"]),"round_open":rid,"closed":False})
        log(f"➕ Cargill Plus suscrito: {r1(qty)} tn | techo {r1(v['techo'])} | bonif {r1(v['bonificacion'])} (ronda {rid}). Generó forward 1:1 a {r1(forward+v['bonificacion'])}.")

    elif t=="duplo":
        dq=r1(float(qty)/rounds_left)
        st.session_state.duplos.append({
            "id":f"dp_{rid}_{len(st.session_state.duplos)+1}",
            "qty_total":r1(qty),"trigger":r05(v["trigger"]),"accum":r05(v["accum"]),
            "round_open":rid,"daily_qty":dq,"fixed_qty":0.0,"fixed_avg":0.0,
            "released_qty":0.0,"processed":0.0,"acum_last_round":None
        })
        log(f"🔁 Duplo suscrito: {r1(qty)} tn | disparador {r1(v['trigger'])} | acumulación {r1(v['accum'])} (ronda {rid}).")

# ========= Lógica por turno =========
def weighted_avg(old_qty,old_avg,add_qty,add_price):
    add_qty=float(add_qty); add_price=float(add_price)
    if add_qty<=0: return r1(old_qty), r1(old_avg)
    new_qty=old_qty+add_qty
    new_avg=(old_qty*old_avg + add_qty*add_price)/new_qty if new_qty>0 else 0.0
    return r1(new_qty), r1(new_avg)

def process_round_effects():
    df=st.session_state.rounds_df; idx=st.session_state.round_idx
    matba=float(df.iloc[idx]["matba_price"]); rid=int(df.iloc[idx]["round_id"])

    # ULTRABANDA (acumula)
    for band in st.session_state.bandas:
        if rid<band["round_open"] or band["qty_rem"]<=EPS: continue
        qty=min(band["daily_qty"], band["qty_rem"])
        if qty<=EPS: continue
        if matba < band["piso"]: px=band["piso"]-band["prima"]
        elif matba > band["techo"]: px=band["techo"]-band["prima"]
        else: px=matba-band["prima"]
        band["fixed_qty"],band["fixed_avg"]=weighted_avg(band["fixed_qty"],band["fixed_avg"],qty,px)
        band["qty_rem"]=max(0.0,band["qty_rem"]-qty)
        band["acum_last_round"]=rid

    # Ultra Piso (acumula)
    for up in st.session_state.ultra_pisos:
        if rid<up["round_open"] or up["qty_rem"]<=EPS: continue
        qty=min(up["daily_qty"], up["qty_rem"])
        if qty<=EPS: continue
        px=max(matba, up["strike"]) - up["prima"]
        up["fixed_qty"],up["fixed_avg"]=weighted_avg(up["fixed_qty"],up["fixed_avg"],qty,px)
        up["qty_rem"]=max(0.0, up["qty_rem"]-qty)
        up["acum_last_round"]=rid

    # Duplos
    for d in st.session_state.duplos:
        if rid<d["round_open"] or d["processed"]>=d["qty_total"]-EPS: continue
        remaining=max(0.0,d["qty_total"]-d["processed"]); qty=min(d["daily_qty"], remaining)
        if qty<=EPS: continue
        if matba>=d["accum"]:
            px=d["accum"]; d["fixed_qty"],d["fixed_avg"]=weighted_avg(d["fixed_qty"],d["fixed_avg"],qty,px)
        elif d["trigger"]<=matba<d["accum"]:
            half=qty/2.0; px=d["accum"]
            d["fixed_qty"],d["fixed_avg"]=weighted_avg(d["fixed_qty"],d["fixed_avg"],half,px)
            d["released_qty"]=r1(d["released_qty"]+half)
        else:
            d["released_qty"]=r1(d["released_qty"]+qty)
        d["processed"]=r1(d["processed"]+qty)
        d["acum_last_round"]=rid

# ========= Early fix =========
def early_fix_ultra_piso_all_pending(up_id):
    df=st.session_state.rounds_df; idx=st.session_state.round_idx
    matba=float(df.iloc[idx]["matba_price"]); rid=int(df.iloc[idx]["round_id"])
    for up in st.session_state.ultra_pisos:
        if up["id"]!=up_id:
            continue
        pending = max(0.0, up["qty_rem"])
        if pending <= EPS:
            log("ℹ️ Ultra Piso: no hay pendiente para fijar.")
            return
        px = max(matba, up["strike"]) - up["prima"]
        add_forward(pending, px, rid, f"Fijación Ultra Piso (piso {r1(up['strike'])})")
        up["early_fixed_qty"] = up.get("early_fixed_qty", 0.0) + pending
        up["qty_rem"] = 0.0
        log(f"✅ Ultra Piso fijado completamente (pendiente {r1(pending)} tn) a {r1(px)}.")
        return

# ========= Cierre (robusto) =========
def finalize_results():
    if st.session_state.already_finalized:
        return None

    rounds_df = st.session_state.rounds_df
    last_row   = rounds_df.iloc[-1]
    last_matba = float(last_row["matba_price"])
    last_fwd   = fwd_from_matba(last_matba)
    rid        = int(last_row["round_id"])

    # --- Pisos asegurados ---
    for piso in list(st.session_state.pisos):
        if not isinstance(piso, dict):
            continue
        # pendiente
        if float(piso.get("qty_rem", 0.0)) > EPS:
            px = max(last_matba, float(piso["strike"])) - float(piso["prima"])
            add_forward(float(piso["qty_rem"]), px, rid, f"Piso — Cierre (piso {r1(piso['strike'])})")
            log(f"🔒 Cierre Piso: {r1(piso['qty_rem'])} tn a {r1(px)}.")
            piso["qty_rem"] = 0.0

    # --- Ultra Piso ---
    for up_item in list(st.session_state.ultra_pisos):
        if not isinstance(up_item, dict):
            continue
        # pendiente
        if float(up_item.get("qty_rem", 0.0)) > EPS:
            px = max(last_matba, float(up_item["strike"])) - float(up_item["prima"])
            add_forward(float(up_item["qty_rem"]), px, rid, f"Ultra Piso — Cierre (piso {r1(up_item['strike'])})")
            log(f"🔒 Cierre Ultra Piso (pendiente): {r1(up_item['qty_rem'])} tn a {r1(px)}.")
            up_item["qty_rem"] = 0.0
        # acumulado
        if float(up_item.get("fixed_qty", 0.0)) > EPS:
            add_forward(float(up_item["fixed_qty"]), float(up_item["fixed_avg"]), rid,
                        f"Ultra Piso — Acumulado (piso {r1(up_item['strike'])})")
            log(f"📊 Ultra Piso — Acumulado volcados: {r1(up_item['fixed_qty'])} tn a {r1(up_item['fixed_avg'])}.")
            up_item["fixed_qty"] = 0.0  # evitar duplicado

    # --- ULTRABANDA ---
    for band in list(st.session_state.bandas):
        if not isinstance(band, dict):
            continue
        piso  = float(band["piso"])
        techo = float(band["techo"])
        prima = float(band["prima"])

        # pendiente
        if float(band.get("qty_rem", 0.0)) > EPS:
            if last_matba < piso:
                px = piso - prima
            elif last_matba > techo:
                px = techo - prima
            else:
                px = last_matba - prima
            add_forward(float(band["qty_rem"]), px, rid,
                        f"ULTRABANDA — Cierre (piso {r1(piso)} / techo {r1(techo)})")
            log(f"🔒 Cierre ULTRABANDA (pendiente): {r1(band['qty_rem'])} tn a {r1(px)}.")
            band["qty_rem"] = 0.0

        # acumulado
        if float(band.get("fixed_qty", 0.0)) > EPS:
            add_forward(float(band["fixed_qty"]), float(band["fixed_avg"]), rid,
                        f"ULTRABANDA — Acumulado (piso {r1(piso)} / techo {r1(techo)})")
            log(f"📊 ULTRABANDA — Acumulado volcados: {r1(band['fixed_qty'])} tn a {r1(band['fixed_avg'])}.")
            band["fixed_qty"] = 0.0

    # --- Duplos ---
    for duplo in list(st.session_state.duplos):
        if not isinstance(duplo, dict):
            continue
        trigger = float(duplo["trigger"])
        accum   = float(duplo["accum"])

        # acumulado
        if float(duplo.get("fixed_qty", 0.0)) > EPS:
            add_forward(float(duplo["fixed_qty"]), float(duplo["fixed_avg"]), rid,
                        f"Duplo — (disp {r1(trigger)} / acum {r1(accum)}) — Acumulado")
            log(f"📊 Duplo — Acumulado: {r1(duplo['fixed_qty'])} tn a {r1(duplo['fixed_avg'])}.")
            duplo["fixed_qty"] = 0.0

        # liberado y no-procesado al forward
        unproc = max(0.0, float(duplo.get("qty_total", 0.0)) - float(duplo.get("processed", 0.0)))
        if float(duplo.get("released_qty", 0.0)) > EPS:
            add_forward(float(duplo["released_qty"]), last_fwd, rid, "Duplo — Liberado Cierre")
            log(f"🔚 Duplo — Liberado Cierre: {r1(duplo['released_qty'])} tn a {r1(last_fwd)}.")
            duplo["released_qty"] = 0.0
        if unproc > EPS:
            add_forward(unproc, last_fwd, rid, "Duplo — No Procesado Cierre")
            log(f"🔚 Duplo — No Procesado Cierre: {r1(unproc)} tn a {r1(last_fwd)}.")
            duplo["processed"] = float(duplo.get("processed", 0.0)) + unproc

    # --- Cargill Plus (expiración) ---
    for cp in list(st.session_state.cplus):
        if not isinstance(cp, dict) or cp.get("closed"):
            continue
        px = last_matba if last_matba > float(cp["techo"]) else last_fwd
        add_forward(float(cp["qty"]), px, rid, "Cargill Plus — Expiración")
        log(f"⏱️ Cargill Plus — Expiración: {r1(cp['qty'])} tn a {r1(px)}.")
        cp["closed"] = True

    # --- Venta faltante a cosecha ---
    fdf_all = pd.DataFrame(st.session_state.forwards)
    sold_total = float(fdf_all["qty"].sum()) if not fdf_all.empty else 0.0
    missing = max(0.0, float(st.session_state.total_volume) - sold_total)
    if missing > EPS:
        add_forward(missing, last_fwd, rid, "Venta volumen pendiente a cosecha")
        log(f"🌾 Venta volumen pendiente a cosecha: {r1(missing)} tn a {r1(last_fwd)}.")

    # KPIs
    fdf_all = pd.DataFrame(st.session_state.forwards)
    sold_qty = float(fdf_all["qty"].sum()) if not fdf_all.empty else 0.0
    sold_rev = float((fdf_all["qty"] * fdf_all["price"]).sum()) if not fdf_all.empty else 0.0
    avg_px   = (sold_rev / sold_qty) if sold_qty > 0 else 0.0

    st.session_state.already_finalized = True
    return {
        "avg_price_final": r1(avg_px),
        "benchmarks": {
            "Último día (MATBA)": r1(last_matba),
            "Promedio simple del período (MATBA)": r1(float(st.session_state.rounds_df["matba_price"].mean()))
        }
    }

# ========= Vista: Forwards + acumulados mientras corre =========
def forwards_view_df() -> pd.DataFrame:
    rows = [f for f in st.session_state.forwards]
    if not st.session_state.already_finalized:
        # Ultra Piso — Acumulado
        for up in st.session_state.ultra_pisos:
            if up["fixed_qty"]>EPS:
                rr = up.get("acum_last_round") or int(st.session_state.rounds_df.iloc[st.session_state.round_idx]["round_id"])
                rows.append({
                    "qty": r1(up["fixed_qty"]), "price": r1(up["fixed_avg"]),
                    "round": int(rr),
                    "tool": f"Ultra Piso — Acumulado (piso {r1(up['strike'])})"
                })
        # ULTRABANDA — Acumulado
        for band in st.session_state.bandas:
            if band["fixed_qty"]>EPS:
                rr = band.get("acum_last_round") or int(st.session_state.rounds_df.iloc[st.session_state.round_idx]["round_id"])
                rows.append({
                    "qty": r1(band["fixed_qty"]), "price": r1(band["fixed_avg"]),
                    "round": int(rr),
                    "tool": f"ULTRABANDA — Acumulado (piso {r1(band['piso'])} / techo {r1(band['techo'])})"
                })
        # Duplos — Acumulado
        for d in st.session_state.duplos:
            if d["fixed_qty"]>EPS:
                rr = d.get("acum_last_round") or int(st.session_state.rounds_df.iloc[st.session_state.round_idx]["round_id"])
                rows.append({
                    "qty": r1(d["fixed_qty"]), "price": r1(d["fixed_avg"]),
                    "round": int(rr),
                    "tool": f"Duplo — (disp {r1(d['trigger'])} / acum {r1(d['accum'])}) — Acumulado"
                })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["qty","price","round","tool"])
    return df_round1(df)

# ========= App =========
init_state()
df=st.session_state.rounds_df; idx=st.session_state.round_idx
mkt=cur_market(); matba=float(mkt["matba_price"]); forward=fwd_from_matba(matba)

st.title("🧪📈 Simulador de Soluciones Agrofinancieras (v0.15)")
st.caption("UP y ULTRABANDA con mismas bases de strike; ULTRABANDA: techo = strike + 15 y primas aún más baratas. Curva simulada con movimientos amplios. Panel de notificaciones y control de capacidad.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    total = st.number_input("Volumen total (tn)", min_value=1, max_value=2_000_000,
                            value=int(st.session_state.total_volume), step=50, disabled=st.session_state.already_finalized)
    if total != st.session_state.total_volume: st.session_state.total_volume = int(total)

    st.write("---")
    st.subheader("🗂️ Fuente de datos")
    source = st.radio("Elegí origen:", options=["Simulada v0.7","Archivo CSV/Excel"],
                      index=0 if st.session_state.data_source=="Simulada v0.7" else 1, disabled=st.session_state.already_finalized)
    up = st.file_uploader("Cargar archivo (date, matba_price)", type=["csv","xlsx","xls"]) if (source!="Simulada v0.7" and not st.session_state.already_finalized) else None
    if st.button("Aplicar fuente de datos / Reiniciar", use_container_width=True, disabled=st.session_state.already_finalized):
        ok,msg = apply_data_source(source, up)
        if ok: st.session_state.data_source = source; st.success(msg); st.rerun()
        else: st.error(msg)
    if st.session_state.data_source!="Simulada v0.7" and st.session_state.uploaded_snapshot:
        st.caption(f"Archivo en uso: {st.session_state.uploaded_snapshot}")

    st.write("---")
    st.subheader("🕒 Ronda actual")
    st.metric("Ronda", f"{int(mkt['round_id'])} / {int(df.iloc[-1]['round_id'])}")
    st.write(f"**Fecha:** {mkt['date']}")
    st.write(f"**Precio Soja Mayo MATBA 2026:** {r1(matba)}")
    st.write(f"**Forward soja mayo:** {r1(forward)}")

    c1,c2 = st.columns(2)
    with c1:
        if st.button("⏭️ Pasar de turno", use_container_width=True, disabled=idx>=len(df)-1 or st.session_state.already_finalized):
            process_round_effects()
            if st.session_state.round_idx < len(df)-1:
                st.session_state.round_idx += 1
                st.rerun()
    with c2:
        if st.button("🏁 Finalizar y calcular", use_container_width=True, disabled=st.session_state.already_finalized):
            process_round_effects()
            res=finalize_results()
            if res is not None:
                st.session_state["final_res"]=res
            st.session_state.round_idx = len(df)-1
            st.rerun()

# Catálogo + agregar
left,right = st.columns([1.15,1])
with left:
    st.subheader("🧰 Soluciones disponibles")
    catalog = build_catalog(mkt, idx, last_idx())
    st.dataframe(catalog[["nombre"]], use_container_width=True, hide_index=True)
    st.markdown("#### 1) Elegí herramienta y variante")
    c1,c2 = st.columns(2)
    with c1:
        tool = st.selectbox("Herramienta", options=catalog["tool_id"],
                            format_func=lambda t: catalog.set_index("tool_id").loc[t,"nombre"],
                            disabled=st.session_state.already_finalized)
    with c2:
        variants = catalog.set_index("tool_id").loc[tool,"variants"]
        var_idx = st.selectbox("Variante", options=list(range(len(variants))),
                               format_func=lambda i: variants[i]["label"],
                               disabled=st.session_state.already_finalized)
    if st.button("✅ Seleccionar", use_container_width=True, disabled=st.session_state.already_finalized):
        select_tool_variant(tool, variants[var_idx]); st.success("Herramienta seleccionada. Completá la cantidad y agregá.")

    st.markdown("#### 2) Completar y agregar a posición")
    with st.form("add_form"):
        qty = st.number_input("Cantidad (tn)", min_value=1, max_value=1_000_000, value=100, step=50, disabled=st.session_state.already_finalized)
        det=""
        if st.session_state.sel_tool and st.session_state.sel_variant:
            t=st.session_state.sel_tool; v=st.session_state.sel_variant
            if t=="forward": det=f"Forward: {v['price']}"
            elif t in ("piso","ultra_piso"): det=f"Strike {v['strike']} / Prima {v['prima']}"
            elif t=="ultra_banda": det=f"Piso {v['piso']} / Techo {v['techo']} / Prima {v['prima']}"
            elif t=="cargill_plus": det=f"Techo {v['techo']} / Bonificación {v['bonificacion']}"
            elif t=="duplo": det=f"Disparador {v['trigger']} / Acumulación {v['accum']}"
        left_cap = capacity_left()
        st.info((det or "Elegí y apretá **Seleccionar** arriba para ver los campos.") + f"  | Capacidad libre: **{r1(left_cap)} tn**")
        if st.form_submit_button("➕ Agregar", disabled=st.session_state.already_finalized):
            add_decision(int(qty)); st.rerun()

    # Panel de notificaciones / historial
    st.markdown("#### 📝 Notificaciones")
    if st.session_state.logs:
        for m in reversed(st.session_state.logs[-12:]):
            st.write(m)
    else:
        st.caption("Acá vas a ver un historial de movimientos, avisos de capacidad, cierres, etc.")

with right:
    st.subheader("📦 Tu posición")

    # Forwards (vista agregada)
    with st.expander("Forwards (efectivos)", expanded=True):
        fwd_df = forwards_view_df()
        st.dataframe(fwd_df, hide_index=True, use_container_width=True)

    # Pisos
    if st.session_state.pisos:
        st.markdown("**PISO ASEGURADO**")
        p_df = pd.DataFrame([{
            "id":p["id"],"cantidad total":p["qty_total"],"cantidad remanente":p["qty_rem"],
            "piso":p["strike"],"prima":p["prima"],"cantidad fijada":p["fixed_qty"],
            "precio promedio":p["fixed_avg"],"cantidad pendiente":max(0.0,p["qty_total"]-p["fixed_qty"])
        } for p in st.session_state.pisos])
        st.dataframe(df_round1(p_df.drop(columns=["id"])), hide_index=True, use_container_width=True)
        for p in st.session_state.pisos:
            with st.expander(f"Fijación anticipada — Piso Asegurado (piso {r1(p['strike'])})", expanded=False):
                col1,col2,col3 = st.columns([1.2,1,1.2])
                col1.write(f"Fijado: {r1(p['fixed_qty'])} tn a {r1(p['fixed_avg'])}")
                qfix = col2.number_input("Cantidad a fijar ahora", min_value=0.0, max_value=float(p["qty_rem"]),
                                         value=float(p["qty_rem"]), step=1.0, key=f"pfix_{p['id']}")
                if col3.button("Fijar ahora", key=f"pfbtn_{p['id']}", disabled=st.session_state.already_finalized):
                    df_now=st.session_state.rounds_df; matba_now=float(df_now.iloc[st.session_state.round_idx]["matba_price"])
                    fix_price=max(matba_now,p["strike"])-p["prima"]
                    add_forward(qfix, fix_price, int(df_now.iloc[st.session_state.round_idx]["round_id"]),
                                f"Fijación Piso Asegurado (piso {r1(p['strike'])})")
                    p["qty_rem"]=r1(p["qty_rem"]-qfix)
                    p["fixed_qty"],p["fixed_avg"]=weighted_avg(p["fixed_qty"],p["fixed_avg"],qfix,fix_price)
                    p["early_fixed_qty"]=p.get("early_fixed_qty",0.0)+qfix
                    log(f"✅ Piso Asegurado: fijaste {r1(qfix)} tn a {r1(fix_price)}.")
                    st.rerun()

    # Ultra Piso
    if st.session_state.ultra_pisos:
        st.markdown("**ULTRA PISO (Piso Promediando)**")
        up_df = pd.DataFrame([{
            "id":up["id"],"cantidad total":up["qty_total"],"cantidad remanente":up["qty_rem"],
            "piso":up["strike"],"prima":up["prima"],"cantidad diaria":up["daily_qty"],
            "cantidad fijada (acumulado)":up["fixed_qty"],"precio promedio (acumulado)":up["fixed_avg"],
            "cantidad pendiente":max(0.0,up["qty_total"]-up["fixed_qty"])
        } for up in st.session_state.ultra_pisos])
        st.dataframe(df_round1(up_df.drop(columns=["id"])), hide_index=True, use_container_width=True)

        for up in st.session_state.ultra_pisos:
            with st.expander(f"Fijación anticipada — Ultra Piso (piso {r1(up['strike'])})", expanded=False):
                c1,c2,c3 = st.columns([1.3,1,1.5])
                pendiente = r1(max(0.0, up["qty_rem"]))
                c1.write(f"Acumulado: **{r1(up['fixed_qty'])} tn** a **{r1(up['fixed_avg'])}**")
                c2.write(f"Pendiente: **{pendiente} tn**")
                if c3.button("Fijar TODO el pendiente al valor de hoy", key=f"upfix_{up['id']}", disabled=st.session_state.already_finalized):
                    early_fix_ultra_piso_all_pending(up["id"])
                    st.rerun()

    # ULTRABANDA
    if st.session_state.bandas:
        st.markdown("**ULTRABANDA**")
        ub_df = pd.DataFrame([{
            "cantidad total":band["qty_total"],"cantidad remanente":band["qty_rem"],
            "piso":band["piso"],"techo":band["techo"],"prima":band["prima"],
            "cantidad diaria":band["daily_qty"],"cantidad fijada":band["fixed_qty"],"precio promedio":band["fixed_avg"]
        } for band in st.session_state.bandas])
        st.dataframe(df_round1(ub_df), hide_index=True, use_container_width=True)

    # Cargill Plus
    if st.session_state.cplus:
        st.markdown("**Cargill Plus (Potenciador)**")
        st.dataframe(df_round1(pd.DataFrame(st.session_state.cplus)), hide_index=True, use_container_width=True)

    # Duplos
    if st.session_state.duplos:
        st.markdown("**Duplos (Acumuladores)**")
        dp_df = pd.DataFrame([{
            "cantidad total":d["qty_total"], "nivel disparador":d["trigger"], "nivel de acumulación":d["accum"],
            "cantidad diaria":d["daily_qty"], "cantidad con precio":d["fixed_qty"], "precio acumulado":d["fixed_avg"],
            "cantidad liberada":d["released_qty"]
        } for d in st.session_state.duplos])
        st.dataframe(df_round1(dp_df), hide_index=True, use_container_width=True)

# KPIs / gráficos
fdf = forwards_view_df()
sold_qty = float(fdf["qty"].sum()) if not fdf.empty else 0.0
sold_rev = float((fdf["qty"]*fdf["price"]).sum()) if not fdf.empty else 0.0
avg_now = r1((sold_rev/sold_qty) if sold_qty>0 else 0.0)

sold_eff, committed_wo, committed_total = committed_breakdown()

st.markdown("### 📈 Vista general")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Forwards vendidos (tn)", int(round(sold_eff)))
k2.metric("Comprometido sin precio (tn)", int(round(committed_wo)))
k3.metric("Total comprometido (tn)", int(round(committed_total)))
k4.metric("Capacidad libre (tn)", int(round(capacity_left())))
st.caption("El total comprometido no puede superar el volumen total configurado.")

st.markdown("### 🧭 Volumen vendido por ronda")
plot_df = df.copy(); plot_df["ventas_tn"]=0.0
for s in fdf.to_dict("records"):
    ridx = int(s["round"])-1
    if 0<=ridx<len(plot_df): plot_df.loc[ridx,"ventas_tn"] += float(s["qty"])
st.plotly_chart(px.bar(plot_df, x="date", y="ventas_tn", title="Volumen vendido por ronda"),
                use_container_width=True)

# Resultado final + comparativa acotada
if "final_res" in st.session_state:
    st.markdown("---"); st.header("🏆 Resultado final")
    res=st.session_state["final_res"]
    st.subheader(f"Promedio final alcanzado: **{res['avg_price_final']} USD/tn**")
    st.markdown("**Benchmarks (referencia MATBA)**")
    for k,v in res["benchmarks"].items(): st.write(f"- {k}: {v} USD/tn")

    comp_df=pd.DataFrame({
        "Estrategia":["Vos (simulación)"]+list(res["benchmarks"].keys()),
        "Precio (USD/tn)":[res["avg_price_final"]]+list(res["benchmarks"].values())
    })

    # Eje vertical acotado ~20 USD (±10) con dtick=1 y barras angostas
    vmin=float(comp_df["Precio (USD/tn)"].min()); vmax=float(comp_df["Precio (USD/tn)"].max())
    center=(vmin+vmax)/2.0
    y0=center-10.0; y1=center+10.0
    fig = px.bar(comp_df, x="Estrategia", y="Precio (USD/tn)", title="Comparativa de cierre")
    fig.update_traces(width=0.35)
    fig.update_yaxes(range=[y0,y1], dtick=1)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("v0.15 — Nombre nuevo; UP ~25% < Piso; ULTRABANDA (mismos strikes) techo = strike+15 y ~30% < UP; curva más volátil; panel de notificaciones; control de capacidad; cierre con venta pendiente a cosecha. Forward = MATBA − 3.")
