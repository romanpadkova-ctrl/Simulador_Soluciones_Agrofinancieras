import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
import uuid
from typing import Dict, Any

# ==================== BOOT / PAGE ====================
st.set_page_config(page_title="Simulador de Soluciones Agrofinancieras", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] > div:first-child { background: linear-gradient(180deg,#0f1a12,#0b0f0c); }
    .block-container { padding-top: 1rem; }
    section[data-testid="stSidebar"] button { width: 100% !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HEARTBEAT anti-inactividad (ping real cada 30s) ---
components.html(
    """
    <script>
      (function() {
        const ping = () => {
          // intenta un endpoint interno súper liviano
          fetch("/_stcore/healthz?" + Date.now(), {cache: "no-store"})
            .catch(() => {
              // fallback: hace un HEAD a la misma página
              fetch(window.location.href.split("#")[0] + "?" + Date.now(), {
                method: "HEAD",
                cache: "no-store"
              }).catch(()=>{});
            });
        };
        // primer ping rápido y luego cada 30s
        ping();
        setInterval(ping, 30000);
      })();
    </script>
    """,
    height=0,
)

# Keep-alive (si no existe en tu versión no rompe)
try:
    st.autorefresh(interval=55_000, key="keepalive", debounce=True)
except Exception:
    pass

# ==================== HELPERS ====================
EPS = 1e-6
def r05(x): return round(round(float(x)*2)/2, 1)
def r1(x):  return round(float(x), 1)

def df_round1(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float).round(1)
    return df

def fwd_from_matba(matba: float) -> float:
    return r05(matba - 3.0)

def log(msg: str):
    st.session_state.logs.append(msg)

def safe_rerun():
    if hasattr(st, "experimental_rerun"): st.experimental_rerun()
    else: st.rerun()


# ==================== PERSISTENCIA DE ESTADO (anti-reset) ====================
@st.cache_data(show_spinner=False)
def _state_store() -> Dict[str, Dict[str, Any]]:
    return {}

def get_or_create_sid() -> str:
    qp = st.query_params
    sid = qp.get("sid", None)
    if not sid:
        sid = uuid.uuid4().hex[:12]
        st.query_params["sid"] = sid
    elif isinstance(sid, list):
        sid = sid[0]
    return sid

PERSIST_KEYS = [
    "data_source","uploaded_snapshot","rounds_df","round_idx","total_volume",
    "forwards","pisos","ultra_pisos","bandas","cplus","duplos","sel_tool",
    "sel_variant","already_finalized","logs","sold_by_round","undo_stack","final_res"
]

def snapshot_state() -> Dict[str, Any]:
    snap = {}
    ss = st.session_state
    for k in PERSIST_KEYS:
        if k in ss:
            snap[k] = ss[k]
    return snap

def restore_state(snap: Dict[str, Any]):
    if not snap:
        return
    ss = st.session_state
    for k, v in snap.items():
        ss[k] = v

def persist_session(sid: str):
    store = _state_store()
    store[sid] = snapshot_state()
    _state_store.clear()
    _ = _state_store()

def try_restore_session(sid: str):
    store = _state_store()
    if sid in store and store[sid]:
        restore_state(store[sid])

# ==================== ROUNDS (CURVA) ====================
def demo_rounds(start_month=9, rounds=9):
    pattern = [290, 300, 315, 312, 295, 290, 305, 292, 280]
    rows=[]
    for i in range(rounds):
        rows.append({
            "round_id": i+1,
            "date": f"2025-{((start_month-1+i)%12)+1:02d}-01",
            "matba_price": r05(pattern[i]),
        })
    return pd.DataFrame(rows)

# ==================== PRICING RULES ====================
VOL_MULT = {"baja":0.8,"media":1.0,"alta":1.3}
def tenor_months(curr_idx, last_idx): return max(1, last_idx - curr_idx)

def premium_option_base(matba: float, months_left: int, vol_level="media") -> float:
    mult = VOL_MULT.get(vol_level, 1.0)
    time_factor = (months_left / 12.0) ** 0.7
    price_factor = (matba / 300.0) ** 0.9
    base = 0.020 * matba * time_factor * price_factor * mult
    return r05(base)

def piso_variants(matba, prem_base):
    atm = r05(max(10.0, prem_base * 1.30))
    m5  = r05(max( 6.0, atm * 0.60))
    m10 = r05(max( 3.0, atm * 0.35))
    return [
        {"label": f"Strike {r05(matba)} / Prima {atm}",   "strike": r05(matba),   "prima": atm},
        {"label": f"Strike {r05(matba-5)} / Prima {m5}",  "strike": r05(matba-5), "prima": m5},
        {"label": f"Strike {r05(matba-10)} / Prima {m10}", "strike": r05(matba-10), "prima": m10},
    ]

def ultra_piso_variants_from_piso(piso_opts):
    opts=[]
    for o in piso_opts:
        strike = r05(o["strike"])
        prima  = r05(max(1.0, o["prima"] * 0.75))
        opts.append({"label": f"Strike {strike} / Prima {prima}", "strike": strike, "prima": prima})
    return opts

def ultra_banda_variants_from_ultrapiso(up_opts):
    opts=[]
    for o in up_opts:
        piso  = r05(o["strike"])
        techo = r05(piso + 15)
        prima = r05(max(1.0, o["prima"] * 0.70))
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
    t1=r05(matba-3);  a1=r05(matba+10)
    t2=r05(matba-5);  a2=r05(matba+7)
    t3=r05(matba-10); a3=r05(matba+5)
    opts=[{"trigger":t1,"accum":a1},{"trigger":t2,"accum":a2},{"trigger":t3,"accum":a3}]
    for o in opts: o["label"]=f"Disparador {o['trigger']} / Acumulación {o['accum']}"
    return opts

def build_catalog(mkt_row, curr_idx, last_i):
    matba=float(mkt_row["matba_price"])
    forward=fwd_from_matba(matba)
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

# ==================== STATE ====================
def init_state():
    ss = st.session_state
    if "data_source" not in ss: ss.data_source="Simulada v0.7"
    if "uploaded_snapshot" not in ss: ss.uploaded_snapshot=None
    if "rounds_df" not in ss: ss.rounds_df=demo_rounds()
    if "round_idx" not in ss: ss.round_idx=0
    if "total_volume" not in ss: ss.total_volume=1000
    if "forwards" not in ss: ss.forwards=[]
    if "pisos" not in ss: ss.pisos=[]
    if "ultra_pisos" not in ss: ss.ultra_pisos=[]
    if "bandas" not in ss: ss.bandas=[]
    if "cplus" not in ss: ss.cplus=[]
    if "duplos" not in ss: ss.duplos=[]
    if "sel_tool" not in ss: ss.sel_tool=None
    if "sel_variant" not in ss: ss.sel_variant=None
    if "already_finalized" not in ss: ss.already_finalized=False
    if "logs" not in ss: ss.logs=[]
    if "sold_by_round" not in ss: ss.sold_by_round={}
    if "undo_stack" not in ss: ss.undo_stack=[]

def reset_position():
    ss = st.session_state
    ss.round_idx=0
    ss.forwards=[]; ss.pisos=[]; ss.ultra_pisos=[]; ss.bandas=[]; ss.cplus=[]; ss.duplos=[]
    ss.sel_tool=None; ss.sel_variant=None
    ss.already_finalized=False
    ss.logs=[]; ss.sold_by_round={}; ss.undo_stack=[]

def cur_market(): return st.session_state.rounds_df.iloc[st.session_state.round_idx].to_dict()
def last_idx(): return len(st.session_state.rounds_df)-1
def add_forward(qty, price, round_id, tag):
    st.session_state.forwards.append({"qty": r1(qty), "price": r1(price), "round": int(round_id), "tool": tag})

# ==================== MÉTRICAS CAPACIDAD ====================
def committed_breakdown():
    """
    sold_eff: forwards efectivos (solo forwards, incl. C+ forward 1:1)
    with_price: volumen con precio (herramientas) -> NO suma forwards
    without_price: pendiente (herramientas + techos C+)
    total_committed = sold_eff + with_price + without_price
    """
    ss = st.session_state
    sold_eff = sum(f["qty"] for f in ss.forwards)

    p_con  = sum(p.get("fixed_qty",0.0) for p in ss.pisos)
    up_con = sum(up.get("fixed_qty",0.0) for up in ss.ultra_pisos)
    ub_con = sum(b.get("fixed_qty",0.0) for b in ss.bandas)
    dp_con = sum(d.get("fixed_qty",0.0) for d in ss.duplos)
    with_price = r1(p_con + up_con + ub_con + dp_con)

    p_pending  = sum(max(0.0, p.get("qty_total",0.0)  - p.get("fixed_qty",0.0)) for p in ss.pisos)
    up_pending = sum(max(0.0, up.get("qty_total",0.0) - up.get("fixed_qty",0.0)) for up in ss.ultra_pisos)
    ub_pending = sum(max(0.0, b.get("qty_total",0.0)  - b.get("fixed_qty",0.0)) for b in ss.bandas)
    dp_pending = sum(max(0.0, d.get("qty_total",0.0)  - d.get("fixed_qty",0.0)) for d in ss.duplos)
    cp_pending = sum(c["qty"] for c in ss.cplus if not c.get("closed", False))
    without_price = r1(p_pending + up_pending + ub_pending + dp_pending + cp_pending)

    total_committed = r1(sold_eff + with_price + without_price)
    return r1(sold_eff), r1(with_price), r1(without_price), r1(total_committed)

def capacity_left():
    _, _, _, tot = committed_breakdown()
    return max(0.0, float(st.session_state.total_volume) - float(tot))

# ==================== DECISIONES / ALTA ====================
def push_undo():
    ss = st.session_state
    snapshot = {
        "forwards": ss.forwards.copy(),
        "pisos": [p.copy() for p in ss.pisos],
        "ultra_pisos": [p.copy() for p in ss.ultra_pisos],
        "bandas": [b.copy() for b in ss.bandas],
        "cplus": [c.copy() for c in ss.cplus],
        "duplos": [d.copy() for d in ss.duplos],
        "sold_by_round": ss.sold_by_round.copy(),
        "logs": ss.logs.copy(),
    }
    ss.undo_stack.append(snapshot)

def undo_last():
    ss = st.session_state
    if not ss.undo_stack:
        log("ℹ️ No hay acción para deshacer.")
        return
    snap = ss.undo_stack.pop()
    ss.forwards = snap["forwards"]
    ss.pisos = snap["pisos"]
    ss.ultra_pisos = snap["ultra_pisos"]
    ss.bandas = snap["bandas"]
    ss.cplus = snap["cplus"]
    ss.duplos = snap["duplos"]
    ss.sold_by_round = snap["sold_by_round"]
    ss.logs = snap["logs"]
    log("↩️ Se deshizo la última acción.")

def select_tool_variant(tool_id, variant):
    st.session_state.sel_tool=tool_id
    st.session_state.sel_variant=variant

def add_sold_round(rid, qty):
    if qty <= EPS: return
    st.session_state.sold_by_round[rid] = st.session_state.sold_by_round.get(rid, 0.0) + float(qty)

def add_decision(qty:int):
    ss = st.session_state
    if not ss.sel_tool or not ss.sel_variant:
        return
    push_undo()

    mkt=cur_market(); rid=int(mkt["round_id"]); matba=float(mkt["matba_price"]); forward=fwd_from_matba(matba)
    total_rounds=len(ss.rounds_df)
    rounds_left = (total_rounds - rid)  # comienza en la ronda SIGUIENTE
    if rounds_left <= 0: rounds_left = 1

    t=ss.sel_tool; v=ss.sel_variant
    left = capacity_left()

    if t == "cargill_plus":
        if qty*2 > left + EPS:
            log(f"⚠️ No se puede suscribir: Cargill Plus {r1(qty)} tn equivale a {r1(qty*2)} tn y supera capacidad libre {r1(left)} tn.")
            return
    else:
        if qty > left + EPS:
            log(f"⚠️ No se puede suscribir: capacidad libre {r1(left)} tn, intentaste {r1(qty)} tn.")
            return

    if t=="forward":
        add_forward(qty, forward, rid, "Forward")
        add_sold_round(rid, qty)
        log(f"✅ Venta Forward {r1(qty)} tn a {r1(forward)} (ronda {rid}).")

    elif t=="piso":
        ss.pisos.append({
            "id":f"piso_{rid}_{len(ss.pisos)+1}",
            "qty_total":r1(qty),"qty_rem":r1(qty),"strike":r05(v["strike"]),"prima":r05(v["prima"]),
            "round_open":rid,"fixed_qty":0.0,"fixed_avg":0.0
        })
        log(f"🛡️ Piso Asegurado suscrito: {r1(qty)} tn | strike {r1(v['strike'])} | prima {r1(v['prima'])} (ronda {rid}).")

    elif t=="ultra_piso":
        dq=float(qty)/rounds_left
        ss.ultra_pisos.append({
            "id":f"up_{rid}_{len(ss.ultra_pisos)+1}",
            "qty_total":float(qty),"qty_rem":float(qty),"strike":r05(v["strike"]),"prima":r05(v["prima"]),
            "round_open":rid,"daily_qty":dq,"fixed_qty":0.0,"fixed_avg":0.0,
            "acum_last_round": None
        })
        log(f"🧮 Ultra Piso suscrito: {r1(qty)} tn | strike {r1(v['strike'])} | prima {r1(v['prima'])} (ronda {rid}).")

    elif t=="ultra_banda":
        dq=float(qty)/rounds_left
        ss.bandas.append({
            "id":f"ub_{rid}_{len(ss.bandas)+1}",
            "qty_total":float(qty),"qty_rem":float(qty),
            "piso":r05(v["piso"]),"techo":r05(v["techo"]),"prima":r05(v["prima"]),
            "round_open":rid,"daily_qty":dq,"fixed_qty":0.0,"fixed_avg":0.0,
            "acum_last_round": None
        })
        log(f"📦 ULTRABANDA suscrita: {r1(qty)} tn | piso {r1(v['piso'])} / techo {r1(v['techo'])} | prima {r1(v['prima'])} (ronda {rid}).")

    elif t=="cargill_plus":
        add_forward(qty, forward+float(v["bonificacion"]), rid,
                    f"Cargill Plus — Forward ({r1(forward)} + {r1(v['bonificacion'])})")
        add_sold_round(rid, qty)
        ss.cplus.append({"qty":r1(qty),"techo":r05(v["techo"]),
                         "bonificacion":r05(v["bonificacion"]),"round_open":rid,"closed":False})
        log(f"➕ Cargill Plus suscrito: {r1(qty)} tn | techo {r1(v['techo'])} | bonif {r1(v['bonificacion'])} (ronda {rid}). Generó forward 1:1 a {r1(forward+v['bonificacion'])}.")

    elif t=="duplo":
        dq=r1(float(qty)/rounds_left)
        ss.duplos.append({
            "id":f"dp_{rid}_{len(ss.duplos)+1}",
            "qty_total":r1(qty),"trigger":r05(v["trigger"]),"accum":r05(v["accum"]),
            "round_open":rid,"daily_qty":dq,"fixed_qty":0.0,"fixed_avg":0.0,
            "released_qty":0.0,"processed":0.0,"acum_last_round":None
        })
        log(f"🔁 Duplo suscrito: {r1(qty)} tn | disparador {r1(v['trigger'])} | acumulación {r1(v['accum'])} (ronda {rid}).")

    # persistimos luego de la decisión
    try:
        persist_session(sid)
    except Exception:
        pass

# ==================== MOTOR DE RONDA ====================
def weighted_avg(old_qty,old_avg,add_qty,add_price):
    add_qty=float(add_qty); add_price=float(add_price)
    if add_qty<=0: return r1(old_qty), r1(old_avg)
    new_qty=old_qty+add_qty
    new_avg=(old_qty*old_avg + add_qty*add_price)/new_qty if new_qty>0 else 0.0
    return r1(new_qty), r1(new_avg)

def process_round_effects():
    df=st.session_state.rounds_df; idx=st.session_state.round_idx
    matba=float(df.iloc[idx]["matba_price"]); rid=int(df.iloc[idx]["round_id"])

    # ULTRABANDA (desde ronda siguiente a la apertura)
    for b in st.session_state.bandas:
        if rid <= b["round_open"] or b["qty_rem"]<=EPS:
            continue
        qty=min(b["daily_qty"], b["qty_rem"])
        if qty<=EPS: continue
        if matba < b["piso"]: px=b["piso"]-b["prima"]
        elif matba > b["techo"]: px=b["techo"]-b["prima"]
        else: px=matba-b["prima"]
        b["fixed_qty"],b["fixed_avg"]=weighted_avg(b["fixed_qty"],b["fixed_avg"],qty,px)
        b["qty_rem"]=max(0.0,b["qty_rem"]-qty)
        b["acum_last_round"]=rid
        add_sold_round(rid, qty)

    # Ultra Piso
    for up in st.session_state.ultra_pisos:
        if rid <= up["round_open"] or up["qty_rem"]<=EPS:
            continue
        qty=min(up["daily_qty"], up["qty_rem"])
        if qty<=EPS: continue
        px=max(matba, up["strike"]) - up["prima"]
        up["fixed_qty"],up["fixed_avg"]=weighted_avg(up["fixed_qty"],up["fixed_avg"],qty,px)
        up["qty_rem"]=max(0.0, up["qty_rem"]-qty)
        up["acum_last_round"]=rid
        add_sold_round(rid, qty)

    # Duplos
    for d in st.session_state.duplos:
        if rid <= d["round_open"] or d["processed"]>=d["qty_total"]-EPS:
            continue
        remaining=max(0.0,d["qty_total"]-d["processed"]); qty=min(d["daily_qty"], remaining)
        if qty<=EPS: continue
        if matba>=d["accum"]:
            px=d["accum"]; d["fixed_qty"],d["fixed_avg"]=weighted_avg(d["fixed_qty"],d["fixed_avg"],qty,px)
            add_sold_round(rid, qty)
        elif d["trigger"]<=matba<d["accum"]:
            half=qty/2.0; px=d["accum"]
            d["fixed_qty"],d["fixed_avg"]=weighted_avg(d["fixed_qty"],d["fixed_avg"],half,px)
            d["released_qty"]=r1(d["released_qty"]+half)
            add_sold_round(rid, half)
        else:
            d["released_qty"]=r1(d["released_qty"]+qty)
        d["processed"]=r1(d["processed"]+qty)
        d["acum_last_round"]=rid

# ========= Fijaciones anticipadas =========
def early_fix_ultra_piso_all_pending(up_id):
    df=st.session_state.rounds_df; idx=st.session_state.round_idx
    matba=float(df.iloc[idx]["matba_price"]); rid=int(df.iloc[idx]["round_id"])
    for up in st.session_state.ultra_pisos:
        if up["id"]!=up_id: continue
        pending = max(0.0, up["qty_rem"])
        if pending <= EPS:
            log("ℹ️ Ultra Piso: no hay pendiente para fijar.")
            return
        px = max(matba, up["strike"]) - up["prima"]
        up["qty_rem"] = 0.0
        up["fixed_qty"], up["fixed_avg"] = weighted_avg(up["fixed_qty"], up["fixed_avg"], pending, px)
        add_sold_round(rid, pending)
        log(f"✅ Ultra Piso fijado completamente (pendiente {r1(pending)} tn) a {r1(px)}.")
        return

# ==================== CIERRE ====================
def finalize_results():
    if st.session_state.already_finalized:
        return None
    df=st.session_state.rounds_df; last_matba=float(df.iloc[-1]["matba_price"]); rid=int(df.iloc[-1]["round_id"])
    last_forward = fwd_from_matba(last_matba)

    # Piso asegurado -> todo a con precio
    for p in st.session_state.pisos:
        if p.get("qty_rem",0.0)>EPS:
            px=max(last_matba,p["strike"])-p["prima"]
            qty=p["qty_rem"]; p["qty_rem"]=0.0
            p["fixed_qty"],p["fixed_avg"]=weighted_avg(p["fixed_qty"],p["fixed_avg"],qty,px)
            add_sold_round(rid, qty)
            log(f"🔒 Cierre Piso: {r1(qty)} tn a {r1(px)}.")

    # Ultra Piso
    for up in st.session_state.ultra_pisos:
        if up.get("qty_rem",0.0)>EPS:
            px=max(last_matba,up["strike"])-up["prima"]
            qty=up["qty_rem"]; up["qty_rem"]=0.0
            up["fixed_qty"],up["fixed_avg"]=weighted_avg(up["fixed_qty"],up["fixed_avg"],qty,px)
            add_sold_round(rid, qty)
            log(f"🔒 Cierre Ultra Piso (pendiente): {r1(qty)} tn a {r1(px)}.")

    # ULTRABANDA
    for b in st.session_state.bandas:
        if b.get("qty_rem",0.0)>EPS:
            if last_matba<b["piso"]: px=b["piso"]-b["prima"]
            elif last_matba>b["techo"]: px=b["techo"]-b["prima"]
            else: px=last_matba-b["prima"]
            qty=b["qty_rem"]; b["qty_rem"]=0.0
            b["fixed_qty"],b["fixed_avg"]=weighted_avg(b["fixed_qty"],b["fixed_avg"],qty,px)
            add_sold_round(rid, qty)
            log(f"🔒 Cierre ULTRABANDA (pendiente): {r1(qty)} tn a {r1(px)}.")

    # Duplos: libera / no procesado -> con precio al último
    for d in st.session_state.duplos:
        unproc=max(0.0,d["qty_total"]-d["processed"])
        if d.get("released_qty",0.0)>EPS:
            qty = d["released_qty"]; d["released_qty"]=0.0
            d["fixed_qty"], d["fixed_avg"]=weighted_avg(d["fixed_qty"], d["fixed_avg"], qty, last_forward)
            add_sold_round(rid, qty)
            log(f"🔚 Duplo — Liberado Cierre: {r1(qty)} tn a {r1(last_forward)}.")
        if unproc>EPS:
            d["processed"]=r1(d["processed"]+unproc)
            d["fixed_qty"], d["fixed_avg"]=weighted_avg(d["fixed_qty"], d["fixed_avg"], unproc, last_forward)
            add_sold_round(rid, unproc)
            log(f"🔚 Duplo — No Procesado Cierre: {r1(unproc)} tn a {r1(last_forward)}.")

    # C+ techos -> con precio (no como forward en lista de forwards; lo contamos en métricas/posición de herramientas)
    for c in st.session_state.cplus:
        if c.get("closed"): continue
        px = last_matba if last_matba>c["techo"] else last_forward
        qty = c["qty"]
        add_sold_round(rid, qty)
        c["closed"] = True
        log(f"⏱️ Cargill Plus — Expiración: {r1(qty)} tn a {r1(px)}.")

    # Producción libre sin comprometer (si existiera) → crear una línea visible en posición
    ss = st.session_state
    sold_eff, with_price, without_price, total_committed = committed_breakdown()
    missing = max(0.0, float(ss.total_volume) - float(total_committed))
    if missing > EPS:
        # registrar en gráfico por ronda
        add_sold_round(rid, missing)
        # agregar a posición como un "forward" rotulado de expiración para que se vea
        add_forward(missing, last_forward, rid, "Expiración — Volumen libre sin comprometer")
        log(f"🌾 Producción libre sin comprometer fijada en la última ronda: {r1(missing)} tn a {r1(last_forward)}.")

    ss.already_finalized=True

    # KPI final ponderado (forwards + con precio)
    fdf = forwards_view_df(include_only_forwards=False)
    sold_qty=float(fdf["qty"].sum()) if not fdf.empty else 0.0
    sold_rev=float((fdf["qty"]*fdf["price"]).sum()) if not fdf.empty else 0.0
    avg_px=(sold_rev/sold_qty) if sold_qty>0 else 0.0

    return {"avg_price_final": r1(avg_px),
            "benchmarks":{"Último día (MATBA)": r1(last_matba),
                          "Promedio simple del período (MATBA)": r1(float(ss.rounds_df["matba_price"].mean()))}
           }

# ==================== VISTAS / POSICIÓN ====================
def forwards_view_df(include_only_forwards=True) -> pd.DataFrame:
    rows = []
    # Forwards efectivos + lo que decidamos mostrar como línea de forward (expiración libre)
    for f in st.session_state.forwards:
        rows.append(f.copy())

    # Mostrar SIEMPRE las fijaciones de herramientas
    if not include_only_forwards:
        rid_now = int(st.session_state.rounds_df.iloc[st.session_state.round_idx]["round_id"])

        for p in st.session_state.pisos:
            if p.get("fixed_qty",0.0) > EPS:
                rows.append({"qty": r1(p["fixed_qty"]), "price": r1(p["fixed_avg"]),
                             "round": rid_now, "tool": f"Piso Asegurado — Fijado (piso {r1(p['strike'])})"})

        for up in st.session_state.ultra_pisos:
            if up.get("fixed_qty",0.0) > EPS:
                rr = up.get("acum_last_round") or rid_now
                rows.append({"qty": r1(up["fixed_qty"]), "price": r1(up["fixed_avg"]),
                             "round": rr, "tool": f"Ultra Piso — Acumulado (piso {r1(up['strike'])})"})

        for b in st.session_state.bandas:
            if b.get("fixed_qty",0.0) > EPS:
                rr = b.get("acum_last_round") or rid_now
                rows.append({"qty": r1(b["fixed_qty"]), "price": r1(b["fixed_avg"]),
                             "round": rr, "tool": f"ULTRABANDA — Acumulado (piso {r1(b['piso'])} / techo {r1(b['techo'])})"})

        for d in st.session_state.duplos:
            if d.get("fixed_qty",0.0)>EPS:
                rr = d.get("acum_last_round") or rid_now
                rows.append({"qty": r1(d["fixed_qty"]), "price": r1(d["fixed_avg"]),
                             "round": rr, "tool": f"Duplo — Acumulado (disp {r1(d['trigger'])} / acum {r1(d['accum'])})"})

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["qty","price","round","tool"])
    return df_round1(df)

# ==================== APP ====================
sid = get_or_create_sid()
init_state()
try_restore_session(sid)
df=st.session_state.rounds_df; idx=st.session_state.round_idx
mkt=cur_market(); matba=float(mkt["matba_price"]); forward=fwd_from_matba(matba)

st.title("🧪📈 Simulador de Soluciones Agrofinancieras")

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
        st.session_state.rounds_df=demo_rounds(); st.session_state.uploaded_snapshot=None
        reset_position(); st.session_state.data_source = "Simulada v0.7"
        st.success("Usando curva simulada."); persist_session(sid); safe_rerun()

    st.write("---")
    st.subheader("🕒 Ronda actual")
    st.metric("Ronda", f"{int(mkt['round_id'])} / {int(df.iloc[-1]['round_id'])}")
    st.write(f"**Fecha:** {mkt['date']}")
    st.write(f"**Precio Soja Mayo MATBA 2026:** {r1(matba)}")
    st.write(f"**Forward soja mayo:** {r1(forward)}")

    # <<< CLAVE >>> primero avanza la ronda, luego aplica efectos
    btn_next = st.button("⏭️ Pasar de turno", use_container_width=True, disabled=idx>=len(df)-1 or st.session_state.already_finalized)
    if btn_next:
        if st.session_state.round_idx < len(df)-1:
            st.session_state.round_idx += 1   # avanzamos
            process_round_effects()           # aplicamos efectos con EL NUEVO PRECIO
            persist_session(sid)
            safe_rerun()

    if st.button("🏁 Finalizar y calcular", use_container_width=True, disabled=st.session_state.already_finalized):
        res=finalize_results()
        if res is not None:
            st.session_state["final_res"]=res
        st.session_state.round_idx = len(df)-1
        persist_session(sid)
        safe_rerun()

    if st.button("↩️ Deshacer última acción", use_container_width=True, disabled=st.session_state.already_finalized):
        undo_last(); persist_session(sid); safe_rerun()

# Catálogo + agregar
left, right = st.columns([1.15, 1])
with left:
    st.subheader("🧰 Soluciones disponibles")
    catalog = build_catalog(mkt, idx, last_idx())
    st.dataframe(catalog[["nombre"]], use_container_width=True, hide_index=True)

    st.markdown("#### 1) Elegí herramienta y variante")
    c1, c2 = st.columns(2)
    with c1:
        tool = st.selectbox(
            "Herramienta",
            options=catalog["tool_id"],
            format_func=lambda t: catalog.set_index("tool_id").loc[t, "nombre"],
            disabled=st.session_state.already_finalized,
        )
    with c2:
        variants = catalog.set_index("tool_id").loc[tool, "variants"]
        var_idx = st.selectbox(
            "Variante",
            options=list(range(len(variants))),
            format_func=lambda i: variants[i]["label"],
            disabled=st.session_state.already_finalized,
        )

    if st.button("✅ Seleccionar", use_container_width=True, disabled=st.session_state.already_finalized):
        select_tool_variant(tool, variants[var_idx])
        st.success("Herramienta seleccionada. Completá la cantidad y agregá.")

    st.markdown("#### 2) Completar y agregar a posición")
    with st.form("add_form", clear_on_submit=False):
        qty = st.number_input("Cantidad (tn)", min_value=1, max_value=1_000_000, value=100, step=50, disabled=st.session_state.already_finalized)

        det = ""
        if st.session_state.sel_tool and st.session_state.sel_variant:
            t = st.session_state.sel_tool
            v = st.session_state.sel_variant
            if t == "forward": det = f"Forward: {r1(v['price'])}"
            elif t in ("piso","ultra_piso"): det = f"Strike {r1(v['strike'])} / Prima {r1(v['prima'])}"
            elif t=="ultra_banda": det = f"Piso {r1(v['piso'])} / Techo {r1(v['techo'])} / Prima {r1(v['prima'])}"
            elif t=="cargill_plus": det = f"Techo {r1(v['techo'])} / Bonificación {r1(v['bonificacion'])}"
            elif t=="duplo": det = f"Disparador {r1(v['trigger'])} / Acumulación {r1(v['accum'])}"
        left_cap = capacity_left()
        st.info((det or "Elegí y apretá **Seleccionar** arriba para ver los campos.") + f"  | Capacidad libre: **{r1(left_cap)} tn**")

        submitted = st.form_submit_button("➕ Agregar", disabled=st.session_state.already_finalized)
        if submitted:
            add_decision(int(qty)); persist_session(sid); safe_rerun()

    st.markdown("#### 📝 Notificaciones")
    if st.session_state.logs:
        for m in reversed(st.session_state.logs[-12:]): st.write(m)
    else:
        st.caption("Historial de movimientos, cierres, etc.")

with right:
    st.subheader("📦 Tu posición")

    with st.expander("Forwards / fijados (acumulado)", expanded=True):
        fwd_df = forwards_view_df(include_only_forwards=False)
        st.dataframe(fwd_df, hide_index=True, use_container_width=True)

    # PISO
    if st.session_state.pisos:
        st.markdown("**PISO ASEGURADO**")
        p_df = pd.DataFrame([{
            "id":p["id"],"ronda suscripción":p["round_open"],
            "cantidad total":p["qty_total"],"cantidad remanente":p["qty_rem"],
            "piso":p["strike"],"prima":p["prima"],"cantidad fijada":p["fixed_qty"],
            "precio promedio (con precio)":p["fixed_avg"],"cantidad pendiente":max(0.0,p["qty_total"]-p["fixed_qty"])
        } for p in st.session_state.pisos])
        st.dataframe(df_round1(p_df.drop(columns=["id"])), hide_index=True, use_container_width=True)
        for p in st.session_state.pisos:
            with st.expander(f"Fijación anticipada — Piso Asegurado (piso {r1(p['strike'])})", expanded=False):
                col1,col2,col3 = st.columns([1.2,1,1.2])
                col1.write(f"Fijado: {r1(p['fixed_qty'])} tn a {r1(p['fixed_avg'])}")
                qmax = float(p["qty_rem"])
                qfix = col2.number_input("Cantidad a fijar ahora", min_value=0.0, max_value=qmax, value=min(50.0,qmax), step=1.0, key=f"pfix_{p['id']}")
                if col3.button("Fijar cantidad", key=f"pfbtn_{p['id']}", disabled=qmax<=0 or st.session_state.already_finalized):
                    df_now=st.session_state.rounds_df; rid_now=int(df_now.iloc[st.session_state.round_idx]["round_id"])
                    matba_now=float(df_now.iloc[st.session_state.round_idx]["matba_price"])
                    fix_price=max(matba_now,p["strike"])-p["prima"]
                    p["qty_rem"]=r1(p["qty_rem"]-qfix)
                    p["fixed_qty"],p["fixed_avg"]=weighted_avg(p["fixed_qty"],p["fixed_avg"],qfix,fix_price)
                    add_sold_round(rid_now, qfix)
                    log(f"✅ Piso Asegurado: fijaste {r1(qfix)} tn a {r1(fix_price)}."); persist_session(sid); safe_rerun()

    # UP
    if st.session_state.ultra_pisos:
        st.markdown("**ULTRA PISO (Piso Promediando)**")
        up_df = pd.DataFrame([{
            "id":up["id"],"ronda suscripción":up["round_open"],
            "cantidad total":up["qty_total"],"cantidad remanente":up["qty_rem"],
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
                if c3.button("Fijar TODO el pendiente al valor de hoy", key=f"upfix_{up['id']}", disabled=pendiente<=0 or st.session_state.already_finalized):
                    early_fix_ultra_piso_all_pending(up["id"]); persist_session(sid); safe_rerun()

    # UB
    if st.session_state.bandas:
        st.markdown("**ULTRABANDA**")
        ub_df = pd.DataFrame([{
            "ronda suscripción":b["round_open"],
            "cantidad total":b["qty_total"],"cantidad remanente":b["qty_rem"],
            "piso":b["piso"],"techo":b["techo"],"prima":b["prima"],
            "cantidad diaria":b["daily_qty"],"cantidad fijada":b["fixed_qty"],"precio promedio (acumulado)":b["fixed_avg"],
            "cantidad pendiente":max(0.0,b["qty_total"]-b["fixed_qty"])
        } for b in st.session_state.bandas])
        st.dataframe(df_round1(ub_df), hide_index=True, use_container_width=True)

    # C+
    if st.session_state.cplus:
        st.markdown("**Cargill Plus (Potenciador)**")
        c_df = pd.DataFrame([{
            "ronda suscripción":c["round_open"],
            "cantidad (forward 1:1 ya generado)":c["qty"], "techo":c["techo"], "bonificación":c["bonificacion"],
            "pendiente techo (tn)": (0 if c.get("closed") else c["qty"]),
            "estado": "Cerrado" if c.get("closed") else "Pendiente"
        } for c in st.session_state.cplus])
        st.dataframe(df_round1(c_df), hide_index=True, use_container_width=True)

    # Duplos
    if st.session_state.duplos:
        st.markdown("**Duplos (Acumuladores)**")
        dp_df = pd.DataFrame([{
            "ronda suscripción":d["round_open"],
            "cantidad total":d["qty_total"], "nivel disparador":d["trigger"], "nivel de acumulación":d["accum"],
            "cantidad diaria":d["daily_qty"], "cantidad con precio":d["fixed_qty"], "precio acumulado":d["fixed_avg"],
            "cantidad liberada":d["released_qty"]
        } for d in st.session_state.duplos])
        st.dataframe(df_round1(dp_df), hide_index=True, use_container_width=True)

# ==================== KPIs / GRÁFICOS ====================
sold_eff, with_price, without_price, total_committed = committed_breakdown()
prod_pendiente = r1(max(0.0, float(st.session_state.total_volume) - float(total_committed)))

# Promedio acumulado ponderado (forwards + con precio)
mix_df = forwards_view_df(include_only_forwards=False)
mix_qty = float(mix_df["qty"].sum()) if not mix_df.empty else 0.0
mix_rev = float((mix_df["qty"]*mix_df["price"]).sum()) if not mix_df.empty else 0.0
prom_acum = r1(mix_rev/mix_qty) if mix_qty>0 else 0.0

st.markdown("### 📈 Vista general")
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Forwards vendidos (tn)", int(round(sold_eff)))
k2.metric("Comprometido con precio (tn)", int(round(with_price)))
k3.metric("Comprometido sin precio (tn)", int(round(without_price)))
k4.metric("Total comprometido (tn)", int(round(total_committed)))
k5.metric("Producción pendiente de definir decisión (tn)", int(round(prod_pendiente)))
st.caption(f"Precio promedio acumulado (ponderado) de lo que ya tiene precio: **{prom_acum} USD/tn**")

# Gráfico por ronda (estático con X=ronda)
plot_df = pd.DataFrame({"ronda": list(st.session_state.sold_by_round.keys()),
                        "ventas_tn": list(st.session_state.sold_by_round.values())}).sort_values("ronda")
if not plot_df.empty:
    fig = px.bar(plot_df, x="ronda", y="ventas_tn", title="Volumen con precio por ronda (según decisiones)")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Aún no hay volumen con precio registrado por ronda.")

# Resultado final + comparativa
if "final_res" in st.session_state and st.session_state.already_finalized:
    st.markdown("---"); st.header("🏆 Resultado final")
    res=st.session_state["final_res"]
    st.subheader(f"Promedio final alcanzado (ponderado, forwards + con precio): **{res['avg_price_final']} USD/tn**")
    st.markdown("**Benchmarks (referencia MATBA)**")
    for k,v in res["benchmarks"].items(): st.write(f"- {k}: {v} USD/tn")

    comp_df=pd.DataFrame({
        "Estrategia":["Promedio alcanzado"]+list(res["benchmarks"].keys()),
        "Precio (USD/tn)":[res["avg_price_final"]]+list(res["benchmarks"].values())
    })
    fig2 = px.bar(comp_df, x="Estrategia", y="Precio (USD/tn)", title="Comparativa de cierre")
    fig2.update_traces(width=0.35)
    st.plotly_chart(fig2, use_container_width=True)


# Persistimos al final por las dudas
try:
    persist_session(sid)
except Exception:
    pass
