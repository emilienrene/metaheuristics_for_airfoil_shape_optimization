"""
rAIFoil Optimization Dashboard
Run with: python3 dashboard.py
Then open: http://localhost:5050
"""

from flask import Flask, jsonify, render_template_string, request
import pandas as pd
import glob
import os
import subprocess
import signal
import numpy as np

app = Flask(__name__)
GA_DIR = os.path.dirname(os.path.abspath(__file__))  
BASE_DIR = os.path.dirname(GA_DIR)         

def get_latest_opt():
    dirs = sorted(glob.glob(os.path.join(GA_DIR,"optimization_*")),key=lambda x:int(x.split("_")[-1]))
    return dirs[-1] if dirs else None

def get_convergence(opt_dir):
    path = os.path.join(opt_dir,"convergence.csv")
    if not os.path.isfile(path): return []
    df = pd.read_csv(path)
    return df[["generation","best_fitness"]].to_dict(orient="records")

def get_best_airfoil_coords(opt_dir):
    path = os.path.join(opt_dir,"best_airfoil.dat")
    if not os.path.isfile(path): return None,None,None
    airfoil_path = open(path).read().strip()
    if not os.path.isfile(airfoil_path): return None,None,None
    coords = np.loadtxt(airfoil_path,skiprows=1)
    x,y = coords[:,0].tolist(),coords[:,1].tolist()
    return x,y,os.path.basename(airfoil_path)

def get_best_performance(opt_dir):
    eval_dirs = sorted(glob.glob(os.path.join(opt_dir,"evaluation_mach_*")),key=lambda x:float(x.split("_")[-1]))
    if not eval_dirs: return None
    combined_path = os.path.join(eval_dirs[-1],"DD_80_combined.csv")
    if not os.path.isfile(combined_path): return None
    df = pd.read_csv(combined_path)
    if df.empty: return None
    row = df.iloc[0]
    cl = row.get("CL",None); cd = row.get("CD",None)
    ld = round(float(cl)/float(cd),4) if cd and float(cd)!=0 else None
    return {"Airfoil":str(row.get("Airfoil_idx","—")),"AOA":round(float(row.get("AOA",0)),4),
            "CL":round(float(cl),4) if cl is not None else "—",
            "CD":round(float(cd),6) if cd is not None else "—","L/D":ld,
            "Cl_max":round(float(row.get("Cl_max",0)),4),"cost":round(float(row.get("cost",0)),6)}

def kill_optimizer():
    try:
        result = subprocess.run(["pgrep","-f","GA_Optimizer_NoConstraints"],capture_output=True,text=True)
        pids = result.stdout.strip().split()
        killed = []
        for pid in pids:
            try: os.kill(int(pid),signal.SIGTERM); killed.append(pid)
            except ProcessLookupError: pass
        return killed
    except Exception: return []

def is_optimizer_running():
    result = subprocess.run(["pgrep","-f","GA_Optimizer_NoConstraints"],capture_output=True,text=True)
    return result.returncode == 0

@app.route("/api/data")
def api_data():
    opt_dir = get_latest_opt()
    if not opt_dir: return jsonify({"error":"No optimization folder found"})
    x,y,name = get_best_airfoil_coords(opt_dir)
    return jsonify({"opt_dir":os.path.basename(opt_dir),
                    "gen_count":len(glob.glob(os.path.join(opt_dir,"generation_*"))),
                    "convergence":get_convergence(opt_dir),
                    "airfoil":{"x":x,"y":y,"name":name},
                    "performance":get_best_performance(opt_dir),
                    "running":is_optimizer_running()})

@app.route("/api/launch",methods=["POST"])
def api_launch():
    data = request.get_json()
    required = ["aoa_min","aoa_max","mach_cd","reynolds","cl_target",
                "mach_clmax","num_gen","popul_size","coeff_cd","coeff_clmax"]
    for k in required:
        if k not in data or str(data[k]).strip()=="":
            return jsonify({"ok":False,"error":f"Missing field: {k}"}),400
    lines = [str(data[k]) for k in required] + ["y"]
    input_path = os.path.join(GA_DIR,"input_questions.txt")
    with open(input_path,"w") as f: f.write("\n".join(lines)+"\n")
    log_path = os.path.join(GA_DIR,"log.txt")
    print(f"DEBUG GA_DIR: {GA_DIR}", flush=True)
    print(f"DEBUG script exists: {os.path.isfile(os.path.join(GA_DIR, 'GA_Optimizer_NoConstraints.sh'))}", flush=True)
    subprocess.Popen(f"./GA_Optimizer_NoConstraints.sh > {log_path} < {input_path} &", shell=True, cwd=GA_DIR)
    return jsonify({"ok":True})

@app.route("/api/stop",methods=["POST"])
def api_stop():
    killed = kill_optimizer()
    if killed: return jsonify({"ok":True,"killed":killed})
    return jsonify({"ok":False,"error":"No optimizer process found"})

CONFIG_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>rAIFoil — Configure</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#0b0e14;--panel:#12161f;--border:#1e2535;--accent:#4f9cf9;--accent2:#a79bfa;
  --green:#34d399;--red:#f87171;--text:#e2e8f0;--muted:#64748b;
  --mono:'DM Mono',monospace;--sans:'DM Sans',sans-serif;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;display:flex;flex-direction:column;}
header{display:flex;align-items:center;justify-content:space-between;padding:18px 32px;
  border-bottom:1px solid var(--border);background:var(--panel);position:sticky;top:0;z-index:10;}
.logo{font-family:var(--mono);font-size:1.1rem;letter-spacing:.12em;color:var(--accent);}
.logo span{color:var(--accent2);}
.badge{font-family:var(--mono);font-size:.72rem;padding:4px 10px;border-radius:4px;
  background:#1a2236;border:1px solid var(--border);color:var(--muted);letter-spacing:.05em;}
main{flex:1;display:flex;align-items:flex-start;justify-content:center;padding:40px 24px;}
.config-card{background:var(--panel);border:1px solid var(--border);border-radius:12px;
  width:100%;max-width:720px;padding:36px 40px;display:flex;flex-direction:column;gap:26px;}
.card-header{display:flex;flex-direction:column;gap:6px;border-bottom:1px solid var(--border);padding-bottom:20px;}
.card-title{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);}
.card-title strong{color:var(--accent);font-weight:500;}
.card-subtitle{font-size:.82rem;color:var(--muted);font-family:var(--mono);}
.section-label{font-family:var(--mono);font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:var(--accent2);}
.field-group{display:flex;flex-direction:column;gap:14px;}
.field-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.field{display:flex;flex-direction:column;gap:7px;}
label{font-family:var(--mono);font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
label .unit{color:#2a3a52;margin-left:4px;}
input[type=number]{background:#0e1320;border:1px solid var(--border);border-radius:6px;
  padding:10px 14px;font-family:var(--mono);font-size:.95rem;color:var(--text);
  outline:none;transition:border-color .18s;width:100%;-moz-appearance:textfield;}
input[type=number]::-webkit-outer-spin-button,input[type=number]::-webkit-inner-spin-button{-webkit-appearance:none;}
input:focus{border-color:var(--accent);box-shadow:0 0 0 2px rgba(79,156,249,.12);}
input.error{border-color:var(--red);}
/* cost function */
.cost-box{background:#0a0d14;border:1px solid var(--border);border-radius:8px;padding:20px 24px;display:flex;flex-direction:column;gap:12px;}
.cost-label{font-family:var(--mono);font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:var(--accent2);}
.cost-formula{display:flex;align-items:center;justify-content:center;padding:8px 0 4px;min-height:64px;}
.cf-frac{display:inline-flex;flex-direction:column;align-items:center;}
.cf-num,.cf-den{font-family:var(--mono);font-size:1.1rem;color:var(--text);padding:3px 8px;white-space:nowrap;}
.cf-line{width:100%;height:1.5px;background:var(--muted);margin:4px 0;}
.cf-exp{font-size:.7rem;vertical-align:super;color:var(--accent);font-family:var(--mono);font-weight:500;transition:color .2s;}
.cf-exp.ph{color:var(--muted);}
.cf-star{font-size:.68rem;vertical-align:super;color:var(--accent2);font-family:var(--mono);}
.cost-hint{font-family:var(--mono);font-size:.7rem;color:var(--muted);text-align:center;letter-spacing:.04em;transition:color .2s;}
/* summary */
.summary-box{background:#0e1320;border:1px solid var(--border);border-radius:8px;
  padding:16px 18px;font-family:var(--mono);font-size:.8rem;color:var(--muted);line-height:2;display:none;}
.summary-box.visible{display:block;}
.sum-row{display:flex;justify-content:space-between;}
.sum-val{color:var(--text);}.sum-arrow{color:var(--accent2);}
.divider{border:none;border-top:1px solid var(--border);}
.actions{display:flex;align-items:center;gap:14px;}
.btn-go{flex:1;background:var(--accent);color:#0b0e14;border:none;border-radius:7px;
  padding:13px 28px;font-family:var(--mono);font-size:.85rem;font-weight:500;
  letter-spacing:.1em;cursor:pointer;transition:background .18s,transform .1s;text-transform:uppercase;}
.btn-go:hover{background:#7ab8fb;}.btn-go:active{transform:scale(.98);}
.btn-go:disabled{background:var(--border);color:var(--muted);cursor:not-allowed;}
.btn-dash{background:transparent;color:var(--muted);border:1px solid var(--border);
  border-radius:7px;padding:13px 20px;font-family:var(--mono);font-size:.78rem;
  letter-spacing:.08em;cursor:pointer;text-decoration:none;transition:border-color .18s,color .18s;white-space:nowrap;}
.btn-dash:hover{border-color:var(--accent2);color:var(--accent2);}
.status-msg{font-family:var(--mono);font-size:.78rem;text-align:center;padding:10px;border-radius:6px;display:none;}
.status-msg.ok{display:block;color:var(--green);background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.2);}
.status-msg.err{display:block;color:var(--red);background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.2);}
.spinner{display:inline-block;width:12px;height:12px;border:2px solid rgba(11,14,20,.3);
  border-top-color:#0b0e14;border-radius:50%;animation:spin .7s linear infinite;margin-right:8px;vertical-align:middle;}
@keyframes spin{to{transform:rotate(360deg);}}
</style>
</head>
<body>
<header>
  <div class="logo">rAI<span>Foil</span> — Optimization Setup</div>
  <span class="badge">GA_Optimizer_NoConstraints</span>
</header>
<main><div class="config-card">
  <div class="card-header">
    <div class="card-title">Configure — <strong>Optimization Parameters</strong></div>
    <div class="card-subtitle">Parameters will be written to input_questions.txt and the optimizer launched in background.</div>
  </div>

  <div class="field-group">
    <div class="section-label">Angle of Attack</div>
    <div class="field-row">
      <div class="field"><label>AoA Min <span class="unit">(°)</span></label><input type="number" id="aoa_min" placeholder="-5" step="0.1"/></div>
      <div class="field"><label>AoA Max <span class="unit">(°)</span></label><input type="number" id="aoa_max" placeholder="15" step="0.1"/></div>
    </div>
  </div>

  <div class="field-group">
    <div class="section-label">Flow Conditions</div>
    <div class="field-row">
      <div class="field"><label>Mach — C<sub>d</sub> objective</label><input type="number" id="mach_cd" placeholder="0.3" step="0.01" min="0.01" max="0.99"/></div>
      <div class="field"><label>Mach — C<sub>L,max</sub> objective</label><input type="number" id="mach_clmax" placeholder="0.2" step="0.01" min="0.01" max="0.99"/></div>
    </div>
    <div class="field-row">
      <div class="field"><label>Reynolds Number</label><input type="number" id="reynolds" placeholder="3000000" step="1000"/></div>
      <div class="field"><label>Target C<sub>L</sub></label><input type="number" id="cl_target" placeholder="1.2" step="0.01"/></div>
    </div>
  </div>

  <div class="field-group">
    <div class="section-label">Genetic Algorithm</div>
    <div class="field-row">
      <div class="field"><label>Number of Generations</label><input type="number" id="num_gen" placeholder="300" step="1" min="1"/></div>
      <div class="field"><label>Population Size</label><input type="number" id="popul_size" placeholder="50" step="1" min="2"/></div>
    </div>
  </div>

  <div class="field-group">
    <div class="section-label">Cost Function Exponents</div>
    <div class="field-row">
      <div class="field"><label>C<sub>d</sub> Exponent</label><input type="number" id="coeff_cd" placeholder="1" step="0.1" min="0"/></div>
      <div class="field"><label>C<sub>L,max</sub> Exponent</label><input type="number" id="coeff_clmax" placeholder="1" step="0.1" min="0"/></div>
    </div>
    <div class="cost-box">
      <div class="cost-label">Objective Function — live preview</div>
      <div class="cost-formula">
        <div class="cf-frac">
          <div class="cf-num">(C<span class="cf-star">*</span><sub>L,max</sub>)<span class="cf-exp ph" id="expClmax">&#945;</span></div>
          <div class="cf-line"></div>
          <div class="cf-den">(C<span class="cf-star">*</span><sub>d</sub>)<span class="cf-exp ph" id="expCd">&#946;</span></div>
        </div>
      </div>
      <div class="cost-hint" id="costHint">Enter exponents above to preview</div>
    </div>
  </div>

  <hr class="divider"/>

  <div class="summary-box" id="summaryBox">
    <div class="sum-row"><span>AoA range</span><span class="sum-val"><span id="s_aoa_min">—</span> <span class="sum-arrow">→</span> <span id="s_aoa_max">—</span> °</span></div>
    <div class="sum-row"><span>Mach (C<sub>d</sub>)</span><span class="sum-val" id="s_mach_cd">—</span></div>
    <div class="sum-row"><span>Mach (C<sub>L,max</sub>)</span><span class="sum-val" id="s_mach_clmax">—</span></div>
    <div class="sum-row"><span>Reynolds</span><span class="sum-val" id="s_reynolds">—</span></div>
    <div class="sum-row"><span>Target C<sub>L</sub></span><span class="sum-val" id="s_cl_target">—</span></div>
    <div class="sum-row"><span>Generations</span><span class="sum-val" id="s_num_gen">—</span></div>
    <div class="sum-row"><span>Population</span><span class="sum-val" id="s_popul_size">—</span></div>
    <div class="sum-row"><span>Exponents (C<sub>L,max</sub> / C<sub>d</sub>)</span><span class="sum-val"><span id="s_coeff_clmax">—</span> / <span id="s_coeff_cd">—</span></span></div>
  </div>

  <div class="status-msg" id="statusMsg"></div>
  <div class="actions">
    <a class="btn-dash" href="/dashboard">&#8599; Dashboard</a>
    <button class="btn-go" id="btnGo" onclick="launch()">Launch Optimizer</button>
  </div>
</div></main>

<script>
const fields=["aoa_min","aoa_max","mach_cd","reynolds","cl_target","mach_clmax","num_gen","popul_size","coeff_cd","coeff_clmax"];
fields.forEach(id=>{
  document.getElementById(id).addEventListener("input",()=>{
    updateSummary();
    document.getElementById(id).classList.remove("error");
    document.getElementById("statusMsg").className="status-msg";
  });
});
["coeff_cd","coeff_clmax"].forEach(id=>document.getElementById(id).addEventListener("input",updateCost));

function updateCost(){
  const cd=document.getElementById("coeff_cd").value.trim();
  const cl=document.getElementById("coeff_clmax").value.trim();
  const eCd=document.getElementById("expCd"); const eCl=document.getElementById("expClmax");
  const hint=document.getElementById("costHint");
  eCd.textContent=cd||"\u03B2"; eCd.className=cd?"cf-exp":"cf-exp ph";
  eCl.textContent=cl||"\u03B1"; eCl.className=cl?"cf-exp":"cf-exp ph";
  if(cd&&cl){hint.textContent="Maximising (C\u2217\u2C7C,max)^"+cl+" / (C\u2217\u2091)^"+cd;hint.style.color="var(--accent2)";}
  else{hint.textContent="Enter exponents above to preview";hint.style.color="var(--muted)";}
}

function updateSummary(){
  const v={};let any=false;
  fields.forEach(id=>{const val=document.getElementById(id).value.trim();v[id]=val;if(val)any=true;});
  const box=document.getElementById("summaryBox");
  if(!any){box.classList.remove("visible");return;}
  box.classList.add("visible");
  document.getElementById("s_aoa_min").textContent=v.aoa_min||"—";
  document.getElementById("s_aoa_max").textContent=v.aoa_max||"—";
  document.getElementById("s_mach_cd").textContent=v.mach_cd||"—";
  document.getElementById("s_mach_clmax").textContent=v.mach_clmax||"—";
  document.getElementById("s_reynolds").textContent=v.reynolds?Number(v.reynolds).toLocaleString():"—";
  document.getElementById("s_cl_target").textContent=v.cl_target||"—";
  document.getElementById("s_num_gen").textContent=v.num_gen||"—";
  document.getElementById("s_popul_size").textContent=v.popul_size||"—";
  document.getElementById("s_coeff_clmax").textContent=v.coeff_clmax||"—";
  document.getElementById("s_coeff_cd").textContent=v.coeff_cd||"—";
}

async function launch(){
  const btn=document.getElementById("btnGo");const msg=document.getElementById("statusMsg");
  let valid=true;const payload={};
  fields.forEach(id=>{const el=document.getElementById(id);const v=el.value.trim();
    if(!v){el.classList.add("error");valid=false;}else payload[id]=v;});
  if(!valid){msg.className="status-msg err";msg.textContent="Please fill in all fields before launching.";return;}
  btn.disabled=true;btn.innerHTML='<span class="spinner"></span>Launching\u2026';msg.className="status-msg";
  try{
    const res=await fetch("/api/launch",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    const d=await res.json();
    if(d.ok){msg.className="status-msg ok";msg.textContent="\u2713 Optimizer launched \u2014 redirecting to dashboard\u2026";
      setTimeout(()=>window.location.href="/dashboard",2000);}
    else throw new Error(d.error||"Unknown error");
  }catch(e){msg.className="status-msg err";msg.textContent="\u2717 Launch failed: "+e.message;
    btn.disabled=false;btn.innerHTML="Launch Optimizer";}
}
</script>
</body></html>"""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>rAIFoil Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#0b0e14;--panel:#12161f;--border:#1e2535;--accent:#4f9cf9;--accent2:#a78bfa;
  --green:#34d399;--red:#f87171;--text:#e2e8f0;--muted:#64748b;
  --mono:'DM Mono',monospace;--sans:'DM Sans',sans-serif;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;}
header{display:flex;align-items:center;justify-content:space-between;padding:18px 32px;
  border-bottom:1px solid var(--border);background:var(--panel);position:sticky;top:0;z-index:10;}
.logo{font-family:var(--mono);font-size:1.1rem;letter-spacing:.12em;color:var(--accent);}
.logo span{color:var(--accent2);}
.header-right{display:flex;align-items:center;gap:12px;}
.badge{font-family:var(--mono);font-size:.72rem;padding:4px 10px;border-radius:4px;
  background:#1a2236;border:1px solid var(--border);color:var(--muted);letter-spacing:.05em;}
.badge.live{border-color:var(--green);color:var(--green);}
.badge.stopped{border-color:var(--red);color:var(--red);}
.btn-hdr{font-family:var(--mono);font-size:.72rem;padding:5px 13px;border-radius:4px;
  background:transparent;letter-spacing:.06em;cursor:pointer;text-decoration:none;transition:background .15s;border:1px solid;}
.btn-new{border-color:var(--accent2);color:var(--accent2);}
.btn-new:hover{background:rgba(167,139,250,.1);}
.btn-stop{border-color:var(--red);color:var(--red);}
.btn-stop:hover{background:rgba(248,113,113,.1);}
.btn-stop:disabled{border-color:var(--border);color:var(--muted);cursor:not-allowed;pointer-events:none;}
#pulse{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 1.4s infinite;}
#pulse.idle{background:var(--muted);animation:none;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.7);}}
main{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:24px 32px;max-width:1400px;margin:0 auto;}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:22px 24px;display:flex;flex-direction:column;gap:14px;}
.panel.full-width{grid-column:1/-1;}
.panel-title{font-family:var(--mono);font-size:.72rem;letter-spacing:.14em;color:var(--muted);
  text-transform:uppercase;border-bottom:1px solid var(--border);padding-bottom:10px;}
.panel-title strong{color:var(--accent);font-weight:500;}
.chart-wrap{position:relative;width:100%;height:260px;}
#airfoilWrap{height:200px;}
.perf-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:12px;}
.kpi{background:#0e1320;border:1px solid var(--border);border-radius:8px;padding:14px 16px;display:flex;flex-direction:column;gap:6px;}
.kpi-label{font-family:var(--mono);font-size:.65rem;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;}
.kpi-value{font-family:var(--mono);font-size:1.25rem;font-weight:500;color:var(--text);}
.kpi-value.highlight{color:var(--accent);}.kpi-value.green{color:var(--green);}
.empty{color:var(--muted);font-family:var(--mono);font-size:.8rem;text-align:center;padding:40px 0;}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:100;align-items:center;justify-content:center;}
.modal-overlay.visible{display:flex;}
.modal{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:32px 36px;max-width:380px;width:90%;display:flex;flex-direction:column;gap:20px;}
.modal-title{font-family:var(--mono);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--red);}
.modal-body{font-family:var(--mono);font-size:.83rem;color:var(--muted);line-height:1.7;}
.modal-actions{display:flex;gap:12px;}
.btn-confirm-stop{flex:1;background:var(--red);color:#0b0e14;border:none;border-radius:7px;padding:11px;
  font-family:var(--mono);font-size:.8rem;font-weight:500;letter-spacing:.08em;cursor:pointer;text-transform:uppercase;transition:background .15s;}
.btn-confirm-stop:hover{background:#fca5a5;}
.btn-cancel{flex:1;background:transparent;color:var(--muted);border:1px solid var(--border);border-radius:7px;padding:11px;
  font-family:var(--mono);font-size:.8rem;letter-spacing:.08em;cursor:pointer;transition:border-color .15s;}
.btn-cancel:hover{border-color:var(--muted);}
.toast{position:fixed;bottom:28px;right:28px;background:var(--panel);font-family:var(--mono);
  font-size:.8rem;padding:10px 18px;border-radius:7px;z-index:200;display:none;animation:fadeIn .2s ease;}
.toast.visible{display:block;}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px);}to{opacity:1;}}
</style>
</head>
<body>
<div class="modal-overlay" id="stopModal">
  <div class="modal">
    <div class="modal-title">&#9888; Stop Optimization</div>
    <div class="modal-body">This will send SIGTERM to the running optimizer process. The current generation will not be saved. Are you sure?</div>
    <div class="modal-actions">
      <button class="btn-cancel" onclick="closeModal()">Cancel</button>
      <button class="btn-confirm-stop" onclick="confirmStop()">Stop Run</button>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>

<header>
  <div class="logo">rAI<span>Foil</span> &#8212; Optimization Dashboard</div>
  <div class="header-right">
    <a class="btn-hdr btn-new" href="/">+ New Run</a>
    <button class="btn-hdr btn-stop" id="btnStop" onclick="openModal()" disabled>&#9724; Stop Run</button>
    <span class="badge" id="optLabel">&#8212;</span>
    <span class="badge" id="genLabel">GEN &#8212;</span>
    <span class="badge live" id="runBadge">LIVE</span>
    <div id="pulse"></div>
  </div>
</header>
<main>
  <div class="panel">
    <div class="panel-title">Convergence &#8212; <strong>Best Fitness / Generation</strong></div>
    <div class="chart-wrap"><canvas id="convChart"></canvas></div>
  </div>
  <div class="panel">
    <div class="panel-title">Best Airfoil Shape &#8212; <strong id="airfoilName">&#8212;</strong></div>
    <div class="chart-wrap" id="airfoilWrap"><canvas id="airfoilChart"></canvas></div>
  </div>
  <div class="panel full-width">
    <div class="panel-title">Top Airfoil Performance &#8212; <strong>at Cl-target AoA</strong></div>
    <div class="perf-grid" id="perfGrid"><div class="empty">Waiting for data&#8230;</div></div>
  </div>
</main>
<script>
const CD={responsive:true,maintainAspectRatio:false,animation:{duration:400},plugins:{legend:{display:false}}};
const convChart=new Chart(document.getElementById("convChart").getContext("2d"),{
  type:"line",data:{labels:[],datasets:[{data:[],borderColor:"#4f9cf9",backgroundColor:"rgba(79,156,249,0.08)",borderWidth:2,pointRadius:3,pointBackgroundColor:"#4f9cf9",tension:0.3,fill:true}]},
  options:{...CD,scales:{
    x:{ticks:{color:"#64748b",font:{family:"DM Mono",size:11}},grid:{color:"#1e2535"},title:{display:true,text:"Generation",color:"#64748b",font:{family:"DM Mono",size:11}}},
    y:{ticks:{color:"#64748b",font:{family:"DM Mono",size:11}},grid:{color:"#1e2535"},title:{display:true,text:"Best Fitness",color:"#64748b",font:{family:"DM Mono",size:11}}}}}
});
const airfoilChart=new Chart(document.getElementById("airfoilChart").getContext("2d"),{
  type:"scatter",data:{datasets:[{data:[],pointRadius:2.5,pointBackgroundColor:"#4f9cf9",pointBorderWidth:0}]},
  options:{...CD,scales:{
    x:{ticks:{color:"#64748b",font:{family:"DM Mono",size:11}},grid:{color:"#1e2535"},title:{display:true,text:"x/c",color:"#64748b",font:{family:"DM Mono",size:11}}},
    y:{ticks:{color:"#64748b",font:{family:"DM Mono",size:11}},grid:{color:"#1e2535"},min:-0.2,max:0.2,title:{display:true,text:"z/c",color:"#64748b",font:{family:"DM Mono",size:11}}}}}
});
function renderPerf(p){
  if(!p){document.getElementById("perfGrid").innerHTML='<div class="empty">No performance data yet.</div>';return;}
  const items=[{label:"Airfoil",value:"airfoil_"+p.Airfoil,cls:""},{label:"AoA (\u00b0)",value:p.AOA,cls:""},
    {label:"CL",value:p.CL,cls:"highlight"},{label:"CD",value:p.CD,cls:""},
    {label:"L / D",value:p["L/D"],cls:"green"},{label:"Cl max",value:p.Cl_max,cls:""},{label:"Cost",value:p.cost,cls:"highlight"}];
  document.getElementById("perfGrid").innerHTML=items.map(i=>`<div class="kpi"><span class="kpi-label">${i.label}</span><span class="kpi-value ${i.cls}">${i.value??"\u2014"}</span></div>`).join("");
}
function setRunning(r){
  document.getElementById("btnStop").disabled=!r;
  const b=document.getElementById("runBadge"),p=document.getElementById("pulse");
  if(r){b.textContent="LIVE";b.className="badge live";p.className="";}
  else{b.textContent="STOPPED";b.className="badge stopped";p.className="idle";}
}
async function refresh(){
  try{
    const d=await(await fetch("/api/data")).json();
    if(d.error)return;
    document.getElementById("optLabel").textContent=d.opt_dir;
    document.getElementById("genLabel").textContent="GEN "+d.gen_count;
    setRunning(d.running);
    if(d.convergence&&d.convergence.length){convChart.data.labels=d.convergence.map(r=>r.generation);convChart.data.datasets[0].data=d.convergence.map(r=>r.best_fitness);convChart.update();}
    if(d.airfoil&&d.airfoil.x){document.getElementById("airfoilName").textContent=d.airfoil.name||"\u2014";airfoilChart.data.datasets[0].data=d.airfoil.x.map((xi,i)=>({x:xi,y:d.airfoil.y[i]}));airfoilChart.update();}
    renderPerf(d.performance);
  }catch(e){console.warn("Fetch failed:",e);}
}
function openModal(){document.getElementById("stopModal").classList.add("visible");}
function closeModal(){document.getElementById("stopModal").classList.remove("visible");}
function showToast(msg,color){
  const t=document.getElementById("toast");t.textContent=msg;
  t.style.cssText=`border:1px solid ${color};color:${color};`;
  t.classList.add("visible");setTimeout(()=>t.classList.remove("visible"),4000);
}
async function confirmStop(){
  closeModal();
  try{
    const d=await(await fetch("/api/stop",{method:"POST"})).json();
    if(d.ok){showToast("\u2713 Optimizer stopped (PID "+d.killed.join(", ")+")","#34d399");setRunning(false);}
    else showToast("\u2717 "+(d.error||"Could not stop"),"#f87171");
  }catch(e){showToast("\u2717 Stop request failed","#f87171");}
}
refresh();setInterval(refresh,100);
</script>
</body></html>"""

@app.route("/")
def index(): return render_template_string(CONFIG_HTML)

@app.route("/dashboard")
def dashboard(): return render_template_string(DASHBOARD_HTML)

if __name__=="__main__":
    print("rAIFoil \u2192 http://localhost:5050")
    app.run(host="0.0.0.0",port=5050,debug=False)
