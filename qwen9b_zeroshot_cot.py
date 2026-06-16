import os, sys, re, json, time, pickle
from llama_cpp import Llama
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download

DATA_DIR         = "/kaggle/input/datasets/filippodg/factkg7/"
TEST_SET_PATH    = os.path.join(DATA_DIR, "factkg_test.pickle")
KG_CONTEXTS_PATH = os.path.join(DATA_DIR, "kg_contexts.json")
STATE_FILE       = "zeroshot_checkpoint.json"

MODEL_REPO       = "unsloth/Qwen3.5-9B-GGUF"
MODEL_FILE       = "Qwen3.5-9B-Q8_0.gguf"
MODEL_PATH       = os.path.join(os.getcwd(), MODEL_FILE)

RESET_STATE = True
PRINT_EXAMPLES_EVERY = 100

N_CTX        = 512
N_GPU_LAYERS = -1
N_THREADS    = 2
MAX_TOKENS   = 64
TEMPERATURE  = 0.0

RST="\033[0m"; BLD="\033[1m"; DIM="\033[2m"
GRN="\033[92m"; RED="\033[91m"; YLW="\033[93m"
CYN="\033[96m"; BLU="\033[94m"; MGT="\033[95m"

def _bar(v, tot, w=26):
    f = int(w * v / max(tot, 1))
    return f"[{'█'*f}{'░'*(w-f)}]"

def _hdr(txt, col=CYN):
    W=70; p=(W-len(txt)-2)//2
    print(f"\n{col}{BLD}{'═'*W}\n{'═'*p} {txt} {'═'*(W-p-len(txt)-2)}\n{'═'*W}{RST}")

def _sec(txt, col=BLU):
    print(f"\n{col}{BLD}{'─'*5} {txt} {'─'*(58-len(txt))}{RST}")

def _eta(s):
    return f"{s:.0f}s" if s < 60 else f"{int(s)//60}m {int(s)%60:02d}s"

_KG_NORM_RE = re.compile(r"subject:\s*(.*?)\s*\|\s*property:\s*(.*?)\s*\|\s*object:\s*(.*)")

def parse_response(text: str) -> tuple[int, str]:
    cleaned = text.strip().upper()
    if "TRUE" in cleaned and "FALSE" not in cleaned:
        return 1, text.strip()
    if "FALSE" in cleaned and "TRUE" not in cleaned:
        return 0, text.strip()
    words = re.findall(r'\b(TRUE|FALSE)\b', cleaned)
    if words:
        last_word = words[-1]
        if last_word == "TRUE":  return 1, text.strip()
        if last_word == "FALSE": return 0, text.strip()
    return -1, text.strip()

def build_prompt(claim: str, kg_lines: list[str]) -> tuple[list[dict], list[str]]:
    cleaned_kg_lines = []
    for line in kg_lines:
        match = _KG_NORM_RE.match(line)
        if match:
            sub, prop, obj = match.groups()
            if "~" in prop:
                sub, prop, obj = obj, prop.replace('~', '').strip(), sub
            cleaned_kg_lines.append(
                f"The {prop.replace('_', ' ')} of {sub.replace('_', ' ')} "
                f"is {obj.replace('_', ' ')}."
            )
        else:
            cleaned_kg_lines.append(line.replace("_", " "))

    kg_section = (
        "Fact: " + " ".join(cleaned_kg_lines)
        if cleaned_kg_lines
        else "No background facts available. Use your internal knowledge."
    )
    instruction = (
        f"Context: {kg_section}\n"
        f"Claim: {claim}\n\n"
        f"Task: Write exactly one single short sentence of logical reasoning "
        f"about the claim based on the context, and then conclude your response "
        f"with either TRUE or FALSE.\n"
        f"Answer:"
    )
    return (
        [
            {"role": "system", "content": "You are a factual verification assistant."},
            {"role": "user",   "content": instruction},
        ],
        cleaned_kg_lines,
    )

def save_checkpoint(filepath, current_idx, y_true, y_pred, correct, fn, fp,
                    unparseable, elapsed):
    state = {
        "last_index":        current_idx,
        "y_true":            y_true,
        "y_pred":            y_pred,
        "correct_count":     correct,
        "fn_count":          fn,
        "fp_count":          fp,
        "unparseable_count": unparseable,
        "elapsed_time":      elapsed,
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=4)
    print(f"\n{YLW}[i] Stato salvato: {filepath}{RST}")

def load_checkpoint(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

_hdr("CONTROL MODEL STATUS")
if not os.path.exists(MODEL_PATH):
    print(f"{YLW}[i] Download modello…{RST}")
    t_down = time.time()
    try:
        hf_hub_download(
            repo_id=MODEL_REPO, filename=MODEL_FILE,
            local_dir=os.getcwd(), local_dir_use_symlinks=False,
        )
        print(f"  {GRN}[✓]{RST} Scaricato in {time.time()-t_down:.1f}s!")
    except Exception as e:
        print(f"{RED}[✗] Errore: {e}{RST}"); sys.exit(1)
else:
    print(f"  {GRN}[✓]{RST} Modello presente: {MODEL_PATH}")

_hdr("FACTKG EVALUATOR")

if not os.path.exists(KG_CONTEXTS_PATH) or not os.path.exists(TEST_SET_PATH):
    print(f"{RED}[✗] File non trovati in {DATA_DIR}{RST}"); sys.exit(1)

with open(KG_CONTEXTS_PATH, encoding="utf-8") as f:
    kg_data: dict = json.load(f)
with open(TEST_SET_PATH, "rb") as f:
    test_set: dict = pickle.load(f)

filtered_instances = [
    kg_data[str(idx)]
    for idx, (claim, meta) in enumerate(test_set.items())
    if str(idx) in kg_data
]
total = len(filtered_instances)

if total == 0:
    print(f"{RED}[✗] Nessuna istanza.{RST}"); sys.exit(0)

print(f"  {GRN}[✓]{RST} Istanze totali: {total}")

(start_idx, y_true, y_pred, correct_count,
 fn_count, fp_count, unparseable_count, previous_elapsed) = (0, [], [], 0, 0, 0, 0, 0.0)

checkpoint = load_checkpoint(STATE_FILE)
if checkpoint and not RESET_STATE:
    start_idx = checkpoint["last_index"] + 1
    if start_idx >= total:
        start_idx = 0
    else:
        y_true            = checkpoint["y_true"]
        y_pred            = checkpoint["y_pred"]
        correct_count     = checkpoint["correct_count"]
        fn_count          = checkpoint["fn_count"]
        fp_count          = checkpoint["fp_count"]
        unparseable_count = checkpoint["unparseable_count"]
        previous_elapsed  = checkpoint["elapsed_time"]
        print(f"  {YLW}[→]{RST} Riprendo dall'indice {BLD}{start_idx}{RST}")

_sec("Inizializzazione modello su GPU")
t0 = time.time()
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
    n_threads=N_THREADS,
    flash_attn=True,
    verbose=False,
)
print(f"  {GRN}[✓]{RST} Pronto in {time.time()-t0:.1f}s")

_hdr("AVVIO VALUTAZIONE")
start_time = time.time()

try:
    for idx in range(start_idx, total):
        item = filtered_instances[idx]
        label = item["label"]
        y_true.append(label)

        messages, final_kg_lines = build_prompt(item["claim"], item["kg_lines"])
        out = llm.create_chat_completion(
            messages=messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE
        )
        raw = out['choices'][0]['message']['content']
        pred_val, resp_clean = parse_response(raw)

        if pred_val == -1:
            unparseable_count += 1
            y_pred.append(0)
        else:
            y_pred.append(pred_val)
            if pred_val == label:
                correct_count += 1
            else:
                if label == 1 and pred_val == 0: fn_count += 1
                else:                             fp_count += 1

        processed = idx - start_idx + 1
        elapsed   = time.time() - start_time
        speed     = processed / max(elapsed, 1e-9)
        print(
            f"\r{_bar(idx + 1, total)} {BLD}{idx+1:>4}/{total}{RST}  "
            f"{GRN+'✓'+RST if pred_val == label else RED+'✗'+RST}  "
            f"Acc {YLW}{accuracy_score(y_true, y_pred):.3f}{RST}  "
            f"{speed:.2f} it/s  ETA {CYN}{_eta((total - idx - 1) / speed)}{RST}",
            end="", flush=True,
        )

        if (idx + 1) % PRINT_EXAMPLES_EVERY == 0:
            print()
            _sec(f"LOG #{idx+1}", MGT)
            print(
                f"  {BLD}Claim:{RST}       {item['claim']}\n"
                f"  {BLD}Label Vera:{RST}  {'TRUE' if label == 1 else 'FALSE'}\n"
                f"  {BLD}Output:{RST}\n{DIM}{resp_clean}{RST}\n"
            )

except KeyboardInterrupt:
    print(f"\n\n{RED}[!] Interruzione. Salvataggio…{RST}")
    save_checkpoint(
        STATE_FILE, len(y_true) - 1 + start_idx,
        y_true, y_pred, correct_count, fn_count, fp_count,
        unparseable_count,
        (time.time() - start_time) + previous_elapsed,
    )
    sys.exit(0)

print()
elapsed_tot = (time.time() - start_time) + previous_elapsed
_hdr("METRICHE FINALI", CYN)
print(f"\n{BLD}  Classification Report:{RST}")
for line in classification_report(
    y_true, y_pred, target_names=["FALSE", "TRUE"], zero_division=0
).splitlines():
    print(f"    {line}")

print(f"\n{BLD}  Accuracy finale   : {YLW}{BLD}{accuracy_score(y_true, y_pred):.4f}{RST}")
print(f"{BLD}  Unparseable       : {YLW}{unparseable_count}{RST}")
print(f"{BLD}  Tempo totale      : {elapsed_tot/60:.2f} min{RST}")

if os.path.exists(STATE_FILE):
    os.remove(STATE_FILE)
_hdr("FINE", CYN)