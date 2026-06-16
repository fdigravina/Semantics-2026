import os
import sys
import re
import json
import time
import pickle
from groq import Groq
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

GROQ_API_KEY     = ""
RESET_CHECKPOINT = False

FILTER_TYPE      = "all"
DATA_DIR         = "./data"
TEST_SET_PATH    = os.path.join(DATA_DIR, "factkg_test.pickle")
KG_CONTEXTS_PATH = os.path.join(DATA_DIR, "kg_contexts.json")
CHECKPOINT_PATH  = os.path.join(DATA_DIR, "eval_checkpoint.json")

MODELS_CONFIG = [
	{"name": "llama-3.1-8b-instant", "rpm": 30, "tpm": 14400, "rpd": 6000, "tpd": 500000},
]

EXHAUSTED_DAILY_MODELS = set()

PRINT_EXAMPLES_EVERY = 100
MAX_TOKENS           = 512  
TEMPERATURE          = 0.0

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

_VERDICT_RE = re.compile(r'Verdict\s*:\s*(TRUE|FALSE)', re.I)
_TRUE_RE    = re.compile(r'\bTRUE\b')
_FALSE_RE   = re.compile(r'\bFALSE\b')
_KG_NORM_RE = re.compile(r"subject:\s*(.*?)\s*\|\s*property:\s*(.*?)\s*\|\s*object:\s*(.*)")

def parse_response(text: str) -> tuple[int, str, str]:
	m = _VERDICT_RE.search(text)
	if m:
		return (1 if m.group(1).upper() == 'TRUE' else 0), text.strip(), 'verdict'

	upper     = text.strip().upper()
	true_pos  = [m.start() for m in _TRUE_RE.finditer(upper)]
	false_pos = [m.start() for m in _FALSE_RE.finditer(upper)]
	if true_pos or false_pos:
		last_t = true_pos[-1]  if true_pos  else -1
		last_f = false_pos[-1] if false_pos else -1
		return (1 if last_t > last_f else 0), upper, 'text'

	return -1, f"[UNPARSEABLE] '{text}'", 'unparseable'

SYSTEM_PROMPT = "You are a fact checker.\n"

def build_prompt(claim: str, kg_lines: list[str]) -> tuple[list[dict], list[str]]:
	cleaned_kg_lines = []
	for line in kg_lines:
		match = _KG_NORM_RE.match(line)
		if match:
			sub, prop, obj = match.groups()
			if "~" in prop:
				cleaned_kg_lines.append(f"subject: {obj} | property: {prop.replace('~', '').strip()} | object: {sub}")
			else:
				cleaned_kg_lines.append(line)
		else:
			cleaned_kg_lines.append(line)

	if cleaned_kg_lines:
		kg_section = "KG evidence:\n" + "\n".join(cleaned_kg_lines) + "."
	else:
		kg_section = "No Knowledge Graph data available for this claim."

	instruction = (
		f"Context:\n{kg_section}\n\n"
		f"Claim: {claim}\n\n"
		"Instructions:\n"
		"Evaluate the claim based only on the KG context. Don't make assumptions.\n"
		"Write your reasoning in a few sentences.\n"
		"End your response with EXACTLY: Verdict: TRUE or Verdict: FALSE.\n\n"
		"Reasoning:"
	)
	return [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user",   "content": instruction},
	], cleaned_kg_lines

def call_groq_with_fallback(client: Groq, messages: list[dict]) -> tuple[str, str]:
	global EXHAUSTED_DAILY_MODELS
	for model in MODELS_CONFIG:
		model_name = model["name"]
		if model_name in EXHAUSTED_DAILY_MODELS:
			continue
			
		backoff_time = 2.0
		while True:
			try:
				chat_completion = client.chat.completions.create(
					messages=messages,
					model=model_name,
					temperature=TEMPERATURE,
					max_tokens=MAX_TOKENS,
				)
				return chat_completion.choices[0].message.content, model_name
			except Exception as e:
				err_msg = str(e)
				print(err_msg)
				if "429" in err_msg or "rate_limit" in err_msg:
					if "RPD" in err_msg or "TPD" in err_msg:
						EXHAUSTED_DAILY_MODELS.add(model_name)
						break 
					
					if "TPM" in err_msg:
						if any(x in model_name for x in ["120b", "70b", "20b"]):
							break 
					
					time.sleep(backoff_time)
					backoff_time *= 1.5
					continue
				else:
					break
	raise RuntimeError("Tutti i modelli Groq hanno esaurito i limiti giornalieri (RPD/TPD)!")


_hdr(f"FACTKG EVALUATOR (GROQ API)  ·  Filtro: {FILTER_TYPE.upper()}")

if not os.path.exists(KG_CONTEXTS_PATH) or not os.path.exists(TEST_SET_PATH):
	print(f"{RED}[✗] File necessari non trovati in {DATA_DIR}{RST}")
	sys.exit(1)

if GROQ_API_KEY == "IL_TUO_GROQ_API_KEY_QUI" or not GROQ_API_KEY:
	print(f"{RED}[✗] Inserisci una chiave API valida nella variabile GROQ_API_KEY all'inizio del file.{RST}")
	sys.exit(1)

with open(KG_CONTEXTS_PATH, encoding="utf-8") as f:
	kg_data: dict = json.load(f)

with open(TEST_SET_PATH, "rb") as f:
	test_set: dict = pickle.load(f)

test_instances = list(test_set.items())

filtered_instances = []
for idx, (claim, meta) in enumerate(test_instances):
	str_idx = str(idx)
	if str_idx in kg_data:
		types = meta.get("types", [])
		if FILTER_TYPE == "all" or FILTER_TYPE in types:
			filtered_instances.append(kg_data[str_idx])

total = len(filtered_instances)
if total == 0:
	print(f"{RED}[✗] Nessuna istanza per il filtro: '{FILTER_TYPE}'{RST}")
	sys.exit(0)

# ────────── GESTIONE CHECKPOINT ──────────
start_idx = 0
y_true, y_pred = [], []
verdict_count = 0
text_count = 0
unparseable_count = 0
correct_count = 0
fn_count = 0
fp_count = 0

if not RESET_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
	_sec("Caricamento Checkpoint Precedente")
	try:
		with open(CHECKPOINT_PATH, "r", encoding="utf-8") as ck_f:
			ckpt = json.load(amber_data := ck_f)
		start_idx         = ckpt["last_processed_idx"] + 1
		y_true            = ckpt["metrics"]["y_true"]
		y_pred            = ckpt["metrics"]["y_pred"]
		verdict_count     = ckpt["counters"]["verdict_count"]
		text_count        = ckpt["counters"]["text_count"]
		unparseable_count = ckpt["counters"]["unparseable_count"]
		correct_count     = ckpt["counters"]["correct_count"]
		fn_count          = ckpt["counters"]["fn_count"]
		fp_count          = ckpt["counters"]["fp_count"]
		print(f"  {GRN}[✓]{RST} Ripristinato checkpoint. Ripresa dal claim {BLD}#{start_idx+1}{RST} di {total}")
	except Exception as e:
		print(f"  {YLW}[!] Errore nel caricamento del checkpoint ({e}). Riapertura da zero.{RST}")
		start_idx = 0
else:
	if RESET_CHECKPOINT:
		print(f"  {YLW}[!] Flag RESET_CHECKPOINT attivo. Inizio nuova sessione pulita.{RST}")
	else:
		print(f"  {GRN}[✓]{RST} Nessun checkpoint trovato. Inizio da zero.")

print(f"  {GRN}[✓]{RST} Istanze rimaste da elaborare: {total - start_idx}")

_sec("Inizializzazione Client Groq")
t0 = time.time()
client = Groq(api_key=GROQ_API_KEY)
print(f"  {GRN}[✓]{RST} Pronto in {time.time()-t0:.1f}s")

_hdr("AVVIO VALUTAZIONE")
start_time = time.time()
MIN_REQUEST_INTERVAL = 2.0 

# Iteriamo solo sugli elementi non ancora completati
for idx in range(start_idx, total):
	req_start = time.time()
	item = filtered_instances[idx]
	
	claim    = item["claim"]
	true_val = item["label"]
	kg_lines = item["kg_lines"]

	y_true.append(true_val)
	messages, final_kg_lines = build_prompt(claim, kg_lines)
	
	raw, model_used = call_groq_with_fallback(client, messages)
	pred_val, resp_clean, src = parse_response(raw)

	if pred_val == -1:
		unparseable_count += 1
		y_pred.append(true_val)
	else:
		y_pred.append(pred_val)
		if src == 'verdict': verdict_count += 1
		else:                text_count    += 1

		if pred_val == true_val:
			correct_count += 1
		else:
			if true_val == 1 and pred_val == 0: fn_count += 1
			else:                               fp_count += 1

	# Salvataggio immediato sul file di Checkpoint ad ogni iterazione
	checkpoint_data = {
		"last_processed_idx": idx,
		"metrics": {
			"y_true": y_true,
			"y_pred": y_pred
		},
		"counters": {
			"verdict_count": verdict_count,
			"text_count": text_count,
			"unparseable_count": unparseable_count,
			"correct_count": correct_count,
			"fn_count": fn_count,
			"fp_count": fp_count
		}
	}
	with open(CHECKPOINT_PATH, "w", encoding="utf-8") as ck_f:
		json.dump(checkpoint_data, ck_f, ensure_ascii=False, indent=4)

	# Stampa l'avanzamento a schermo
	elapsed = time.time() - start_time
	speed   = (idx + 1 - start_idx) / max(elapsed, 1e-9)
	eta     = (total - idx - 1) / max(speed, 1e-9)
	acc_now = accuracy_score(y_true, y_pred)
	ok      = GRN + "✓" + RST if pred_val == true_val else RED + "✗" + RST
	bar     = _bar(idx + 1, total)
	mod_slug = model_used.split("/")[-1][:12]
	
	print(
		f"\r{bar} {BLD}{idx+1:>4}/{total}{RST}  {ok}  "
		f"Acc {YLW}{acc_now:.3f}{RST}  {speed:.2f} it/s  "
		f"ETA {CYN}{_eta(eta)}{RST}  [{DIM}{src[:3]}/{mod_slug}{RST}]",
		end="", flush=True
	)

	if (idx + 1) % PRINT_EXAMPLES_EVERY == 0:
		print()
		_sec(f"LOG #{idx+1}", MGT)
		print(f"  {BLD}Contesto KG:{RST}")
		if final_kg_lines:
			for line in final_kg_lines:
				print(f"    {CYN}• {line}{RST}")
		else:
			print(f"    {YLW}Nessun dato KG disponibile{RST}")
		print(f"  {BLD}Claim:{RST}       {claim}")
		print(f"  {BLD}Label Vera:{RST}  {GRN}TRUE{RST}" if true_val == 1 else f"  {BLD}Label Vera:{RST}  {RED}FALSE{RST}")
		print(f"  {BLD}Risposta ({model_used}):{RST}")
		for line in resp_clean.splitlines():
			print(f"    {DIM}{line}{RST}")
		print(f"  {BLD}CONTI:{RST}       Giuste: {GRN}{correct_count}{RST}  |  FP: {YLW}{fp_count}{RST}  |  FN: {RED}{fn_count}{RST}\n")

	req_elapsed = time.time() - req_start
	if req_elapsed < MIN_REQUEST_INTERVAL:
		time.sleep(MIN_REQUEST_INTERVAL - req_elapsed)

print()
elapsed_tot = time.time() - start_time
final_acc   = accuracy_score(y_true, y_pred)
cm          = confusion_matrix(y_true, y_pred)

_hdr("METRICHE FINALI", CYN)

print(f"\n{BLD}  Qualità parsing:{RST}")
print(f"    Formato 'Verdict: X' corretto : {GRN}{verdict_count}{RST} ({100*verdict_count/total:.0f}%)")
print(f"    Fallback testo                : {YLW}{text_count}{RST}")
print(f"    Unparseable                   : {RED}{unparseable_count}{RST} ({100*unparseable_count/total:.1f}%)")

if cm.size == 4:
	tn, fp_cm, fn_cm, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
	prec = tp / max(tp + fp_cm, 1)
	rec  = tp / max(tp + fn_cm, 1)
	f1   = 2 * prec * rec / max(prec + rec, 1e-9)
	print(f"\n{BLD}  Confusion matrix:{RST}")
	print(f"    {DIM}{'':>14} Pred FALSE   Pred TRUE{RST}")
	print(f"    {BLD}Real FALSE    {RST}{GRN}{tn:>9}{RST}  {RED}{fp_cm:>9}{RST}  {DIM}TN={tn} FP={fp_cm}{RST}")
	print(f"    {BLD}Real TRUE     {RST}{RED}{fn_cm:>9}{RST}  {GRN}{tp:>9}{RST}  {DIM}FN={fn_cm} TP={tp}{RST}")
	print(f"\n    Precision: {YLW}{prec:.4f}{RST}  |  "
		  f"Recall: {YLW}{rec:.4f}{RST}  |  "
		  f"F1: {YLW}{f1:.4f}{RST}")

print(f"\n{BLD}  Classification Report:{RST}")
for line in classification_report(
		y_true, y_pred, target_names=["FALSE", "TRUE"], zero_division=0
	).splitlines():
	print(f"    {line}")

print(f"\n{BLD}  ── Sommario ──────────────────────────────────────────────{RST}")
print(f"    Accuracy finale : {YLW}{BLD}{final_acc:.4f}{RST}  ({int(final_acc*total)}/{total})")
print(f"    Giuste totali   : {GRN}{correct_count}{RST}")
print(f"    False Negative  : {RED}{fn_count}{RST}")
print(f"    False Positive  : {YLW}{fp_count}{RST}")
print(f"    Unparseable     : {RED}{unparseable_count}{RST}")
print(f"    Tempo totale    : {elapsed_tot/60:.2f} min")
print(f"    Velocità media  : {total/elapsed_tot:.2f} it/s")

# Al termine con successo dell'intero dataset, rimuoviamo il file di checkpoint temporaneo
if os.path.exists(CHECKPOINT_PATH):
	os.remove(CHECKPOINT_PATH)

_hdr("FINE", CYN)