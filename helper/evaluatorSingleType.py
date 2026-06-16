import os, sys, re, json, time, pickle, random
from llama_cpp import Llama
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FILTER_TYPE = "negation"

DATA_DIR         = "./data"
TEST_SET_PATH    = os.path.join(DATA_DIR, "factkg_test.pickle")
KG_CONTEXTS_PATH = os.path.join(DATA_DIR, "kg_contexts.json")
MODEL_PATH       = "/home/filippo/Scrivania/llama-b9158/granite-4.1-3b-BF16.gguf"

PRINT_EXAMPLES_EVERY = 10000
SEED                 = 1

N_CTX        = 1024
N_GPU_LAYERS = 0
N_THREADS    = 4
MAX_TOKENS   = 256
TEMPERATURE  = 0.0

random.seed(SEED)

RST="\033[0m"; BLD="\033[1m"; DIM="\033[2m"
GRN="\033[92m"; RED="\033[91m"; YLW="\033[93m"
CYN="\033[96m"; BLU="\033[94m"; MGT="\033[95m"; WHT="\037[97m"

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

SYSTEM_PROMPT = "You are a contradiction finder. You have to deal with incomplete contexts.\n"

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
		"Explain the reasoning in a single sentence.\n"
		"End your response with EXACTLY: Verdict: TRUE or Verdict: FALSE.\n\n"
		"Reasoning:"
	)
	return [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user",   "content": instruction},
	], cleaned_kg_lines

_hdr(f"FACTKG EVALUATOR  ·  Filtro: {FILTER_TYPE.upper()}")

if not os.path.exists(KG_CONTEXTS_PATH) or not os.path.exists(TEST_SET_PATH):
	print(f"{RED}[✗] File necessari non trovati in {DATA_DIR}{RST}")
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

print(f"  {GRN}[✓]{RST} Istanze totali filtrate: {total}")

_sec("Inizializzazione modello")
t0 = time.time()
llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX,
			n_gpu_layers=N_GPU_LAYERS, n_threads=N_THREADS, verbose=False)
print(f"  {GRN}[✓]{RST} Pronto in {time.time()-t0:.1f}s")

y_true, y_pred       = [], []
verdict_count        = 0
text_count           = 0
unparseable_count    = 0
correct_count        = 0
fn_count             = 0
fp_count             = 0

_hdr("AVVIO VALUTAZIONE")
start_time = time.time()

for idx, item in enumerate(filtered_instances):
	claim    = item["claim"]
	true_val = item["label"]
	kg_lines = item["kg_lines"]

	y_true.append(true_val)
	messages, final_kg_lines = build_prompt(claim, kg_lines)
	
	out = llm.create_chat_completion(
		messages=messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE
	)
	raw = out['choices'][0]['message']['content']
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

	if (idx + 1) % 5 == 0 or (idx + 1) == total:
		elapsed = time.time() - start_time
		speed   = (idx + 1) / elapsed
		eta     = (total - idx - 1) / max(speed, 1e-9)
		acc_now = accuracy_score(y_true, y_pred)
		ok      = GRN + "✓" + RST if pred_val == true_val else RED + "✗" + RST
		bar     = _bar(idx + 1, total)
		print(
			f"\r{bar} {BLD}{idx+1:>4}/{total}{RST}  {ok}  "
			f"Acc {YLW}{acc_now:.3f}{RST}  {speed:.2f} it/s  "
			f"ETA {CYN}{_eta(eta)}{RST}  [{DIM}{src[:3]}{RST}]",
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
		print(f"  {BLD}Claim:{RST}       {WHT}{claim}{RST}")
		print(f"  {BLD}Label Vera:{RST}  {GRN}TRUE{RST}" if true_val == 1 else f"  {BLD}Label Vera:{RST}  {RED}FALSE{RST}")
		print(f"  {BLD}Risposta:{RST}")
		for line in resp_clean.splitlines():
			print(f"    {DIM}{line}{RST}")
		print(f"  {BLD}CONTI:{RST}       Giuste: {GRN}{correct_count}{RST}  |  FP: {YLW}{fp_count}{RST}  |  FN: {RED}{fn_count}{RST}\n")

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
_hdr("FINE", CYN)