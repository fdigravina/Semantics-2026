import os, sys, re, json, time, pickle
from llama_cpp import Llama, LlamaGrammar
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR         = "./data"
TEST_SET_PATH    = os.path.join(DATA_DIR, "factkg_test.pickle")
KG_CONTEXTS_PATH = os.path.join(DATA_DIR, "kg_contexts.json")
STATE_FILE       = "zeroshot_checkpoint.json"

MODEL_PATH       = "/home/filippo/Scrivania/models/Qwen3.5-9B-Q8_0.gguf"

RESET_STATE = True
PRINT_EXAMPLES_EVERY = 1000

N_CTX        = 256
N_GPU_LAYERS = 0
N_THREADS    = 4
MAX_TOKENS   = 3
TEMPERATURE  = 0.0

RST="\033[0m"; BLD="\033[1m"; DIM="\033[2m"
GRN="\033[92m"; RED="\033[91m"; YLW="\033[93m"
CYN="\033[96m"; BLU="\033[94m"; MGT="\033[95m";

GRAMMAR_STR = 'root ::= "TRUE" | "FALSE"'

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
	if "TRUE" in cleaned:
		return 1, cleaned
	if "FALSE" in cleaned:
		return 0, cleaned
	return -1, cleaned

def build_prompt(claim: str, kg_lines: list[str]) -> tuple[list[dict], list[str]]:
	cleaned_kg_lines = []
	for line in kg_lines:
		match = _KG_NORM_RE.match(line)
		if match:
			sub, prop, obj = match.groups()
			if "~" in prop:
				sub, prop, obj = obj, prop.replace('~', '').strip(), sub
			
			sub_clean = sub.replace("_", " ")
			obj_clean = obj.replace("_", " ")
			prop_clean = prop.replace("_", " ")
			
			cleaned_kg_lines.append(f"The {prop_clean} of {sub_clean} is {obj_clean}.")
		else:
			cleaned_kg_lines.append(line.replace("_", " "))

	if cleaned_kg_lines:
		kg_section = "Fact: " + " ".join(cleaned_kg_lines)
	else:
		kg_section = "No background facts available."

	instruction = (
		f"Based on the context and your internal knowledge, answer TRUE or FALSE.\n"
		f"Context:{kg_section}\n"
		f"Claim: {claim}\n"
		f"Verdict:"
	)
	
	return [
		{"role": "system", "content": "Based on the context, answer TRUE or FALSE."},
		{"role": "user",   "content": instruction},
	], cleaned_kg_lines

def save_checkpoint(filepath, current_idx, y_true, y_pred, correct, fn, fp, unparseable, elapsed):
	"""Salva lo stato corrente su file JSON prima di uscire."""
	state = {
		"last_index": current_idx,
		"y_true": y_true,
		"y_pred": y_pred,
		"correct_count": correct,
		"fn_count": fn,
		"fp_count": fp,
		"unparseable_count": unparseable,
		"elapsed_time": elapsed
	}
	with open(filepath, 'w', encoding='utf-8') as f:
		json.dump(state, f, indent=4)
	print(f"\n{YLW}[i] Stato salvato correttamente in: {filepath}{RST}")

def load_checkpoint(filepath):
	"""Carica lo stato precedente se esiste."""
	if os.path.exists(filepath):
		with open(filepath, 'r', encoding='utf-8') as f:
			return json.load(f)
	return None

_hdr("FACTKG EVALUATOR - FAST MODE (ALL)")

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
		filtered_instances.append(kg_data[str_idx])

total = len(filtered_instances)
if total == 0:
	print(f"{RED}[✗] Nessuna istanza trovata nel dataset.{RST}")
	sys.exit(0)

print(f"  {GRN}[✓]{RST} Istanze totali nel dataset: {total}")

start_idx = 0
y_true, y_pred = [], []
correct_count = 0
fn_count = 0
fp_count = 0
unparseable_count = 0
previous_elapsed = 0.0

checkpoint = load_checkpoint(STATE_FILE)

if checkpoint and not RESET_STATE:
	start_idx = checkpoint["last_index"] + 1
	if start_idx >= total:
		print(f"{YLW}[i] Il checkpoint indica che l'elaborazione era già terminata. Forzo il reset.{RST}")
		start_idx = 0
	else:
		y_true = checkpoint["y_true"]
		y_pred = checkpoint["y_pred"]
		correct_count = checkpoint["correct_count"]
		fn_count = checkpoint["fn_count"]
		fp_count = checkpoint["fp_count"]
		unparseable_count = checkpoint["unparseable_count"]
		previous_elapsed = checkpoint["elapsed_time"]
		print(f"  {YLW}[→]{RST} Checkpoint trovato! Riprendo dall'indice {BLD}{start_idx}{RST} (Giuste finora: {correct_count})")
else:
	if RESET_STATE:
		print(f"  {YLW}[i]{RST} Flag RESET_STATE=True: ignoro eventuali salvataggi precedenti.")
	else:
		print(f"  {GRN}[i]{RST} Nessun checkpoint trovato. Parto da zero.")

_sec("Inizializzazione modello e grammatica")
t0 = time.time()
llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX,
			n_gpu_layers=N_GPU_LAYERS, n_threads=N_THREADS, verbose=False)
grammar = LlamaGrammar.from_string(GRAMMAR_STR)
print(f"  {GRN}[✓]{RST} Pronto in {time.time()-t0:.1f}s")

_hdr("AVVIO VALUTAZIONE")
print(f"{DIM}Premere CTRL+C in qualsiasi momento per stoppare e salvare i progressi.{RST}\n")

start_time = time.time()

try:
	for idx in range(start_idx, total):
		item = filtered_instances[idx]
		claim    = item["claim"]
		true_val = item["label"]
		kg_lines = item["kg_lines"]

		y_true.append(true_val)
		messages, final_kg_lines = build_prompt(claim, kg_lines)
		
		out = llm.create_chat_completion(
			messages=messages, 
			max_tokens=MAX_TOKENS, 
			temperature=TEMPERATURE,
			grammar=grammar
		)
		raw = out['choices'][0]['message']['content']
		pred_val, resp_clean = parse_response(raw)

		if pred_val == -1:
			unparseable_count += 1
			y_pred.append(true_val)
		else:
			y_pred.append(pred_val)
			if pred_val == true_val:
				correct_count += 1
			else:
				if true_val == 1 and pred_val == 0: fn_count += 1
				else:                               fp_count += 1

		elapsed = (time.time() - start_time) + previous_elapsed
		
		processed_this_session = idx - start_idx + 1
		speed   = processed_this_session / (time.time() - start_time) if (time.time() - start_time) > 0 else 1.0
		eta     = (total - idx - 1) / speed
		
		acc_now = accuracy_score(y_true, y_pred)
		ok      = GRN + "✓" + RST if pred_val == true_val else RED + "✗" + RST
		bar     = _bar(idx + 1, total)
		print(
			f"\r{bar} {BLD}{idx+1:>4}/{total}{RST}  {ok}  "
			f"Acc {YLW}{acc_now:.3f}{RST}  {speed:.2f} it/s  "
			f"ETA {CYN}{_eta(eta)}{RST}",
			end="", flush=True
		)

		if (idx + 1) % PRINT_EXAMPLES_EVERY == 0:
			print()
			_sec(f"LOG #{idx+1}", MGT)
			print(f"  {BLD}Claim:{RST}       {claim}")
			print(f"  {BLD}Contesto KG:{RST}")
			if final_kg_lines:
				for line in final_kg_lines:
					print(f"    {CYN}• {line}{RST}")
			else:
				print(f"    {YLW}Nessun dato KG disponibile{RST}")
			print(f"  {BLD}Label Vera:{RST}  {GRN}TRUE{RST}" if true_val == 1 else f"  {BLD}Label Vera:{RST}  {RED}FALSE{RST}")
			print(f"  {BLD}Risposta:{RST}    {resp_clean}")
			print(f"  {BLD}CONTI:{RST}       Giuste: {GRN}{correct_count}{RST}  |  FP: {YLW}{fp_count}{RST}  |  FN: {RED}{fn_count}{RST}\n")

except KeyboardInterrupt:
	print(f"\n\n{RED}[!] Interruzione rilevata! Salvataggio in corso...{RST}")
	total_elapsed_so_far = (time.time() - start_time) + previous_elapsed
	
	current_processed_index = len(y_true) - 1 + start_idx
	
	save_checkpoint(
		STATE_FILE, 
		current_idx=current_processed_index, 
		y_true=y_true, 
		y_pred=y_pred, 
		correct=correct_count, 
		fn=fn_count, 
		fp=fp_count, 
		unparseable=unparseable_count, 
		elapsed=total_elapsed_so_far
	)
	print(f"{RED}Esecuzione stoppata dall'utente. Puoi riprendere riavviando lo script con RESET_STATE = False.{RST}")
	sys.exit(0)

print()
elapsed_tot = (time.time() - start_time) + previous_elapsed
final_acc   = accuracy_score(y_true, y_pred)
cm          = confusion_matrix(y_true, y_pred)

_hdr("METRICHE FINALI", CYN)

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
print(f"    Tempo totale    : {elapsed_tot/60:.2f} min")
print(f"    Velocità media  : {total/elapsed_tot:.2f} it/s")

if os.path.exists(STATE_FILE):
	os.remove(STATE_FILE)

_hdr("FINE", CYN)