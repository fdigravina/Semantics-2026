import os
import pickle
import sys
import time
import random
import sqlite3
import re
import json
from itertools import combinations
from functools import lru_cache
from llama_cpp import Llama
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR      = "./data"
MODEL_PATH    = "/home/filippo/Scrivania/llama-b9158/gemma-4-E4B-it-IQ4_NL.gguf"
TEST_SET_PATH = os.path.join(DATA_DIR, "factkg_test.pickle")
DB_PATH       = os.path.join(DATA_DIR, "dbpedia_light.db")
ERROR_LOG_PATH = os.path.join(DATA_DIR, "errors.jsonl")

LIMIT_TEST           = 100
PRINT_EXAMPLES_EVERY = 1

N_CTX        = 4096  # aumentato per CoT sempre attivo
N_GPU_LAYERS = 0
N_THREADS    = 4

MAX_PATHS        = 30
MAX_ENTITY_PAIRS = 5
MAX_NEIGHBORHOOD = 15
MAX_2HOP         = 15
MAX_3HOP         = 5
DB_CACHE_SIZE    = 512

MAX_TOKENS       = 512
TEMPERATURE      = 0.0
TRUE_PRIOR       = 0.6

RST = "\033[0m"; BLD = "\033[1m"; DIM = "\033[2m"
GRN = "\033[92m"; RED = "\033[91m"; YLW = "\033[93m"
CYN = "\033[96m"; BLU = "\033[94m"; MGT = "\033[95m"; WHT = "\033[97m"

def _bar(v, tot, w=26):
	f = int(w * v / max(tot, 1))
	return f"[{'█'*f}{'░'*(w-f)}]"

def _hdr(txt, col=CYN):
	W = 70; p = (W - len(txt) - 2) // 2
	print(f"\n{col}{BLD}{'═'*W}\n{'═'*p} {txt} {'═'*(W-p-len(txt)-2)}\n{'═'*W}{RST}")

def _sec(txt, col=BLU):
	print(f"\n{col}{BLD}{'─'*5} {txt} {'─'*(58-len(txt))}{RST}")

def _eta(s):
	return f"{s:.0f}s" if s < 60 else f"{int(s)//60}m {int(s)%60:02d}s"

for path, lbl in [(TEST_SET_PATH, "test set"), (DB_PATH, "knowledge graph DB")]:
	if not os.path.exists(path):
		print(f"{RED}[✗] File non trovato: {path}{RST}"); sys.exit(1)
	print(f"  {GRN}[✓]{RST} {lbl}: {CYN}{path}{RST}")

with open(TEST_SET_PATH, 'rb') as f:
	test_set = pickle.load(f)

test_instances = list(test_set.items())
if LIMIT_TEST:
	test_instances = random.sample(test_instances, min(LIMIT_TEST, len(test_instances)))

total = len(test_instances)
print(f"\n{BLD}Istanze: {WHT}{total}{RST}")

with open(ERROR_LOG_PATH, 'w') as f:
	f.write("")
print(f"  {GRN}[✓]{RST} Error log: {CYN}{ERROR_LOG_PATH}{RST}")

_sec("Inizializzazione modello")
t0 = time.time()
llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS,
			n_threads=N_THREADS, verbose=False)
print(f"  {GRN}[✓]{RST} Pronto in {time.time()-t0:.1f}s")

DATE_RE    = re.compile(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{1,2}\s+\w+\s+\d{4}\b')
EXIST_RE   = re.compile(r'\b(is|are|was|were|exist|born|died|located|known)\b', re.I)
MULTI_RE   = re.compile(r'\b(and|both|also|as well|together|each|all)\b', re.I)
NUMBER_RE  = re.compile(r'\b\d+\b')

def classify_claim(claim: str, entities: list) -> str:
	c = claim.strip()
	n_ent = len(entities)
	if DATE_RE.search(c):
		return 'date'
	if n_ent >= 3 or MULTI_RE.search(c):
		return 'multi_hop'
	if NUMBER_RE.search(c) and n_ent <= 2:
		return 'numeric'
	if EXIST_RE.search(c) and n_ent <= 1:
		return 'existence'
	return 'single'

_DB_CONN = None

def _get_conn():
	global _DB_CONN
	if _DB_CONN is None:
		_DB_CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
		_DB_CONN.execute("PRAGMA cache_size = -8000")
		_DB_CONN.execute("PRAGMA temp_store = MEMORY")
		_DB_CONN.execute("PRAGMA journal_mode = WAL")
		_DB_CONN.execute("CREATE INDEX IF NOT EXISTS idx_sub_obj ON triples(subject, object)")
		_DB_CONN.execute("CREATE INDEX IF NOT EXISTS idx_obj_sub ON triples(object, subject)")
		_DB_CONN.execute("CREATE INDEX IF NOT EXISTS idx_sub ON triples(subject)")
		_DB_CONN.execute("CREATE INDEX IF NOT EXISTS idx_obj ON triples(object)")
		_DB_CONN.commit()
	return _DB_CONN

def _clean(e: str) -> str:
	return str(e).strip('"').strip("'").replace(" ", "_")

def _get_entities_degrees_batched(entities: list) -> dict:
	if not entities:
		return {}
	
	cur = _get_conn().cursor()
	placeholders = ",".join(["?"] * len(entities))
	
	out_counts = {e: 0 for e in entities}
	q_out = f"SELECT subject, COUNT(*) FROM triples WHERE subject IN ({placeholders}) GROUP BY subject"
	for r in cur.execute(q_out, entities).fetchall():
		out_counts[r[0]] = r[1]
		
	in_counts = {e: 0 for e in entities}
	q_in = f"SELECT object, COUNT(*) FROM triples WHERE object IN ({placeholders}) GROUP BY object"
	for r in cur.execute(q_in, entities).fetchall():
		in_counts[r[0]] = r[1]
		
	return {e: out_counts[e] + in_counts[e] for e in entities}

@lru_cache(maxsize=DB_CACHE_SIZE)
def _direct(e1: str, e2: str) -> tuple:
	cur = _get_conn().cursor()
	r1 = cur.execute(
		"SELECT relation FROM triples WHERE subject=? AND object=?", (e1, e2)
	).fetchall()
	r2 = cur.execute(
		"SELECT relation FROM triples WHERE subject=? AND object=?", (e2, e1)
	).fetchall()
	return tuple(f"{e1} -[{r[0]}]→ {e2}" for r in r1) + \
		   tuple(f"{e2} -[{r[0]}]→ {e1}" for r in r2)

@lru_cache(maxsize=DB_CACHE_SIZE)
def _hop2(e1: str, e2: str, lim: int) -> tuple:
	cur = _get_conn().cursor()
	rows = cur.execute(
		"""SELECT t1.relation, t1.object, t2.relation
		   FROM triples t1 JOIN triples t2 ON t1.object = t2.subject
		   WHERE t1.subject=? AND t2.object=? LIMIT ?""",
		(e1, e2, lim)
	).fetchall()
	return tuple(f"{e1} -[{r1}]→ {mid} -[{r2}]→ {e2}" for r1, mid, r2 in rows)

@lru_cache(maxsize=DB_CACHE_SIZE)
def _hop3(e1: str, e2: str, lim: int) -> tuple:
	cur = _get_conn().cursor()
	rows = cur.execute(
		"""SELECT t1.relation, t1.object, t2.relation, t2.object, t3.relation
		   FROM triples t1
		   JOIN triples t2 ON t1.object  = t2.subject
		   JOIN triples t3 ON t2.object  = t3.subject
		   WHERE t1.subject=? AND t3.object=? LIMIT ?""",
		(e1, e2, lim)
	).fetchall()
	return tuple(f"{e1} -[{r1}]→ {a} -[{r2}]→ {b} -[{r3}]→ {e2}"
				 for r1, a, r2, b, r3 in rows)

@lru_cache(maxsize=DB_CACHE_SIZE)
def _neighborhood(e: str, lim: int) -> tuple:
	cur = _get_conn().cursor()
	rows = cur.execute(
		"SELECT relation, object FROM triples WHERE subject=? LIMIT ?", (e, lim)
	).fetchall()
	return tuple(f"{e} -[{r}]→ {o}" for r, o in rows)

@lru_cache(maxsize=DB_CACHE_SIZE)
def _neighborhood_incoming(e: str, lim: int) -> tuple:
	cur = _get_conn().cursor()
	rows = cur.execute(
		"SELECT subject, relation FROM triples WHERE object=? LIMIT ?", (e, lim)
	).fetchall()
	return tuple(f"{s} -[{r}]→ {e}" for s, r in rows)

def get_graph_paths(entity_set: list, claim_type: str) -> tuple:
	entities = [_clean(e) for e in entity_set if e]
	if not entities:
		return "No entities found.", "empty"

	degrees = _get_entities_degrees_batched(entities)
	entities = sorted(entities, key=lambda e: degrees.get(e, 0))
	
	paths = []
	ent_pairs = list(combinations(entities[:MAX_ENTITY_PAIRS], 2))

	for e1, e2 in ent_pairs:
		if len(paths) >= MAX_PATHS:
			break
		paths.extend(_direct(e1, e2))
		if len(paths) < MAX_PATHS:
			paths.extend(_hop2(e1, e2, MAX_2HOP))
		if claim_type in ('multi_hop', 'date') and len(paths) < MAX_PATHS // 2:
			paths.extend(_hop3(e1, e2, MAX_3HOP))

	paths = list(dict.fromkeys(paths))

	if len(paths) < 4:
		nb_all = []
		for e in entities[:3]:
			nb_all.extend(_neighborhood(e, MAX_NEIGHBORHOOD))
			nb_all.extend(_neighborhood_incoming(e, 5))
		if nb_all:
			filtered_nb = [t for t in dict.fromkeys(nb_all) if "→" in t]
			paths = paths + filtered_nb
			ctx_type = "paths+neighborhood" if [p for p in paths if "→" in p] else "neighborhood"
		else:
			ctx_type = "empty" if not paths else "paths"
	else:
		ctx_type = "paths"

	if not paths:
		return "No KG data found.", "empty"
	
	result = "\n".join(paths[:MAX_PATHS])
	result = result.replace("~", "")
	return result, ctx_type

SYSTEM_PROMPT = (
	"You are a fact checker. You have access to Knowledge Graph paths as evidence.\n"
)

def build_prompt(claim: str, kg_ctx: str, ctx_type: str, claim_type: str) -> list:
	if ctx_type == "empty":
		kg_section = "No Knowledge Graph data available for this claim."
	elif "neighborhood" in ctx_type and "paths" not in ctx_type:
		kg_section = f"KG properties of the relevant entity:\n{kg_ctx}"
	else:
		kg_section = f"KG evidence paths:\n{kg_ctx}"

	instruction = (
		f"{kg_section}\n\n"
		f"Claim: {claim}\n\n"
		"Instructions:\n"
		"1. Write the sentence in simpler sentences with connectives, then evaluate them.\n"
		"Example: Thomas Jefferson is the French president but he is not French -> "
		"Decomposition: (Thomas Jefferson is the French president) AND NOT(Thomas Jefferson is French) -> "
		"FALSE AND NOT(FALSE) -> FALSE AND TRUE -> FALSE"
		"2. End your response with exactly: Verdict: TRUE or Verdict: FALSE\n\n"
		"Reasoning:"
	)

	messages = [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user",   "content": instruction},
	]
	return messages

_VERDICT_RE = re.compile(r'Verdict\s*:\s*(TRUE|FALSE)', re.I)
_TRUE_RE    = re.compile(r'\bTRUE\b')
_FALSE_RE   = re.compile(r'\bFALSE\b')

def parse_response(text: str) -> tuple:
	m = _VERDICT_RE.search(text)
	if m:
		val = 1 if m.group(1).upper() == 'TRUE' else 0
		return val, text.strip(), 'verdict'

	upper = text.strip().upper()
	true_pos  = [m.start() for m in _TRUE_RE.finditer(upper)]
	false_pos = [m.start() for m in _FALSE_RE.finditer(upper)]

	if true_pos or false_pos:
		last_true  = true_pos[-1]  if true_pos  else -1
		last_false = false_pos[-1] if false_pos else -1
		val = 1 if last_true > last_false else 0
		return val, upper, 'text'

	prior = 1 if TRUE_PRIOR >= 0.5 else 0
	return prior, f"[PRIOR→{'TRUE' if prior else 'FALSE'}] '{text}'", 'prior'

def log_error(idx: int, claim: str, entities: list, kg_ctx: str,
			  ctx_type: str, claim_type: str, raw_response: str,
			  pred_val: int, true_val: int, parse_source: str):
	record = {
		"idx":          idx + 1,
		"claim":        claim,
		"entities":     [str(e) for e in entities],
		"claim_type":   claim_type,
		"ctx_type":     ctx_type,
		"kg_paths":     kg_ctx.splitlines(),
		"raw_response": raw_response,
		"parse_source": parse_source,
		"pred":         "TRUE" if pred_val == 1 else "FALSE",
		"true":         "TRUE" if true_val == 1 else "FALSE",
		"error_type":   (
			"false_negative" if true_val == 1 and pred_val == 0
			else "false_positive"
		),
	}
	with open(ERROR_LOG_PATH, 'a') as f:
		f.write(json.dumps(record, ensure_ascii=False) + "\n")

y_true, y_pred  = [], []
prior_count     = 0
verdict_count   = 0   
ctx_counts      = {"paths": 0, "paths+neighborhood": 0, "neighborhood": 0, "empty": 0}
type_counts     = {"existence": 0, "date": 0, "multi_hop": 0, "numeric": 0, "single": 0}
type_correct    = {k: 0 for k in type_counts}
fn_count        = 0   
fp_count        = 0   

_hdr("AVVIO VALUTAZIONE")
start_time = time.time()

for idx, (claim, meta) in enumerate(test_instances):

	lr = meta['Label']
	true_val = 1 if (lr[0] if isinstance(lr, list) else lr) in [True, "True", 1] else 0
	y_true.append(true_val)

	entities  = meta.get('Entity_set', [])
	ctype     = classify_claim(claim, entities)
	type_counts[ctype] += 1

	kg_ctx, ctx_type = get_graph_paths(entities, ctype)
	ctx_counts[ctx_type] = ctx_counts.get(ctx_type, 0) + 1

	messages = build_prompt(claim, kg_ctx, ctx_type, ctype)

	out = llm.create_chat_completion(
		messages=messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE
	)
	raw = out['choices'][0]['message']['content']
	pred_val, resp_clean, src = parse_response(raw)
	y_pred.append(pred_val)

	if src == 'prior':    prior_count   += 1
	if src == 'verdict':  verdict_count += 1
	if pred_val == true_val:
		type_correct[ctype] += 1
	else:
		log_error(idx, claim, entities, kg_ctx, ctx_type, ctype,
				  raw, pred_val, true_val, src)
		if true_val == 1 and pred_val == 0:
			fn_count += 1
		else:
			fp_count += 1

	elapsed = time.time() - start_time
	speed   = (idx + 1) / elapsed
	eta     = (total - idx - 1) / speed
	acc_now = accuracy_score(y_true, y_pred)
	ok      = GRN + "✓" + RST if pred_val == true_val else RED + "✗" + RST
	bar     = _bar(idx + 1, total)
	src_tag = DIM + src[:3] + RST

	print(
		f"\r{bar} {BLD}{idx+1:>4}/{total}{RST}  {ok}  "
		f"Acc {YLW}{acc_now:.3f}{RST}  {speed:.2f} it/s  "
		f"ETA {CYN}{_eta(eta)}{RST}  [{src_tag}|{DIM}{ctx_type[:4]}{RST}|{DIM}{ctype[:3]}{RST}]",
		end="", flush=True
	)

	if (idx + 1) % PRINT_EXAMPLES_EVERY == 0:
		tv   = GRN + "TRUE"  + RST if true_val == 1 else RED + "FALSE" + RST
		pv   = GRN + "TRUE"  + RST if pred_val == 1 else RED + "FALSE" + RST
		ok_lbl = GRN + BLD + "[ CORRETTO  ]" + RST if pred_val == true_val \
			else RED + BLD + "[ SBAGLIATO ]" + RST

		print()
		_sec(f"LOG #{idx+1}  [{ctype}  |  {ctx_type}  |  src={src}]", MGT)
		print(f"  {BLD}Claim:{RST}   {WHT}{claim}{RST}")
		ent_str = ", ".join(str(e) for e in entities[:5]) if entities else "—"
		print(f"  {BLD}Entità (ordinate):{RST} {ent_str}")
		print(f"  {DIM}{'—'*62}{RST}")
		for line in kg_ctx.splitlines()[:6]:
			print(f"  {DIM}  {line}{RST}")
		extra = len(kg_ctx.splitlines()) - 6
		if extra > 0:
			print(f"  {DIM}  ... (+{extra} righe){RST}")
		print(f"  {DIM}{'—'*62}{RST}")
		print(f"  {BLD}Risposta:{RST}")
		for line in resp_clean.splitlines():
			print(f"    {DIM}{line}{RST}")
		print(f"  {BLD}Pred:{RST} {pv}   {BLD}Reale:{RST} {tv}   {ok_lbl}")
		print(f"  {BLD}Acc:{RST} {YLW}{acc_now:.4f}{RST}  |  "
			  f"prior:{prior_count}  verdict_cnt:{verdict_count}  "
			  f"FN:{RED}{fn_count}{RST}  FP:{YLW}{fp_count}{RST}  "
			  f"paths:{ctx_counts.get('paths',0)}  "
			  f"neigh:{ctx_counts.get('neighborhood',0)+ctx_counts.get('paths+neighborhood',0)}  "
			  f"empty:{ctx_counts.get('empty',0)}\n\n")

print()
elapsed_tot = time.time() - start_time
final_acc   = accuracy_score(y_true, y_pred)
cm          = confusion_matrix(y_true, y_pred)

_hdr("METRICHE FINALI", CYN)

print(f"\n{BLD}  Accuracy per tipo di claim:{RST}")
for k in type_counts:
	n   = type_counts[k]
	if n == 0: continue
	cor = type_correct[k]
	acc = cor / n
	bar = _bar(cor, n, 20)
	col = GRN if acc >= 0.75 else (YLW if acc >= 0.60 else RED)
	print(f"    {CYN}{k:<12}{RST} {bar} {col}{acc:.3f}{RST}  ({cor}/{n})")

print(f"\n{BLD}  Distribuzione contesti KG:{RST}")
for k, v in ctx_counts.items():
	if v == 0: continue
	bar = _bar(v, total, 20)
	print(f"    {CYN}{k:<22}{RST} {bar} {v:>4} ({100*v/total:5.1f}%)")

print(f"\n{BLD}  Qualità parsing:{RST}")
print(f"    Formato 'Verdict: X' corretto : {GRN}{verdict_count}{RST} ({100*verdict_count/total:.0f}%)")
print(f"    Fallback testo                : {YLW}{total-verdict_count-prior_count}{RST}")
print(f"    Fallback prior                : {RED}{prior_count}{RST} ({100*prior_count/total:.1f}%)")

if cm.size == 4:
	tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
	prec  = tp / max(tp + fp, 1)
	rec   = tp / max(tp + fn, 1)
	f1    = 2 * prec * rec / max(prec + rec, 1e-9)
	print(f"\n{BLD}  Confusion matrix:{RST}")
	print(f"    {DIM}{'':>14} Pred FALSE   Pred TRUE{RST}")
	print(f"    {BLD}Real FALSE    {RST}{GRN}{tn:>9}{RST}  {RED}{fp:>9}{RST}  {DIM}TN={tn} FP={fp}{RST}")
	print(f"    {BLD}Real TRUE     {RST}{RED}{fn:>9}{RST}  {GRN}{tp:>9}{RST}  {DIM}FN={fn} TP={tp}{RST}")
	print(f"\n    Precision TRUE: {YLW}{prec:.4f}{RST}  |  "
		  f"Recall TRUE: {YLW}{rec:.4f}{RST}  |  "
		  f"F1: {YLW}{f1:.4f}{RST}")

print(f"\n{BLD}  Classification Report:{RST}")
for line in classification_report(
		y_true, y_pred, target_names=["FALSE","TRUE"], zero_division=0
	).splitlines():
	print(f"    {line}")

print(f"\n{BLD}  ── Sommario ──────────────────────────────────────────────{RST}")
print(f"    Accuracy finale     : {YLW}{BLD}{final_acc:.4f}{RST}  "
	  f"({int(final_acc*total)}/{total})")
print(f"    False Negative      : {RED}{fn_count}{RST}  (TRUE predetto FALSE)")
print(f"    False Positive      : {YLW}{fp_count}{RST}  (FALSE predetto TRUE)")
print(f"    Tempo totale        : {elapsed_tot/60:.2f} min")
print(f"    Velocità media      : {total/elapsed_tot:.2f} it/s")
print(f"    Log errori salvato  : {CYN}{ERROR_LOG_PATH}{RST}")
_hdr("FINE", CYN)