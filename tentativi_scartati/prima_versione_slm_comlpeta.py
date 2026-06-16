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
from collections import deque
from llama_cpp import Llama
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

random.seed(3)

DATA_DIR       = "./data"
MODEL_PATH     = "/home/filippo/Scrivania/llama-b9158/gemma-4-E4B-it-IQ4_NL.gguf"
TEST_SET_PATH  = os.path.join(DATA_DIR, "factkg_test.pickle")
DB_PATH        = os.path.join(DATA_DIR, "dbpedia_light.db")
ERROR_LOG_PATH = os.path.join(DATA_DIR, "errors.jsonl")

LIMIT_TEST           = 100   # None = tutto il dataset
PRINT_EXAMPLES_EVERY = 1

N_CTX        = 2048
N_GPU_LAYERS = 0
N_THREADS    = 4

MAX_PATHS_TO_LLM    = 30
MAX_ENTITY_PAIRS    = 6
BFS_MAX_DEPTH       = 4
BFS_NODE_BUDGET     = 10000
HUB_DEGREE_THRESH   = 20
DB_CACHE_SIZE       = 1024

MAX_TOKENS  = 256
TEMPERATURE = 0.0

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

DATE_RE   = re.compile(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{1,2}\s+\w+\s+\d{4}\b')
EXIST_RE  = re.compile(r'\b(is|are|was|were|exist|born|died|located|known)\b', re.I)
MULTI_RE  = re.compile(r'\b(and|both|also|as well|together|each|all)\b', re.I)
NUMBER_RE = re.compile(r'\b\d+\b')

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
		_DB_CONN.execute("PRAGMA cache_size = -16000")
		_DB_CONN.execute("PRAGMA temp_store = MEMORY")
		_DB_CONN.execute("PRAGMA journal_mode = WAL")
		_DB_CONN.execute("CREATE INDEX IF NOT EXISTS idx_sub ON triples(subject)")
		_DB_CONN.execute("CREATE INDEX IF NOT EXISTS idx_obj ON triples(object)")
		_DB_CONN.execute("CREATE INDEX IF NOT EXISTS idx_sub_obj ON triples(subject, object)")
		_DB_CONN.commit()
	return _DB_CONN

def _clean(e: str) -> str:
	return str(e).strip('"').strip("'").replace(" ", "_")

@lru_cache(maxsize=DB_CACHE_SIZE)
def _out_edges(e: str) -> tuple:
	cur = _get_conn().cursor()
	return tuple(cur.execute(
		"SELECT relation, object FROM triples WHERE subject=?", (e,)
	).fetchall())

@lru_cache(maxsize=DB_CACHE_SIZE)
def _in_edges(e: str) -> tuple:
	cur = _get_conn().cursor()
	return tuple(cur.execute(
		"SELECT subject, relation FROM triples WHERE object=?", (e,)
	).fetchall())

@lru_cache(maxsize=DB_CACHE_SIZE)
def _degree(e: str) -> int:
	cur = _get_conn().cursor()
	r = cur.execute(
		"SELECT COUNT(*) FROM triples WHERE subject=? OR object=?", (e, e)
	).fetchone()
	return r[0] if r else 0

def _get_degrees_batched(entities: list) -> dict:
	if not entities:
		return {}
	cur = _get_conn().cursor()
	ph  = ",".join(["?"] * len(entities))
	out = {e: 0 for e in entities}
	for row in cur.execute(
		f"SELECT subject, COUNT(*) FROM triples WHERE subject IN ({ph}) GROUP BY subject",
		entities
	).fetchall():
		out[row[0]] = row[1]
	for row in cur.execute(
		f"SELECT object, COUNT(*) FROM triples WHERE object IN ({ph}) GROUP BY object",
		entities
	).fetchall():
		out[row[0]] = out.get(row[0], 0) + row[1]
	return out

def bfs_paths(src: str, dst: str, max_depth: int) -> list[str]:
	if src == dst:
		return []

	found: list[str] = []
	
	queue: deque = deque()
	queue.append((src, "", frozenset({src})))
	nodes_expanded = 0

	while queue and nodes_expanded < BFS_NODE_BUDGET:
		node, path_so_far, visited = queue.popleft()
		depth = path_so_far.count("→")  # profondità = numero di archi già percorsi
		nodes_expanded += 1

		if depth >= max_depth:
			continue

		for rel, nbr in _out_edges(node):
			edge = f"{node} -[{rel}]→ {nbr}"
			new_path = edge if not path_so_far else f"{path_so_far} -[{rel}]→ {nbr}"

			if nbr == dst:
				full = edge if not path_so_far else f"{path_so_far} · {edge}"
				full = (path_so_far + f" -[{rel}]→ {nbr}") if path_so_far else edge
				found.append(full)
				if len(found) >= MAX_PATHS_TO_LLM:
					return found
				continue

			if nbr in visited or _degree(nbr) > HUB_DEGREE_THRESH:
				continue

			if depth + 1 < max_depth:
				queue.append((nbr, new_path, visited | {nbr}))

		for sub, rel in _in_edges(node):
			edge = f"{sub} -[{rel}]→ {node}"
			new_path = edge if not path_so_far else f"{path_so_far} ← [{rel}] {sub}"

			real_edge = f"{sub} -[{rel}]→ {node}"
			full_candidate = (path_so_far + " | " + real_edge) if path_so_far else real_edge

			if sub == dst:
				found.append(full_candidate)
				if len(found) >= MAX_PATHS_TO_LLM:
					return found
				continue

			if sub in visited or _degree(sub) > HUB_DEGREE_THRESH:
				continue

			if depth + 1 < max_depth:
				queue.append((sub, full_candidate, visited | {sub}))

	return found

def get_graph_paths(entity_set: list, claim_type: str) -> tuple[str, str]:
	entities = [_clean(e) for e in entity_set if e]
	if not entities:
		return "No entities found.", "empty"

	degrees  = _get_degrees_batched(entities)
	entities = sorted(entities, key=lambda e: degrees.get(e, 0))

	depth = BFS_MAX_DEPTH if claim_type in ('multi_hop', 'date') else 2

	all_paths: list[str] = []

	pairs = list(combinations(entities[:MAX_ENTITY_PAIRS], 2))
	for e1, e2 in pairs:
		if len(all_paths) >= MAX_PATHS_TO_LLM:
			break
		found = bfs_paths(e1, e2, depth)
		all_paths.extend(found)

	seen: dict = dict.fromkeys(all_paths)
	clean: list[str] = list(seen.keys())

	ctx_type = "paths"

	if len(clean) < 4:
		covered = {e for e in entities for p in clean if e in p}
		for e in entities:
			if e not in covered:
				nb_out = [f"{e} -[{r}]→ {o}" for r, o in _out_edges(e)]
				nb_in  = [f"{s} -[{r}]→ {e}" for s, r in _in_edges(e)]
				clean.extend(nb_out[:15])
				clean.extend(nb_in[:5])
		ctx_type = "neighborhood" if not all_paths else "paths+neighborhood"

	if not clean:
		return "No KG data found.", "empty"

	clean = [p for p in clean if "~" not in p]
	result = "\n".join(clean[:MAX_PATHS_TO_LLM])
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
		"1. Write your reasoning in 3/4 sentences.\n"
		"2. Be careful in correctly transposing negations. Absence of data can actually be an information.\n"
		"3. IMPORTANT: End your response with EXACTLY: Verdict: TRUE or Verdict: FALSE.\n\n"
		"Reasoning:"
	)

	return [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user",   "content": instruction},
	]

_VERDICT_RE = re.compile(r'Verdict\s*:\s*(TRUE|FALSE)', re.I)
_TRUE_RE    = re.compile(r'\bTRUE\b')
_FALSE_RE   = re.compile(r'\bFALSE\b')

def parse_response(text: str) -> tuple[int, str, str]:
	m = _VERDICT_RE.search(text)
	if m:
		val = 1 if m.group(1).upper() == 'TRUE' else 0
		return val, text.strip(), 'verdict'

	upper     = text.strip().upper()
	true_pos  = [m.start() for m in _TRUE_RE.finditer(upper)]
	false_pos = [m.start() for m in _FALSE_RE.finditer(upper)]

	if true_pos or false_pos:
		last_true  = true_pos[-1]  if true_pos  else -1
		last_false = false_pos[-1] if false_pos else -1
		val = 1 if last_true > last_false else 0
		return val, upper, 'text'

	return -1, f"[UNPARSEABLE] '{text}'", 'unparseable'

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
		"pred":         "TRUE" if pred_val == 1 else ("FALSE" if pred_val == 0 else "UNKNOWN"),
		"true":         "TRUE" if true_val == 1 else "FALSE",
		"error_type":   (
			"unparseable"    if pred_val == -1 else
			"false_negative" if true_val == 1 and pred_val == 0
			else "false_positive"
		),
	}
	with open(ERROR_LOG_PATH, 'a') as f:
		f.write(json.dumps(record, ensure_ascii=False) + "\n")

y_true, y_pred    = [], []
verdict_count     = 0
text_count        = 0
unparseable_count = 0
ctx_counts        = {"paths": 0, "paths+neighborhood": 0, "neighborhood": 0, "empty": 0}
type_counts       = {"existence": 0, "date": 0, "multi_hop": 0, "numeric": 0, "single": 0}
type_correct      = {k: 0 for k in type_counts}
fn_count          = 0
fp_count          = 0

_hdr("AVVIO VALUTAZIONE")
start_time = time.time()

for idx, (claim, meta) in enumerate(test_instances):

	lr       = meta['Label']
	true_val = 1 if (lr[0] if isinstance(lr, list) else lr) in [True, "True", 1] else 0
	y_true.append(true_val)

	entities = meta.get('Entity_set', [])
	ctype    = classify_claim(claim, entities)
	type_counts[ctype] += 1

	kg_ctx, ctx_type = get_graph_paths(entities, ctype)
	ctx_counts[ctx_type] = ctx_counts.get(ctx_type, 0) + 1

	messages = build_prompt(claim, kg_ctx, ctx_type, ctype)

	out = llm.create_chat_completion(
		messages=messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE
	)
	raw      = out['choices'][0]['message']['content']
	pred_val, resp_clean, src = parse_response(raw)

	if pred_val == -1:
		unparseable_count += 1
		log_error(idx, claim, entities, kg_ctx, ctx_type, ctype,
				  raw, pred_val, true_val, src)
		y_pred.append(true_val)
	else:
		y_pred.append(pred_val)
		if src == 'verdict': verdict_count += 1
		else:                text_count    += 1

		if pred_val == true_val:
			type_correct[ctype] += 1
		else:
			log_error(idx, claim, entities, kg_ctx, ctx_type, ctype,
					  raw, pred_val, true_val, src)
			if true_val == 1 and pred_val == 0: fn_count += 1
			else:                               fp_count += 1

	elapsed = time.time() - start_time
	speed   = (idx + 1) / elapsed
	eta     = (total - idx - 1) / max(speed, 1e-9)
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
		tv    = GRN + "TRUE"  + RST if true_val == 1 else RED + "FALSE" + RST
		pv    = GRN + "TRUE"  + RST if pred_val == 1 else (RED + "FALSE" + RST if pred_val == 0 else YLW + "???" + RST)
		ok_lbl = (GRN + BLD + "[ CORRETTO  ]" + RST if pred_val == true_val
				  else RED + BLD + "[ SBAGLIATO ]" + RST)

		print()
		_sec(f"LOG #{idx+1}  [{ctype}  |  {ctx_type}  |  src={src}]", MGT)
		print(f"  {BLD}Claim:{RST}   {WHT}{claim}{RST}")
		ent_str = ", ".join(str(e) for e in entities[:5]) if entities else "—"
		print(f"  {BLD}Entità (ordinate per grado):{RST} {ent_str}")
		print(f"  {DIM}{'—'*62}{RST}")
		for line in kg_ctx.splitlines()[:8]:
			print(f"  {DIM}  {line}{RST}")
		extra = len(kg_ctx.splitlines()) - 8
		if extra > 0:
			print(f"  {DIM}  ... (+{extra} righe){RST}")
		print(f"  {DIM}{'—'*62}{RST}")
		print(f"  {BLD}Risposta:{RST}")
		for line in resp_clean.splitlines():
			print(f"    {DIM}{line}{RST}")
		print(f"  {BLD}Pred:{RST} {pv}   {BLD}Reale:{RST} {tv}   {ok_lbl}")
		print(f"  {BLD}Acc:{RST} {YLW}{acc_now:.4f}{RST}  |  "
			  f"verdict:{verdict_count}  text:{text_count}  "
			  f"unparseable:{RED}{unparseable_count}{RST}  "
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
	n = type_counts[k]
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
	print(f"\n    Precision TRUE: {YLW}{prec:.4f}{RST}  |  "
		  f"Recall TRUE: {YLW}{rec:.4f}{RST}  |  "
		  f"F1: {YLW}{f1:.4f}{RST}")

print(f"\n{BLD}  Classification Report:{RST}")
for line in classification_report(
		y_true, y_pred, target_names=["FALSE", "TRUE"], zero_division=0
	).splitlines():
	print(f"    {line}")

print(f"\n{BLD}  ── Sommario ──────────────────────────────────────────────{RST}")
print(f"    Accuracy finale     : {YLW}{BLD}{final_acc:.4f}{RST}  "
	  f"({int(final_acc*total)}/{total})")
print(f"    False Negative      : {RED}{fn_count}{RST}  (TRUE predetto FALSE)")
print(f"    False Positive      : {YLW}{fp_count}{RST}  (FALSE predetto TRUE)")
print(f"    Unparseable         : {RED}{unparseable_count}{RST}")
print(f"    Tempo totale        : {elapsed_tot/60:.2f} min")
print(f"    Velocità media      : {total/elapsed_tot:.2f} it/s")
print(f"    Log errori salvato  : {CYN}{ERROR_LOG_PATH}{RST}")
_hdr("FINE", CYN)