import os, sys, re, json, time, pickle, math, sqlite3, functools
from itertools import combinations
from collections import Counter
import nltk
from nltk.corpus import wordnet as wn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR            = "./data"
DECOMPOSED_PATH     = os.path.join(DATA_DIR, "decomposed_claims.json")
DB_PATH             = os.path.join(DATA_DIR, "dbpedia_light.db")
OUT_PATH            = os.path.join(DATA_DIR, "verify_results.json")
STATE_FILE          = "verify_checkpoint.json"

RESET_STATE         = False

MAX_KG_LINES        = 12
SINGLE_FETCH        = 4000
KW_FETCH            = 40

MIN_SUPPORT_SCORE   = 4.0
MIN_CONTRA_SCORE    = 4.0
INVERSE_PENALTY     = 1.5
PROP_SUPPORT_THRESHOLD = 1

LOG_EVERY           = 100

# ANSI colours
RST="\033[0m"; BLD="\033[1m"; DIM="\033[2m"
GRN="\033[92m"; RED="\033[91m"; YLW="\033[93m"
CYN="\033[96m"; BLU="\033[94m"; MGT="\033[95m"

def _hdr(txt, col=CYN):
	W=70; p=(W-len(txt)-2)//2
	print(f"\n{col}{BLD}{'═'*W}\n{'═'*p} {txt} {'═'*(W-p-len(txt)-2)}\n{'═'*W}{RST}")

def _sec(txt, col=BLU):
	print(f"\n{col}{BLD}{'─'*5} {txt} {'─'*(60-len(txt))}{RST}")

def _bar(v, tot, w=28):
	f = int(w * v / max(tot, 1))
	return f"[{'█'*f}{'░'*(w-f)}]"

def _eta(s):
	return f"{s:.0f}s" if s < 60 else f"{int(s)//60}m {int(s)%60:02d}s"

def _ensure_nltk():
	for pkg, test in [
		("averaged_perceptron_tagger_eng", lambda: nltk.pos_tag(["test"])),
		("punkt_tab",                      lambda: nltk.word_tokenize("test")),
		("wordnet",                        lambda: wn.synsets("test")),
	]:
		try:
			test()
		except LookupError:
			nltk.download(pkg, quiet=True)

LITERAL_RE = re.compile(r'^\d+$|^\d{4}-\d{2}-\d{2}$|^\d{4}$')
TOKEN_RE   = re.compile(r'\w+')

_STOPWORDS = frozenset({
	"the","a","an","of","in","at","de","di","del","della","is","are",
	"was","were","be","been","being","do","does","did",
	"to","and","or","not","that","this","it","its","by","for","with",
	"from","on","as","but","he","she","they","we","i","you","his","her",
	"their","our","my","well","actually","read","said","also","very",
	"just","than","then","when","who","which","where","what","how",
})

_VB_STOP = frozenset({
	"is","are","was","were","be","been","being","the","a","an",
	"in","of","to","and","or","not",
})

@functools.lru_cache(maxsize=65536)
def _entity_aliases(entity: str) -> tuple[str, ...]:
	entity  = entity.replace("%27", "'")
	words   = entity.split("_")
	aliases = [entity]

	encoded = entity.replace("'", "%27")
	if encoded != entity:
		aliases.append(encoded)

	words_np = [w for w in words if not w.startswith("(")]
	if words_np != words:
		canon = "_".join(words_np)
		aliases.extend([canon, canon.replace("'", "%27")])

	_ARTICLES = {"the","a","an","of","in","at","de","di","del","della"}
	core = [w for w in words_np if w.lower() not in _ARTICLES]
	if core and core != words_np:
		aliases.append("_".join(core))
		if len(core) == 2:
			aliases.append(f"{core[1]}_{core[0]}")

	no_us = entity.replace("_", " ")
	if no_us != entity:
		aliases.append(no_us)

	seen: set[str] = set()
	result: list[str] = []
	for a in aliases:
		if a and a not in seen:
			seen.add(a); result.append(a)
	return tuple(result)

def normalize_entity(e: str) -> str:
	return e.replace(" ", "_").replace('"', "").replace("'", "")

def normalize_triple(s, r, o) -> tuple[str, str, str]:
	if r.startswith("~"):
		return str(o), r[1:], str(s)
	return str(s), str(r), str(o)

def triple_to_str(s: str, r: str, o: str) -> str:
	return f"subject: {s} | property: {r} | object: {o}"

def open_db(path: str) -> sqlite3.Connection:
	conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
	conn.execute("PRAGMA cache_size=-131072")
	conn.execute("PRAGMA temp_store=MEMORY")
	conn.execute("PRAGMA read_uncommitted=TRUE")
	conn.execute("PRAGMA mmap_size=1073741824")
	conn.execute("PRAGMA journal_mode=OFF")
	return conn

def _exec(conn: sqlite3.Connection, sql: str, params) -> list:
	try:
		return conn.execute(sql, params).fetchall()
	except sqlite3.Error:
		return []

def _by_subjects_in(conn, aliases: tuple[str,...], lim: int) -> list[tuple]:
	if not aliases: return []
	ph = ",".join("?"*len(aliases))
	rows  = _exec(conn, f"SELECT subject,relation,object FROM triples WHERE subject IN ({ph}) LIMIT ?", aliases+(lim,))
	rows += _exec(conn, f"SELECT subject,relation,object FROM triples WHERE object  IN ({ph}) LIMIT ?", aliases+(lim,))
	return rows

def _by_sub_obj(conn, s: str, o: str) -> list[tuple]:
	return _exec(conn, "SELECT subject,relation,object FROM triples WHERE subject=? AND object=?", (s, o))

def _by_obj_sub(conn, o: str, s: str) -> list[tuple]:
	"""Inverse lookup: what if the relation is reversed?"""
	return _exec(conn, "SELECT subject,relation,object FROM triples WHERE subject=? AND object=?", (o, s))

def _relation_keyword_search(conn, aliases: tuple[str,...], keywords: list[str], lim: int) -> list[tuple]:
	if not aliases or not keywords: return []
	ph   = ",".join("?"*len(aliases))
	rows = []
	for kw in keywords[:8]:
		kw_lc = kw.lower()
		if len(kw_lc) <= 2: continue
		rows += _exec(conn,
			f"SELECT subject,relation,object FROM triples WHERE subject IN ({ph}) AND LOWER(relation) LIKE ? LIMIT ?",
			aliases+(f"%{kw_lc}%", lim))
		rows += _exec(conn,
			f"SELECT subject,relation,object FROM triples WHERE object  IN ({ph}) AND LOWER(relation) LIKE ? LIMIT ?",
			aliases+(f"%{kw_lc}%", lim))
	return rows

def build_idf(propositions_per_claim: list[list[str]]) -> dict[str, float]:
	doc_freq  = Counter()
	all_props = [p for props in propositions_per_claim for p in props]
	total     = len(all_props)
	for prop in all_props:
		for t in set(TOKEN_RE.findall(prop.lower())):
			doc_freq[t] += 1
	return {
		term: math.log((total + 1.0) / (freq + 1.0)) + 1.0
		for term, freq in doc_freq.items()
	}

def _extract_verbs(text: str) -> frozenset[str]:
	try:
		tokens = nltk.word_tokenize(text)
		tagged = nltk.pos_tag(tokens)
		return frozenset(
			w.lower() for w, tag in tagged
			if (tag.startswith("VB") or tag.startswith("NN"))
			and w.lower() not in _VB_STOP
			and len(w) > 2
		)
	except Exception:
		return frozenset(t.lower() for t in TOKEN_RE.findall(text) if len(t) > 3)

@functools.lru_cache(maxsize=8192)
def _expand_token(token: str) -> frozenset[str]:
	results = {token}
	synsets = wn.synsets(token)
	if not synsets:
		return frozenset(results)

	for syn in synsets:
		dom = syn.lexname().split(".")
		if dom[0] != "noun" and dom[-1] not in {
			"person","kinship","social","state","event","time","relation","cognition","communication"
		}:
			continue
		for lemma in syn.lemmas():
			name = lemma.name().lower()
			parts = re.split(r'[-_]', name)
			results.update(p for p in parts if p not in _STOPWORDS and len(p) > 2)
			# antonyms → will be used to detect contradictions
			for ant in lemma.antonyms():
				aname = ant.name().lower()
				results.add(f"__ANT__{aname}")  # tagged as antonym
		# narrow hypernyms
		for hyper in syn.hypernyms()[:2]:
			for lem in hyper.lemmas()[:3]:
				n = lem.name().lower()
				if "_" not in n and len(n) > 2 and n not in _STOPWORDS:
					results.add(n)
	return frozenset(results)

def _expand_tokens(tokens: frozenset[str]) -> tuple[frozenset[str], frozenset[str]]:
	pos_tokens: set[str] = set()
	ant_tokens: set[str] = set()
	for t in tokens:
		for exp in _expand_token(t):
			if exp.startswith("__ANT__"):
				ant_tokens.add(exp[7:])
			else:
				pos_tokens.add(exp)
	return frozenset(pos_tokens), frozenset(ant_tokens)

def _score_triple(
	s: str, r: str, o: str,
	pred_tokens:   frozenset[str],
	entity_tokens: frozenset[str],
	idf:           dict[str, float],
	verb_tokens:   frozenset[str],
) -> float:
	r_toks = set(TOKEN_RE.findall(r.lower().replace("_", " ")))
	o_toks = set(TOKEN_RE.findall(o.lower().replace("_", " ")))
	score  = 0.0; matched = False

	for t in pred_tokens:
		if t in r_toks:
			score += 5.0 * idf.get(t, 1.0); matched = True
		elif t in o_toks:
			score += 1.5 * idf.get(t, 1.0); matched = True

	for t in verb_tokens:
		if t in r_toks:
			score += 3.5 * idf.get(t, 1.0); matched = True
		elif t in o_toks:
			score += 1.0 * idf.get(t, 1.0); matched = True

	if not matched:
		return 0.0
	if not LITERAL_RE.match(o.strip()) and not o.startswith("http"):
		score *= 1.1
	return score

def _score_inverse_triple(
	s: str, r: str, o: str,
	pred_tokens:   frozenset[str],
	ant_tokens:    frozenset[str],
	idf:           dict[str, float],
) -> float:
	r_toks = set(TOKEN_RE.findall(r.lower().replace("_", " ")))
	o_toks = set(TOKEN_RE.findall(o.lower().replace("_", " ")))
	score  = 0.0

	for t in ant_tokens:
		if t in r_toks:
			score += 6.0 * idf.get(t, 1.0)
		elif t in o_toks:
			score += 2.0 * idf.get(t, 1.0)

	for t in pred_tokens:
		if t in r_toks:
			score += 2.0 * idf.get(t, 1.0)

	return score * INVERSE_PENALTY

def retrieve_proposition(
	conn:          sqlite3.Connection,
	entities:      list[str],
	proposition:   str,
	max_lines:     int,
	idf:           dict[str, float],
) -> tuple[list[str], list[str]]:
	std_ents = [e for e in entities if e and not LITERAL_RE.match(e)]
	if not std_ents:
		return [], []

	claim_tokens  = frozenset(TOKEN_RE.findall(proposition.lower())) - _STOPWORDS
	entity_tokens = frozenset(
		tok for e in std_ents
		for tok in TOKEN_RE.findall(e.lower().replace("_", " "))
	)
	base_pred = frozenset(
		t for t in claim_tokens
		if t not in entity_tokens and t not in _STOPWORDS and len(t) > 2
	)
	verb_set  = _extract_verbs(proposition)

	pred_tokens_expanded, ant_tokens = _expand_tokens(base_pred)
	verb_expanded, _                 = _expand_tokens(verb_set)

	pred_tokens = frozenset(pred_tokens_expanded)
	verb_tokens = frozenset(verb_expanded)

	seen: set[tuple] = set()
	support_cands: list[tuple] = []
	contra_cands:  list[tuple] = []

	if len(std_ents) >= 2:
		for a, b in combinations(std_ents, 2):
			for a_al in _entity_aliases(a):
				for b_al in _entity_aliases(b):
					for sa, oa in ((a_al, b_al), (b_al, a_al)):
						for row in _by_sub_obj(conn, sa, oa):
							t = normalize_triple(*row)
							if t not in seen:
								seen.add(t); support_cands.append(t)
						for row in _by_obj_sub(conn, sa, oa):
							t = normalize_triple(*row)
							if t not in seen:
								seen.add(t); contra_cands.append(t)

	all_broad: list[tuple] = []
	for e in std_ents:
		aliases = _entity_aliases(e)
		for row in _by_subjects_in(conn, aliases, SINGLE_FETCH):
			t = normalize_triple(*row)
			if t not in seen:
				seen.add(t); all_broad.append(t)

	kw_set = {tok for tok in (pred_tokens | verb_tokens) if len(tok) > 3}
	kw_cands: list[tuple] = []
	for e in std_ents[:3]:
		for row in _relation_keyword_search(conn, _entity_aliases(e), list(kw_set), KW_FETCH):
			t = normalize_triple(*row)
			if t not in seen:
				seen.add(t); kw_cands.append(t)

	ant_cands: list[tuple] = []
	if ant_tokens:
		for e in std_ents[:3]:
			for row in _relation_keyword_search(conn, _entity_aliases(e), list(ant_tokens), KW_FETCH):
				t = normalize_triple(*row)
				if t not in seen:
					seen.add(t); ant_cands.append(t)

	def _top(cands, n=max_lines):
		scored = []
		seen_preds: set[str] = set()
		for t in cands:
			sc = _score_triple(t[0], t[1], t[2], pred_tokens, entity_tokens, idf, verb_tokens)
			if sc >= MIN_SUPPORT_SCORE and t[1] not in seen_preds:
				seen_preds.add(t[1])
				scored.append((sc, t))
		scored.sort(key=lambda x: x[0], reverse=True)
		return [triple_to_str(*t) for _, t in scored[:n]]

	def _top_contra(cands, n=max_lines):
		scored = []
		seen_preds: set[str] = set()
		for t in cands:
			sc = _score_inverse_triple(t[0], t[1], t[2], pred_tokens, ant_tokens, idf)
			if sc >= MIN_CONTRA_SCORE and t[1] not in seen_preds:
				seen_preds.add(t[1])
				scored.append((sc, t))
		scored.sort(key=lambda x: x[0], reverse=True)
		return [triple_to_str(*t) for _, t in scored[:n]]

	support_triples  = _top(support_cands + all_broad + kw_cands)
	contra_triples   = _top_contra(contra_cands + ant_cands)

	return support_triples[:max_lines], contra_triples[:max_lines//2]

def verify_proposition(
	support_triples: list[str],
	contra_triples:  list[str],
) -> int:
	n_support = len(support_triples)
	n_contra  = len(contra_triples)

	if n_support == 0 and n_contra == 0:
		return -1

	if n_contra > 0 and n_support == 0:
		return 0
	if n_contra >= n_support * 2:
		return 0
	if n_support >= PROP_SUPPORT_THRESHOLD:
		return 1
	return 0

def resolve_claim(
	prop_verdicts: list[int],
	connective:    str,
	negated:       bool,
) -> int:
	known   = [v for v in prop_verdicts if v != -1]
	unknown = len(prop_verdicts) - len(known)

	if not prop_verdicts:
		return 0

	if connective == "SINGLE" or len(prop_verdicts) == 1:
		verdict = prop_verdicts[0] if prop_verdicts[0] != -1 else 0

	elif connective == "AND":
		verdict = 1 if all(v == 1 for v in prop_verdicts) else 0

	elif connective == "OR":
		verdict = 1 if any(v == 1 for v in prop_verdicts) or unknown > 0 else 0

	elif connective == "NOT":
		verdict = 1 if sum(1 for v in prop_verdicts if v == 1) > len(prop_verdicts) / 2 else 0
		negated = not negated

	elif connective == "IF-THEN":
		verdict = 1 if any(v == 1 for v in prop_verdicts) else 0

	else:
		verdict = 1 if sum(1 for v in known if v == 1) > len(known) / 2 else 0

	return (1 - verdict) if negated else verdict

def save_checkpoint(filepath, last_idx, y_true, y_pred, results, elapsed):
	with open(filepath, 'w', encoding='utf-8') as f:
		json.dump({
			"last_index": last_idx,
			"y_true":     y_true,
			"y_pred":     y_pred,
			"results":    results,
			"elapsed":    elapsed,
		}, f)

def load_checkpoint(filepath):
	if os.path.exists(filepath):
		with open(filepath, 'r', encoding='utf-8') as f:
			return json.load(f)
	return None

def main():
	_ensure_nltk()
	
	for path, label in [(DECOMPOSED_PATH, "decomposed claims"), (DB_PATH, "KG database")]:
		if not os.path.exists(path):
			sys.exit(f"[✗] File not found: {path}")
		print(f"  {GRN}[✓]{RST} {label}: {CYN}{path}{RST}")

	with open(DECOMPOSED_PATH, encoding="utf-8") as f:
		decomposed: dict = json.load(f)

	instances = sorted(decomposed.items(), key=lambda x: int(x[0]))
	total     = len(instances)
	print(f"  {GRN}[✓]{RST} Claims loaded: {BLD}{total}{RST}")

	print("  Building IDF over all atomic propositions…", end=" ", flush=True)
	all_prop_lists = [v["atomic_propositions"] for _, v in instances]
	idf            = build_idf(all_prop_lists)
	print(f"{GRN}OK{RST}")

	print(f"  Connecting to DB: {CYN}{DB_PATH}{RST}…", end=" ", flush=True)
	conn = open_db(DB_PATH)
	print(f"{GRN}OK{RST}")

	start_idx        = 0
	y_true:  list[int] = []
	y_pred:  list[int] = []
	results: dict      = {}
	previous_elapsed   = 0.0

	ckpt = load_checkpoint(STATE_FILE)
	if ckpt and not RESET_STATE:
		start_idx        = ckpt["last_index"] + 1
		y_true           = ckpt["y_true"]
		y_pred           = ckpt["y_pred"]
		results          = ckpt["results"]
		previous_elapsed = ckpt.get("elapsed", 0.0)
		if start_idx >= total:
			print(f"  {YLW}[i]{RST} Checkpoint complete – resetting.")
			start_idx = 0; y_true = []; y_pred = []; results = {}; previous_elapsed = 0.0
		else:
			print(f"  {YLW}[→]{RST} Resuming from index {BLD}{start_idx}{RST}")
	else:
		print(f"  {GRN}[i]{RST} Starting from scratch." if not RESET_STATE else
			  f"  {YLW}[i]{RST} RESET_STATE=True – ignoring checkpoint.")

	print(f"{DIM}Press CTRL+C to stop and save progress.{RST}\n")

	start_time = time.time()

	try:
		for list_idx in range(start_idx, total):
			str_idx, item = instances[list_idx]

			claim       = item["claim"]
			true_label  = item["label"]
			entities    = item.get("entities", [])
			props       = item["atomic_propositions"]
			connective  = item.get("connectives", "SINGLE")
			negated     = item.get("negated", False)

			prop_verdicts:        list[int]  = []
			prop_support_triples: list[list] = []
			prop_contra_triples:  list[list] = []

			for prop in props:
				prop_ents = [normalize_entity(e) for e in entities] if entities else []
				prop_cap  = [normalize_entity(w) for w in TOKEN_RE.findall(prop) if w[0].isupper() and len(w) > 2]
				all_ents  = list(dict.fromkeys(prop_ents + prop_cap))  # deduplicated

				sup, con = retrieve_proposition(conn, all_ents, prop, MAX_KG_LINES, idf)
				verd     = verify_proposition(sup, con)
				prop_verdicts.append(verd)
				prop_support_triples.append(sup)
				prop_contra_triples.append(con)

			final_pred = resolve_claim(prop_verdicts, connective, negated)

			y_true.append(true_label)
			y_pred.append(final_pred)

			results[str_idx] = {
				"claim":           claim,
				"true_label":      true_label,
				"predicted":       final_pred,
				"correct":         (final_pred == true_label),
				"connective":      connective,
				"negated":         negated,
				"propositions":    [
					{
						"text":       props[i],
						"verdict":    prop_verdicts[i],
						"supporting": prop_support_triples[i],
						"contra":     prop_contra_triples[i],
					}
					for i in range(len(props))
				],
			}

			elapsed = (time.time() - start_time) + previous_elapsed
			done    = list_idx - start_idx + 1
			speed   = done / max(time.time() - start_time, 1e-9)
			eta     = (total - list_idx - 1) / max(speed, 1e-9)
			acc     = accuracy_score(y_true, y_pred) if y_true else 0.0
			ok      = GRN + "✓" + RST if final_pred == true_label else RED + "✗" + RST
			bar     = _bar(list_idx + 1, total)
			print(
				f"\r{bar} {BLD}{list_idx+1:>5}/{total}{RST}  {ok}  "
				f"Acc {YLW}{acc:.3f}{RST}  {speed:.1f} it/s  "
				f"ETA {CYN}{_eta(eta)}{RST}   ",
				end="", flush=True,
			)

			if (list_idx + 1) % LOG_EVERY == 0:
				print()
				_sec(f"LOG #{list_idx+1}", MGT)
				print(f"  {BLD}Claim:{RST}       {claim}")
				print(f"  {BLD}Label:{RST}       {GRN}TRUE{RST}" if true_label else f"  {BLD}Label:{RST}       {RED}FALSE{RST}")
				print(f"  {BLD}Predicted:{RST}   {GRN}TRUE{RST}" if final_pred else f"  {BLD}Predicted:{RST}   {RED}FALSE{RST}")
				print(f"  {BLD}Connective:{RST}  {connective}  Negated: {negated}")
				for i, prop in enumerate(props):
					verd_str = f"{GRN}TRUE{RST}" if prop_verdicts[i] == 1 else (
							   f"{RED}FALSE{RST}" if prop_verdicts[i] == 0 else f"{YLW}?{RST}")
					print(f"  {BLD}Prop {i+1}:{RST} {prop}  → {verd_str}")
					for line in prop_support_triples[i][:2]:
						print(f"    {CYN}+ {line}{RST}")
					for line in prop_contra_triples[i][:1]:
						print(f"    {RED}- {line}{RST}")
				print()

			if (list_idx + 1) % 100 == 0:
				save_checkpoint(STATE_FILE, list_idx, y_true, y_pred, results,
								(time.time() - start_time) + previous_elapsed)

	except KeyboardInterrupt:
		print(f"\n\n{RED}[!] Interrupted – saving checkpoint…{RST}")
		save_checkpoint(STATE_FILE, list_idx,  # type: ignore[name-defined]
						y_true, y_pred, results,
						(time.time() - start_time) + previous_elapsed)
		print(f"{YLW}Resume by re-running with RESET_STATE = False.{RST}")
		_print_metrics(y_true, y_pred, total)
		sys.exit(0)

	elapsed_tot = (time.time() - start_time) + previous_elapsed
	print(f"\n\n{GRN}[✓]{RST} Verification complete in {YLW}{elapsed_tot:.1f}s{RST}  "
		  f"({total/elapsed_tot:.1f} it/s)")

	os.makedirs(DATA_DIR, exist_ok=True)
	print(f"\nSaving → {CYN}{OUT_PATH}{RST}…", end=" ", flush=True)
	with open(OUT_PATH, "w", encoding="utf-8") as f:
		json.dump(results, f, ensure_ascii=False, indent=None, separators=(",", ":"))
	print(f"{GRN}OK{RST}  ({os.path.getsize(OUT_PATH)/1_048_576:.2f} MB)")

	_print_metrics(y_true, y_pred, total)

	if os.path.exists(STATE_FILE):
		os.remove(STATE_FILE)

	conn.close()
	_hdr("DONE", GRN)

def _print_metrics(y_true: list[int], y_pred: list[int], total: int):
	if not y_true:
		print(f"{YLW}[i] No predictions to evaluate.{RST}")
		return

	final_acc = accuracy_score(y_true, y_pred)
	cm        = confusion_matrix(y_true, y_pred)

	_hdr("METRICS", CYN)

	if cm.size == 4:
		tn, fp_cm, fn_cm, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
		prec = tp / max(tp + fp_cm, 1)
		rec  = tp / max(tp + fn_cm, 1)
		f1   = 2 * prec * rec / max(prec + rec, 1e-9)

		print(f"\n{BLD}  Confusion Matrix:{RST}")
		print(f"    {DIM}{'':>14} Pred FALSE   Pred TRUE{RST}")
		print(f"    {BLD}Real FALSE    {RST}{GRN}{tn:>9}{RST}  {RED}{fp_cm:>9}{RST}  {DIM}TN={tn} FP={fp_cm}{RST}")
		print(f"    {BLD}Real TRUE     {RST}{RED}{fn_cm:>9}{RST}  {GRN}{tp:>9}{RST}  {DIM}FN={fn_cm} TP={tp}{RST}")
		print(f"\n    Precision : {YLW}{prec:.4f}{RST}")
		print(f"    Recall    : {YLW}{rec:.4f}{RST}")
		print(f"    F1-score  : {YLW}{f1:.4f}{RST}")

	print(f"\n{BLD}  Classification Report:{RST}")
	for line in classification_report(
		y_true, y_pred, target_names=["FALSE", "TRUE"], zero_division=0
	).splitlines():
		print(f"    {line}")

	correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
	print(f"\n{BLD}  ── Summary ────────────────────────────────────────────────{RST}")
	print(f"    Final Accuracy : {YLW}{BLD}{final_acc:.4f}{RST}  ({correct}/{len(y_true)} evaluated)")
	print(f"    Total claims   : {total}")


if __name__ == "__main__":
	main()