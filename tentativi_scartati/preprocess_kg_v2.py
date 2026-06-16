import os, sys, re, json, pickle, sqlite3, time, math
import functools
from collections import Counter
from itertools import combinations

import nltk
from nltk.corpus import wordnet as wn

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

MAX_KG_LINES  = 10
SINGLE_FETCH  = 5000
MIN_PRED_SCORE = 8.0

DATA_DIR      = "./data"
TEST_SET_PATH = os.path.join(DATA_DIR, "factkg_test.pickle")
DB_PATH       = os.path.join(DATA_DIR, "dbpedia_light.db")
OUT_PATH      = os.path.join(DATA_DIR, "kg_contexts.json")

R  = "\033[0m"; B  = "\033[1m"
G  = "\033[92m"; RE = "\033[91m"; Y  = "\033[93m"; C  = "\033[96m"

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


@functools.lru_cache(maxsize=65536)
def _entity_aliases(entity: str) -> tuple[str, ...]:
	entity = entity.replace("%27", "'")

	words = entity.split("_")
	aliases: list[str] = [entity]

	encoded = entity.replace("'", "%27")
	if encoded != entity:
		aliases.append(encoded)

	words_np = [w for w in words if not w.startswith("(")]
	if words_np != words:
		canon = "_".join(words_np)
		aliases.append(canon)
		aliases.append(canon.replace("'", "%27"))

	_ARTICLES = {"the", "a", "an", "of", "in", "at", "de", "di", "del", "della"}
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
			seen.add(a)
			result.append(a)
	return tuple(result)


_VB_STOP = frozenset({
	"is","are","was","were","be","been","being","the","a","an","in","of","to","and","or","not",
})

def _extract_verbs(claim: str) -> frozenset[str]:
	try:
		tokens = nltk.word_tokenize(claim)
		tagged = nltk.pos_tag(tokens)
		return frozenset(
			w.lower() for w, tag in tagged
			if (tag.startswith("VB") or tag.startswith("NN"))
			and w.lower() not in _VB_STOP
			and len(w) > 2
		)
	except Exception:
		return frozenset(t.lower() for t in TOKEN_RE.findall(claim) if len(t) > 3)

def precompute_verbs(instances: list) -> list[frozenset[str]]:
	print("Pre-calcolo verb/noun tokens (POS tagging)…", end=" ", flush=True)
	result = [_extract_verbs(claim) for claim, _ in instances]
	print(f"{G}OK{R}")
	return result


def precompute_idf(instances: list) -> dict[str, float]:
	doc_freq = Counter()
	total    = len(instances)
	for claim, _ in instances:
		for t in set(TOKEN_RE.findall(claim.lower())):
			doc_freq[t] += 1
	return {
		term: math.log((total + 1.0) / (freq + 1.0)) + 1.0
		for term, freq in doc_freq.items()
	}


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
	conn.execute("PRAGMA cache_size=-128000")   
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

def _by_subjects_IN(conn: sqlite3.Connection, aliases: tuple[str, ...], lim: int) -> list[tuple]:
	if not aliases:
		return []
	ph = ",".join("?" * len(aliases))
	rows = _exec(conn,
		f"SELECT subject,relation,object FROM triples WHERE subject IN ({ph}) LIMIT ?",
		aliases + (lim,))
	rows += _exec(conn,
		f"SELECT subject,relation,object FROM triples WHERE object IN ({ph}) LIMIT ?",
		aliases + (lim,))
	return rows

def _by_sub_obj(conn: sqlite3.Connection, s: str, o: str) -> list[tuple]:
	return _exec(conn,
		"SELECT subject,relation,object FROM triples WHERE subject=? AND object=?", (s, o))

def _relation_keyword_search(
	conn: sqlite3.Connection,
	aliases: tuple[str, ...],
	keywords: list[str],
	lim: int,
) -> list[tuple]:
	if not aliases or not keywords:
		return []
	ph  = ",".join("?" * len(aliases))
	rows: list[tuple] = []
	for kw in keywords[:6]:   
		kw_lc = kw.lower()
		if len(kw_lc) <= 2:
			continue
		rows += _exec(conn,
			f"SELECT subject,relation,object FROM triples "
			f"WHERE subject IN ({ph}) AND LOWER(relation) LIKE ? LIMIT ?",
			aliases + (f"%{kw_lc}%", lim))
		rows += _exec(conn,
			f"SELECT subject,relation,object FROM triples "
			f"WHERE object IN ({ph}) AND LOWER(relation) LIKE ? LIMIT ?",
			aliases + (f"%{kw_lc}%", lim))
	return rows


def _score_triple(
	s: str, r: str, o: str,
	pred_tokens:   frozenset[str],   
	entity_tokens: frozenset[str],   
	idf: dict[str, float],
	verb_tokens:   frozenset[str],
) -> float:
	r_clean = r.lower().replace("_", " ")
	r_toks = set(TOKEN_RE.findall(r_clean))
	o_toks = set(TOKEN_RE.findall(o.lower().replace("_", " ")))
	triple_toks = r_toks | o_toks

	score = 0.0
	matched = False

	for p_tok in pred_tokens:
		if p_tok in r_toks:
			score += 5.0 * idf.get(p_tok, 1.0)
			matched = True
		elif p_tok in o_toks:
			score += 1.5 * idf.get(p_tok, 1.0)
			matched = True

	for v_tok in verb_tokens:
		if v_tok in r_toks:
			score += 3.5 * idf.get(v_tok, 1.0)
			matched = True
		elif v_tok in o_toks:
			score += 1.0 * idf.get(v_tok, 1.0)
			matched = True

	if not matched:
		return 0.0

	o_str = o.strip()
	if o_str and not LITERAL_RE.match(o_str) and not o_str.startswith("http"):
		score *= 1.1

	return score


def score_and_rank_predicate_first(
	candidates:    list[tuple],
	pred_tokens:   frozenset[str],
	entity_tokens: frozenset[str],
	idf:           dict[str, float],
	verb_tokens:   frozenset[str],
) -> list[tuple[float, tuple]]:
	scored: list[tuple[float, tuple]] = []

	for t in candidates:
		sc = _score_triple(t[0], t[1], t[2], pred_tokens, entity_tokens, idf, verb_tokens)
		if sc > 0:
			scored.append((sc, t))

	if not scored:
		all_tokens = pred_tokens | entity_tokens | verb_tokens
		for t in candidates:
			triple_text = f"{t[1]} {t[2]}".lower().replace("_", " ")
			triple_toks = set(TOKEN_RE.findall(triple_text))
			if all_tokens.isdisjoint(triple_toks):
				continue
			common = len(all_tokens & triple_toks)
			if common:
				scored.append((float(common), t))

	scored.sort(key=lambda x: x[0], reverse=True)

	filtered_scored: list[tuple[float, tuple]] = []
	seen_predicates: set[str] = set()

	for score, triple in scored:
		predicate = triple[1]
		if predicate not in seen_predicates:
			seen_predicates.add(predicate)
			filtered_scored.append((score, triple))

	return filtered_scored


@functools.lru_cache(maxsize=4096)
def _expand_token_wordnet(token: str) -> frozenset[str]:
	results = {token}
	synsets = wn.synsets(token)
	if not synsets:
		return frozenset(results)

	allowed_domains = {"person", "kinship", "social", "state", "event", "time", "relation"}
	
	for syn in synsets:
		lex_name = syn.lexname().split(".")[1]
		if lex_name not in allowed_domains and syn.lexname().split(".")[0] != "noun":
			continue

		for lemma in syn.lemmas():
			name = lemma.name().lower()
			if "-" in name or "_" in name:
				parts = name.replace("-", "_").split("_")
				results.update(p for p in parts if p not in _STOPWORDS and len(p) > 2)
			else:
				if name not in _STOPWORDS and len(name) > 2:
					results.add(name)

			if lemma.antonyms():
				for ant in lemma.antonyms():
					ant_name = ant.name().lower()
					if "-" in ant_name or "_" in ant_name:
						parts = ant_name.replace("-", "_").split("_")
						results.update(p for p in parts if p not in _STOPWORDS and len(p) > 2)
					else:
						if ant_name not in _STOPWORDS and len(ant_name) > 2:
							results.add(ant_name)

	for syn in synsets[:2]:
		lex_name = syn.lexname().split(".")[1]
		if lex_name == "kinship" or lex_name == "person":
			for hyper in syn.hypernyms()[:2]:
				for lemma in hyper.lemmas()[:3]:
					n = lemma.name().lower()
					if "_" not in n and "-" not in n and len(n) > 2 and n not in _STOPWORDS:
						results.add(n)
						
	return frozenset(results)


def _predicate_tokens(
	claim_tokens:  frozenset[str],
	entity_tokens: frozenset[str],
) -> frozenset[str]:
	base_tokens = set(
		t for t in claim_tokens
		if t not in entity_tokens and t not in _STOPWORDS and len(t) > 2
	)
	
	final_tokens = set()
	for token in base_tokens:
		final_tokens.update(_expand_token_wordnet(token))
		
	return frozenset(final_tokens)


def retrieve(
	conn:          sqlite3.Connection,
	entities:      list[str],
	claim:         str,
	max_lines:     int,
	idf:           dict[str, float],
	verb_tokens:   frozenset[str],
) -> list[str]:

	std_ents = [e for e in entities if e and not LITERAL_RE.match(e)]
	if not std_ents:
		return []

	claim_tokens  = frozenset(TOKEN_RE.findall(claim.lower())) - _STOPWORDS
	entity_tokens = frozenset(
		tok for e in std_ents
		for tok in TOKEN_RE.findall(e.lower().replace("_", " "))
	)
	
	raw_pred_tokens = _predicate_tokens(claim_tokens, entity_tokens)
	
	expanded_verbs = set()
	for vt in verb_tokens:
		expanded_verbs.update(_expand_token_wordnet(vt))
	
	pred_tokens = frozenset(raw_pred_tokens)
	verb_tokens_expanded = frozenset(expanded_verbs)

	results: list[str] = []
	seen:    set[tuple] = set()

	if len(std_ents) >= 2:
		for a, b in combinations(std_ents, 2):
			if len(results) >= max_lines:
				break
			for a_al in _entity_aliases(a):
				for b_al in _entity_aliases(b):
					for sa, oa in ((a_al, b_al), (b_al, a_al)):
						for row in _by_sub_obj(conn, sa, oa):
							t = normalize_triple(*row)
							if t not in seen:
								seen.add(t)
								results.append(triple_to_str(*t))

	if len(results) >= max_lines:
		return results[:max_lines]

	all_candidates: list[tuple] = []
	for e in std_ents:
		aliases = _entity_aliases(e)
		raw     = _by_subjects_IN(conn, aliases, SINGLE_FETCH)
		for row in raw:
			t = normalize_triple(*row)
			if t not in seen:
				seen.add(t)
				all_candidates.append(t)

	scored = score_and_rank_predicate_first(
		all_candidates, pred_tokens, entity_tokens, idf, verb_tokens_expanded
	)

	best_score = scored[0][0] if scored else 0.0
	if best_score < MIN_PRED_SCORE:
		search_keywords = set()
		for tok in (pred_tokens | verb_tokens_expanded):
			if len(tok) > 3:
				search_keywords.add(tok)
					
		if search_keywords:
			kw_rows = []
			for e in std_ents[:3]:
				aliases = _entity_aliases(e)
				kw_rows += _relation_keyword_search(conn, aliases, list(search_keywords), 30)

			kw_cands: list[tuple] = []
			for row in kw_rows:
				t = normalize_triple(*row)
				if t not in seen:
					seen.add(t)
					kw_cands.append(t)

			if kw_cands:
				kw_scored = score_and_rank_predicate_first(
					kw_cands, pred_tokens, entity_tokens, idf, verb_tokens_expanded
				)
				merged = scored + [(sc * 0.9, t) for sc, t in kw_scored]
				merged.sort(key=lambda x: x[0], reverse=True)
				
				final_merged = []
				seen_merged_preds = set()
				for sc, t in merged:
					if t[1] not in seen_merged_preds:
						seen_merged_preds.add(t[1])
						final_merged.append((sc, t))
				scored = final_merged

	phase1_set = set(results)
	remaining  = max_lines - len(results)
	for _, t in scored:
		if remaining <= 0:
			break
		line = triple_to_str(*t)
		if line not in phase1_set:
			results.append(line)
			phase1_set.add(line)
			remaining -= 1

	return results[:max_lines]


def main():
	_ensure_nltk()

	for path, label in [(TEST_SET_PATH, "test set"), (DB_PATH, "KG DB")]:
		if not os.path.exists(path):
			sys.exit(f"File non trovato: {path}")
		print(f"{G}[✓]{R} {label}: {C}{path}{R}")

	print(f"\nCaricamento intero test set…", end=" ", flush=True)
	with open(TEST_SET_PATH, "rb") as f:
		test_set = pickle.load(f)
	instances = list(test_set.items())
	total     = len(instances)
	print(f"{G}{total} istanze trovate{R}")

	print("Pre-calcolo IDF…", end=" ", flush=True)
	idf_dict = precompute_idf(instances)
	print(f"{G}OK{R}")

	verb_tokens_list = precompute_verbs(instances)

	print(f"Connessione DB: {C}{DB_PATH}{R}…", end=" ", flush=True)
	conn = open_db(DB_PATH)
	print(f"{G}OK{R}\n")

	output: dict[str, dict] = {}
	t_start = time.time()

	for idx, (claim, meta) in enumerate(instances):
		current = idx + 1

		lr        = meta["Label"]
		label_val = 1 if (lr[0] if isinstance(lr, list) else lr) in [True, "True", 1] else 0

		raw_ents = meta.get("Entity_set", [])
		entities = [normalize_entity(e) for e in raw_ents if e]

		kg_lines = retrieve(
			conn, entities, claim, MAX_KG_LINES,
			idf_dict,
			verb_tokens_list[idx],
		)

		output[str(idx)] = {
			"claim"   : claim,
			"label"   : label_val,
			"entities": entities,
			"kg_lines": kg_lines,
		}

		if current % 10 == 0 or current == total:
			elapsed = time.time() - t_start
			speed   = current / max(elapsed, 1e-9)
			eta     = (total - current) / max(speed, 1e-9)
			bar_f   = int(30 * current / total)
			bar = f"[{'█'*bar_f}{'░'*(30-bar_f)}]"
			eta_s   = f"{eta:.0f}s" if eta < 60 else f"{int(eta)//60}m{int(eta)%60:02d}s"
			print(
				f"\r{bar} {B}{current:>5}/{total}{R}  "
				f"{speed:.0f}it/s  ETA {C}{eta_s}{R}  ",
				end="", flush=True,
			)

	elapsed = time.time() - t_start
	print(f"\n\n{G}[✓]{R} Retrieval completato in {Y}{elapsed:.1f}s{R}  "
		  f"({total/elapsed:.1f} it/s)")

	print(f"\nSalvataggio → {C}{OUT_PATH}{R}…", end=" ", flush=True)
	with open(OUT_PATH, "w", encoding="utf-8") as f:
		json.dump(output, f, ensure_ascii=False, indent=None, separators=(",", ":"))
	size_mb = os.path.getsize(OUT_PATH) / 1_048_576
	print(f"{G}OK{R}  ({size_mb:.2f} MB)")

	conn.close()


if __name__ == "__main__":
	main()