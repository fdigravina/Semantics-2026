import os, sys, re, json, pickle, sqlite3, time
from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn

MAX_KG_LINES        = 30      # max triples kept per claim
DB_ROWS_PER_ENTITY  = 5000    # max rows fetched per entity from SQLite
HOP2_MAX_NODES      = 20      # max intermediate nodes for 2-hop expansion

DATA_DIR      = "./data"
TEST_SET_PATH = os.path.join(DATA_DIR, "factkg_test.pickle")
DB_PATH       = os.path.join(DATA_DIR, "dbpedia_light.db")
OUT_PATH      = os.path.join(DATA_DIR, "kg_contexts.json")

_LITERAL_RE  = re.compile(r'^\d+(\.\d+)?$|^\d{4}-\d{2}-\d{2}$|^\d{4}$')

_CLAIM_STOPWORDS = {
	"i", "a", "an", "the", "is", "was", "are", "were", "be", "been",
	"had", "has", "have", "did", "do", "does", "very", "bit", "of",
	"in", "at", "to", "yes", "no", "and", "or", "but", "actually",
	"believe", "think", "man", "woman", "person", "people", "thing",
	"some", "any", "all", "just", "that", "this", "with", "for",
}

def _ensure_nltk():
	checks = [
		("wordnet", lambda: wn.synsets("test")),
		("punkt_tab", lambda: nltk.word_tokenize("test")),
	]
	for pkg, test in checks:
		try:
			test()
		except LookupError:
			nltk.download(pkg, quiet=True)

def open_db(path: str) -> sqlite3.Connection:
	conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
	conn.execute("PRAGMA cache_size=-262144")
	conn.execute("PRAGMA temp_store=MEMORY")
	conn.execute("PRAGMA mmap_size=2147483648")
	conn.execute("PRAGMA journal_mode=OFF")
	conn.execute("PRAGMA read_uncommitted=TRUE")
	conn.execute("PRAGMA threads=4")
	return conn


def _query(conn, sql, params):
	try:
		return conn.execute(sql, params).fetchall()
	except sqlite3.Error:
		return []


def _fetch_by_subject(conn, subjects: list[str], limit: int) -> list[tuple]:
	if not subjects:
		return []
	ph = ",".join("?" * len(subjects))
	return _query(conn,
		f"SELECT subject,relation,object FROM triples "
		f"WHERE subject IN ({ph}) LIMIT ?",
		tuple(subjects) + (limit,))


def _fetch_by_object(conn, objects: list[str], limit: int) -> list[tuple]:
	if not objects:
		return []
	ph = ",".join("?" * len(objects))
	return _query(conn,
		f"SELECT subject,relation,object FROM triples "
		f"WHERE object IN ({ph}) LIMIT ?",
		tuple(objects) + (limit,))

def normalize_entity(e: str) -> str:
	if _LITERAL_RE.match(e):
		return e
	return e.replace(" ", "_").replace('"', "").replace("'", "").strip()


def entity_aliases(entity: str) -> list[str]:
	if _LITERAL_RE.match(entity):
		return [entity]

	entity = entity.replace("%27", "'")
	words  = entity.split("_")
	candidates = [entity]

	enc = entity.replace("'", "%27")
	if enc != entity:
		candidates.append(enc)

	words_np = [w for w in words if not w.startswith("(")]
	if words_np != words:
		candidates.append("_".join(words_np))

	articles = {"the", "a", "an", "of", "in", "at", "de", "di", "del", "della"}
	core = [w for w in words_np if w.lower() not in articles]
	if core != words_np:
		candidates.append("_".join(core))

	no_us = entity.replace("_", " ")
	if no_us != entity:
		candidates.append(no_us)

	seen, result = set(), []
	for a in candidates:
		if a and a not in seen:
			seen.add(a)
			result.append(a)
	return result

def extract_claim_keywords(claim: str) -> list[str]:
	tokens = nltk.word_tokenize(claim.lower())
	keywords = []
	for tok in tokens:
		if tok.isalpha() and len(tok) > 3 and tok not in _CLAIM_STOPWORDS:
			keywords.append(tok)
	return keywords


_wn_cache: dict[str, set[str]] = {}

def _wn_synonyms(word: str) -> set[str]:
	if word in _wn_cache:
		return _wn_cache[word]
	result: set[str] = {word}
	try:
		for syn in wn.synsets(word)[:3]:
			for lemma in syn.lemmas()[:5]:
				result.add(lemma.name().lower().replace("_", " "))
			for hyper in syn.hypernyms()[:2]:
				for lemma in hyper.lemmas()[:3]:
					result.add(lemma.name().lower().replace("_", " "))
	except Exception:
		pass
	_wn_cache[word] = result
	return result


def _score_triple(prop: str, obj: str, keywords: list[str], wn_expanded: dict[str, set[str]]) -> int:
	prop_lower = prop.lower().replace("_", " ")
	obj_lower  = str(obj).lower().replace("_", " ")
	combined   = prop_lower + " " + obj_lower

	score = 0
	for kw in keywords:
		if kw in prop_lower:
			score += 3
		elif kw in obj_lower:
			score += 2
		else:
			# Check WordNet expansions against prop tokens
			prop_tokens = set(prop_lower.split())
			for syn in wn_expanded.get(kw, {kw}):
				syn_tokens = set(syn.split())
				if prop_tokens & syn_tokens:
					score += 1
					break
	return score

def normalize_triple(s, r, o) -> tuple[str, str, str]:
	if r.startswith("~"):
		return str(o), r[1:], str(s)
	return str(s), str(r), str(o)


def triple_to_str(s, r, o) -> str:
	return f"sub: {s} | prop: {r} | obj: {o}"


def retrieve(
	conn: sqlite3.Connection,
	entities: list[str],
	claim: str = "",
) -> list[str]:

	struct_ents  = [e for e in entities if e and not _LITERAL_RE.match(e)]
	literal_ents = [e for e in entities if e and _LITERAL_RE.match(e)]

	if not struct_ents and not literal_ents:
		return []

	triples_set: set[tuple[str, str, str]] = set()
	entity_triples: dict[str, set] = defaultdict(set)

	for entity in struct_ents:
		aliases = entity_aliases(entity)

		for s, p, o in _fetch_by_subject(conn, aliases, DB_ROWS_PER_ENTITY):
			t = normalize_triple(s, p, o)
			entity_triples[entity].add(t)
			triples_set.add(t)

		for s, p, o in _fetch_by_object(conn, aliases, DB_ROWS_PER_ENTITY):
			t = normalize_triple(s, p, o)
			entity_triples[entity].add(t)
			triples_set.add(t)

	for lit in literal_ents:
		for s, p, o in _fetch_by_object(conn, [lit], DB_ROWS_PER_ENTITY // 4):
			triples_set.add(normalize_triple(s, p, o))

	if len(struct_ents) >= 2:
		entity_alias_sets = {
			e: set(a.lower().replace("_", " ") for a in entity_aliases(e))
			for e in struct_ents
		}
		all_entity_aliases_flat = set()
		for aliases in entity_alias_sets.values():
			all_entity_aliases_flat.update(aliases)

		intermediate_nodes: set[str] = set()
		for s, p, o in triples_set:
			if s.lower().replace("_", " ") not in all_entity_aliases_flat:
				intermediate_nodes.add(s)
			if o.lower().replace("_", " ") not in all_entity_aliases_flat:
				intermediate_nodes.add(o)

		intermediate_list = list(intermediate_nodes)[:HOP2_MAX_NODES]
		if intermediate_list:
			for s, p, o in _fetch_by_subject(conn, intermediate_list, DB_ROWS_PER_ENTITY // 4):
				triples_set.add(normalize_triple(s, p, o))
			for s, p, o in _fetch_by_object(conn, intermediate_list, DB_ROWS_PER_ENTITY // 4):
				triples_set.add(normalize_triple(s, p, o))

	keywords = extract_claim_keywords(claim) if claim else []

	if keywords:
		# Pre-expand all keywords once (cached)
		wn_expanded = {kw: _wn_synonyms(kw) for kw in keywords}

		scored = [
			(_score_triple(p, o, keywords, wn_expanded), s, p, o)
			for s, p, o in triples_set
		]
		scored.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
		ordered = [(s, p, o) for _, s, p, o in scored]
	else:
		ordered = sorted(triples_set, key=lambda x: (x[0], x[1], x[2]))

	return [triple_to_str(s, p, o) for s, p, o in ordered[:MAX_KG_LINES]]

def main():
	_ensure_nltk()

	for path, label in [(TEST_SET_PATH, "test set"), (DB_PATH, "KG DB")]:
		if not os.path.exists(path):
			sys.exit(f"File not found: {path}")
		print(f"[✓] {label}: {path}")

	print("Loading test set…", end=" ", flush=True)
	with open(TEST_SET_PATH, "rb") as f:
		test_set = pickle.load(f)
	instances = list(test_set.items())
	print(f"{len(instances)} instances")

	print(f"Connecting to DB: {DB_PATH}…", end=" ", flush=True)
	conn = open_db(DB_PATH)
	print("OK\n")

	output: dict[str, dict] = {}
	t_start = time.time()

	for idx, (claim, meta) in enumerate(instances):
		current = idx + 1

		lr        = meta["Label"]
		label_val = 1 if (lr[0] if isinstance(lr, list) else lr) in [True, "True", 1] else 0

		raw_ents  = meta.get("Entity_set", [])
		entities  = [normalize_entity(e) for e in raw_ents if e]

		kg_lines  = retrieve(conn, entities, claim)

		output[str(idx)] = {
			"claim":    claim,
			"label":    label_val,
			"entities": entities,
			"kg_lines": kg_lines,
		}

		if current % 500 == 0:
			print(f"\n  Sample #{current}: {claim[:100]}...")
			print(f"  Entities: {entities}")
			print(f"  KG lines: {len(kg_lines)}")
			for line in kg_lines[:3]:
				print(f"    {line}")
			if len(kg_lines) > 3:
				print(f"    ... ({len(kg_lines)} total)")

		if current % 10 == 0 or current == len(instances):
			elapsed = time.time() - t_start
			speed   = current / max(elapsed, 1e-9)
			eta     = (len(instances) - current) / max(speed, 1e-9)
			bar_f   = int(30 * current / len(instances))
			bar     = f"[{'█'*bar_f}{'░'*(30-bar_f)}]"
			eta_s   = f"{eta:.0f}s" if eta < 60 else f"{int(eta)//60}m{int(eta)%60:02d}s"
			print(f"\r{bar} {current:>5}/{len(instances)}  {speed:.0f}it/s  ETA {eta_s}  ",
				  end="", flush=True)

	elapsed = time.time() - t_start
	print(f"\n\nDone in {elapsed:.1f}s  ({len(instances)/elapsed:.1f} it/s)")

	print(f"Saving → {OUT_PATH}…", end=" ", flush=True)
	with open(OUT_PATH, "w", encoding="utf-8") as f:
		json.dump(output, f, ensure_ascii=False, separators=(",", ":"))
	print(f"OK  ({os.path.getsize(OUT_PATH)/1_048_576:.2f} MB)")

	conn.close()


if __name__ == "__main__":
	main()