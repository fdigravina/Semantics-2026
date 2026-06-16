import os
import sys
import re
import time
import math
import random
import json
import urllib.request
from collections import defaultdict

DATA_DIR        = "./data"
BEAM_WIDTH      = 50    
MAX_NODE_DEGREE = 2500
RANDOM_SEED     = 42

random.seed(RANDOM_SEED)

WN18RR_DIR  = os.path.join(DATA_DIR, "wn18rr")

WN18RR_URLS = {
	"train.txt": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original/train.txt",
	"valid.txt": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original/valid.txt",
	"test.txt":  "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original/test.txt",
}

_STOP_WORDS = {
	"a","an","the","is","was","are","were","be","been","had","has","have",
	"did","do","does","of","to","and","or","but","that","this","with","for",
	"by","as","on","it","its","he","she","they","who","which","from","also",
	"both","one","two","three","known","about","you","know","familiar","yes",
	"i","me","my","myself","we","our","ours","ourselves","in","at","called","named"
}

def download_file(url: str, dest: str):
	try:
		urllib.request.urlretrieve(url, dest)
	except Exception as e:
		sys.exit(1)

def ensure_datasets():
	for dirpath, urls in [(WN18RR_DIR, WN18RR_URLS)]:
		os.makedirs(dirpath, exist_ok=True)
		for fname, url in urls.items():
			fpath = os.path.join(dirpath, fname)
			if not os.path.exists(fpath) or os.path.getsize(fpath) < 100:
				download_file(url, fpath)

class KGGraph:
	def __init__(self, name: str):
		self.name         = name
		self.node_to_id   = {}
		self.id_to_node   = []
		self.graph        = defaultdict(list)
		self.all_entities = set()
		self.amie_transitions = defaultdict(lambda: defaultdict(float))

	def clean_relation(self, rel: str, dataset: str) -> str:
		r = rel.lstrip('_').lstrip('/')
		r = r.replace('/', ' ').replace('_', ' ').replace('.', ' ')
		return re.sub(r'\s+', ' ', r).strip().lower()

	def load_triples(self, filepath: str, is_train: bool, dataset: str):
		triples = []
		with open(filepath, encoding='utf-8') as f:
			for line in f:
				parts = line.strip().split('\t')
				if len(parts) != 3:
					continue
				s, r, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
				triples.append((s, r, o))
				if is_train:
					self.all_entities.add(s)
					self.all_entities.add(o)

		if is_train:
			degree = defaultdict(int)
			for s, r, o in triples:
				degree[s] += 1
				degree[o] += 1

			for s, r, o in triples:
				if degree[s] > MAX_NODE_DEGREE or degree[o] > MAX_NODE_DEGREE:
					continue
				s_id    = self.get_node_id(s)
				o_id    = self.get_node_id(o)
				r_clean = self.clean_relation(r, dataset)
				self.graph[s_id].append((o_id, r_clean, r, 'F'))
				self.graph[o_id].append((s_id, r_clean, r, 'B'))

		return triples

	def get_node_id(self, node: str) -> int:
		node = node.strip()
		if node not in self.node_to_id:
			self.node_to_id[node] = len(self.id_to_node)
			self.id_to_node.append(node)
		return self.node_to_id[node]

	def compute_amie_transitions(self, train_triples: list):
		r1r2_count   = defaultdict(lambda: defaultdict(int))
		r1_count     = defaultdict(int)
		subj_triples = defaultdict(list)
		for s, r, o in train_triples:
			subj_triples[s].append((r, o))
		for s, pairs in subj_triples.items():
			for r1, o1 in pairs:
				r1_count[r1] += 1
				o1_id = self.node_to_id.get(o1)
				if o1_id is None:
					continue
				for (_, _, r2_raw, _) in self.graph.get(o1_id, []):
					r1r2_count[r1][r2_raw] += 1
		for r1, r2_map in r1r2_count.items():
			total = max(r1_count[r1], 1)
			for r2, cnt in r2_map.items():
				self.amie_transitions[r1][r2] = cnt / total

WN_TEMPLATES = {
	"_hypernym":                    "{s} is a type of {o}",
	"_hyponym":                     "{o} is a type of {s}",
	"_instance_hypernym":           "{s} is an instance of {o}",
	"_member_meronym":              "{s} has member {o}",
	"_member_holonym":              "{o} is member of {s}",
	"_has_part":                    "{s} has part {o}",
	"_part_of":                     "{s} is part of {o}",
	"_derivationally_related_form": "{s} is derivationally related to {o}",
	"_synset_domain_topic_of":      "{s} has domain topic {o}",
	"_member_of_domain_region":     "{s} is in domain region {o}",
	"_member_of_domain_usage":      "{s} is used in domain {o}",
	"_similar_to":                  "{s} is similar to {o}",
	"_verb_group":                  "{s} belongs to verb group {o}",
	"_also_see":                    "{s} is also seen as {o}",
	"_attribute":                   "{s} has attribute {o}",
}

def triple_to_keywords(s: str, rel: str, o: str, dataset: str) -> tuple:
	if dataset == "wn18rr":
		template = WN_TEMPLATES.get(rel, "{s} {rel} {o}")
		claim = template.format(
			s=s.replace('_', ' '), o=o.replace('_', ' '),
			rel=rel.lstrip('_').replace('_', ' ')
		)
	else:
		rel_readable = rel.lstrip('/').replace('/', ' ').replace('_', ' ').replace('.', ' ')
		claim = f"{s} {rel_readable} {o}"

	ent_words = set()
	for e in [s, o]:
		ent_words.update(e.lower().replace('_', ' ').replace('/', '').split())
	raw_words = re.findall(r'\b\w+\b', claim.lower())
	keywords  = tuple(w for w in raw_words
					  if w not in _STOP_WORDS and w not in ent_words and len(w) > 2)
	return keywords

def generate_negatives(pos_triples, all_entities, known_set, n_neg):
	entities_list = list(all_entities)
	pos_set   = set((s, r, o) for s, r, o in pos_triples)
	negatives, idx, attempts = [], 0, 0
	while len(negatives) < n_neg and attempts < n_neg * 50:
		s, r, o = pos_triples[idx % len(pos_triples)]
		if random.random() < 0.5:
			fake = (s, r, random.choice(entities_list))
		else:
			fake = (random.choice(entities_list), r, o)
		if fake not in pos_set and fake not in known_set:
			negatives.append(fake)
		idx += 1
		attempts += 1
	return negatives[:n_neg]

def beam_search(start: str, goal: str, keywords: tuple, kg: KGGraph,
				max_steps: int = 3) -> float:
	start_id = kg.node_to_id.get(start)
	goal_id  = kg.node_to_id.get(goal)
	if start_id is None:
		return -1.0

	beam    = {start_id: (0.0, None, 0)}
	visited = {start_id}

	for step in range(max_steps):
		if not beam:
			break
		cands = defaultdict(lambda: -999.0)
		cmeta = {}
		for active_id, (b_score, last_rel, p_len) in beam.items():
			for next_id, rel_clean, rel_raw, _ in kg.graph.get(active_id, []):
				if keywords:
					mf = sum(1.0 for kw in keywords if kw in rel_clean) / len(keywords)
					ls = math.log(mf + 0.20) if mf > 0 else math.log(0.35)
				else:
					ls = 0.0
				bp   = kg.amie_transitions.get(last_rel, {}).get(rel_raw, 0.0) if last_rel else 0.0
				amie = math.log(1.0 + bp * 4.5)
				cum  = b_score + ls + amie - 0.04 * p_len
				if cum > cands[next_id]:
					cands[next_id] = cum
					cmeta[next_id] = (rel_raw, p_len + 1)

		if not cands:
			break
		sorted_c = sorted(cands.items(), key=lambda x: x[1], reverse=True)[:BEAM_WIDTH]
		beam = {}
		for node_id, score in sorted_c:
			if goal_id is not None and node_id == goal_id:
				return score
			if node_id not in visited:
				visited.add(node_id)
				beam[node_id] = (score, cmeta[node_id][0], cmeta[node_id][1])

	return -1.0

def bidir_beam_search(start: str, goal: str, keywords: tuple, kg: KGGraph,
					  max_steps: int = 4) -> float:
	start_id = kg.node_to_id.get(start)
	goal_id  = kg.node_to_id.get(goal)
	if start_id is None or goal_id is None:
		return -1.0

	fwd_beam = {start_id: (0.0, None, 0)}
	bwd_beam = {goal_id:  (0.0, None, 0)}
	fwd_hist = {start_id: 0.0}
	bwd_hist = {goal_id:  0.0}

	for _ in range(max_steps - 1):
		for beam, hist, transitions in [
			(fwd_beam, fwd_hist, kg.amie_transitions),
			(bwd_beam, bwd_hist, kg.amie_transitions),
		]:
			nxt   = defaultdict(lambda: -999.0)
			nmeta = {}
			for active_id, (score, last_rel, p_len) in beam.items():
				for next_id, rel_clean, rel_raw, _ in kg.graph.get(active_id, []):
					kw   = sum(1.0 for k in keywords if k in rel_clean) / max(len(keywords), 1) if keywords else 0.5
					ls   = math.log(kw + 0.25) if kw > 0 else math.log(0.35)
					bp   = transitions.get(last_rel, {}).get(rel_raw, 0.0) if last_rel else 0.0
					amie = math.log(1.0 + bp * 6.0)
					cum  = score + ls + amie - 0.05 * p_len
					if cum > nxt[next_id]:
						nxt[next_id]  = cum
						nmeta[next_id] = (rel_raw, p_len + 1)
			beam.clear()
			for nid, sc in sorted(nxt.items(), key=lambda x: x[1], reverse=True)[:BEAM_WIDTH]:
				beam[nid] = (sc, nmeta[nid][0], nmeta[nid][1])
				if sc > hist.get(nid, -999.0):
					hist[nid] = sc

		common = set(fwd_hist) & set(bwd_hist)
		if common:
			return max(fwd_hist[n] + bwd_hist[n] for n in common)

	return -1.0

def verify(s: str, o: str, keywords: tuple, kg: KGGraph) -> float:
	sc = bidir_beam_search(s, o, keywords, kg, max_steps=4)
	if sc < -900:
		sc = beam_search(s, o, keywords, kg, max_steps=3)
	if sc < -900:
		sc = -1.0
	return sc

def find_threshold(scores_labels: list) -> float:
	thresholds = sorted(set(s for s, _ in scores_labels))
	best_acc, best_t = 0.0, 0.0
	for t in thresholds:
		acc = sum((s >= t) == l for s, l in scores_labels) / len(scores_labels)
		if acc > best_acc:
			best_acc, best_t = acc, t
	return best_t

class CM:
	def __init__(self): self.tp = self.tn = self.fp = self.fn = 0
	def update(self, gold, pred):
		if   gold == 1 and pred == 1: self.tp += 1
		elif gold == 1 and pred == 0: self.fn += 1
		elif gold == 0 and pred == 1: self.fp += 1
		else:                         self.tn += 1
	@property
	def accuracy(self):  return (self.tp + self.tn) / max(self.tp+self.tn+self.fp+self.fn, 1)
	@property
	def precision(self): return self.tp / max(self.tp + self.fp, 1)
	@property
	def recall(self):    return self.tp / max(self.tp + self.fn, 1)
	@property
	def f1(self):
		p, r = self.precision, self.recall
		return 2*p*r / max(p+r, 1e-9)

def run_benchmark(name: str, dirpath: str, dataset_key: str) -> dict:
	kg = KGGraph(name)
	train_triples = kg.load_triples(os.path.join(dirpath, "train.txt"), is_train=True,  dataset=dataset_key)
	valid_triples = kg.load_triples(os.path.join(dirpath, "valid.txt"), is_train=False, dataset=dataset_key)
	test_triples  = kg.load_triples(os.path.join(dirpath, "test.txt"),  is_train=False, dataset=dataset_key)

	kg.compute_amie_transitions(train_triples)

	known = (set((s,r,o) for s,r,o in train_triples) |
			 set((s,r,o) for s,r,o in valid_triples))
	entities_list = list(kg.all_entities)

	n_val   = len(valid_triples)
	val_neg = generate_negatives(valid_triples, entities_list, known, n_val)
	val_sl  = []
	all_val = [(t, 1) for t in valid_triples] + [(t, 0) for t in val_neg]
	for vi, ((s, r, o), lbl) in enumerate(all_val):
		kw = triple_to_keywords(s, r, o, dataset_key)
		sc = verify(s, o, kw, kg)
		val_sl.append((sc, lbl))
	
	threshold = find_threshold(val_sl)

	n_pos    = len(test_triples)
	neg_s    = generate_negatives(test_triples, entities_list, known, n_pos)
	all_test = [(t, 1) for t in test_triples] + [(t, 0) for t in neg_s]
	random.shuffle(all_test)

	total_len = len(all_test)
	cm = CM()

	for idx, ((s, r, o), gold) in enumerate(all_test):
		kw   = triple_to_keywords(s, r, o, dataset_key)
		sc   = verify(s, o, kw, kg)
		pred = 1 if sc >= threshold else 0
		cm.update(gold, pred)

	return {
		"dataset":   name,
		"n_train":   len(train_triples),
		"n_valid":   n_val,
		"n_test":    total_len,
		"threshold": threshold,
		"accuracy":  cm.accuracy,
		"precision": cm.precision,
		"recall":    cm.recall,
		"f1":        cm.f1,
		"tp": cm.tp, "tn": cm.tn, "fp": cm.fp, "fn": cm.fn,
	}

def print_report(results: list):
	for r in results:
		print(f"\n[{r['dataset']}] n_test={r['n_test']:,}")
		print(f"  Accuracy:  {r['accuracy']:.4f}")
		print(f"  Precision: {r['precision']:.4f}  Recall: {r['recall']:.4f}  F1: {r['f1']:.4f}")
		print(f"  TP={r['tp']}  TN={r['tn']}  FP={r['fp']}  FN={r['fn']}")
		print(f"  Threshold: {r['threshold']:.4f}")

def main():
	ensure_datasets()
	results = []
	results.append(run_benchmark("WN18RR",  WN18RR_DIR, "wn18rr"))
	print_report(results)

	out = os.path.join(DATA_DIR, "transfer_results.json")
	with open(out, "w") as f:
		json.dump({"results": results}, f, indent=2)

if __name__ == "__main__":
	main()