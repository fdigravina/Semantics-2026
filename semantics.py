import os
import sys
import re
import json
import pickle
import sqlite3
import time
import unicodedata
import math
from functools import lru_cache
from collections import defaultdict

BEAM_WIDTH      = 150  
MAX_NODE_DEGREE = 2500 

DATA_DIR       = "./data"
TRAIN_SET_PATH = os.path.join(DATA_DIR, "factkg_train.pickle")
DEV_SET_PATH   = os.path.join(DATA_DIR, "factkg_dev.pickle")
TEST_SET_PATH  = os.path.join(DATA_DIR, "factkg_test.pickle")
DB_PATH        = os.path.join(DATA_DIR, "dbpedia_light.db")
OUT_PATH       = os.path.join(DATA_DIR, "results.json")

_STRIP_QUOTES_RE = re.compile(r'["\'`‘’“”]')
_LITERAL_RE = re.compile(r'^\d+(\.\d+)?$|^\d{4}-\d{2}-\d{2}$|^\d{4}$')

_STOP_WORDS = { 
	"a", "an", "the", "is", "was", "are", "were", "be", "been", "had", "has", "have", 
	"did", "do", "does", "of", "to", "and", "or", "but", "that", "this", "with", "for", 
	"by", "as", "on", "it", "its", "he", "she", "they", "who", "which", "from", "also", 
	"both", "one", "two", "three", "known", "about", "you", "know", "familiar", "yes", 
	"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "in", "at", "called", "named" 
} 

_NEGATION_WORDS = { 
	"not", "no", "never", "neither", "nor", "wasnt", "isnt", "arent", "werent",  
	"didnt", "doesnt", "hasnt", "hadnt", "cannot", "cant", "without", "refused" 
} 

_MULTICLAIM_WORDS = {"and", "both", "along", "together", "with", ",", "as well as", "addition"}

NODE_TO_ID = {}
ID_TO_NODE = []
RAM_GRAPH = defaultdict(list)

def clean_string_completely(s: str) -> str:
	if not s: return ""
	return _STRIP_QUOTES_RE.sub("", s).strip()

def get_node_id(node_str: str) -> int:
	node_clean = clean_string_completely(node_str)
	if node_clean not in NODE_TO_ID:
		NODE_TO_ID[node_clean] = len(ID_TO_NODE)
		ID_TO_NODE.append(node_clean)
	return NODE_TO_ID[node_clean]

class ConfusionMatrix:
	def __init__(self, name="GLOBAL"):
		self.name = name
		self.tp = self.tn = self.fp = self.fn = 0
	def update(self, gold: int, pred: int):
		if gold == 1 and pred == 1:   self.tp += 1
		elif gold == 1 and pred == 0: self.fn += 1
		elif gold == 0 and pred == 1: self.fp += 1
		else:                         self.tn += 1
	@property
	def accuracy(self):  
		return (self.tp + self.tn) / max(self.tp + self.tn + self.fp + self.fn, 1)

class EvaluationManager:
	def __init__(self):
		self.global_cm = ConfusionMatrix("GLOBAL")
		self.valid_types = {"num1", "multi claim", "existence", "multi hop", "negation"}
		self.type_cms = {t: ConfusionMatrix(t) for t in self.valid_types}

	def update(self, gold: int, pred: int, types_list: list):
		self.global_cm.update(gold, pred)
		for t in types_list:
			if t in self.valid_types:
				self.type_cms[t].update(gold, pred)

	def print_report(self):
		print(f"\n=======================================================")
		print(f" REASONING TYPE       |  ACC     |  FP      |  FN      |")
		print(f"=======================================================")
		for t_name in sorted(self.valid_types):
			cm = self.type_cms[t_name]
			print(f" {t_name:<20} |  {cm.accuracy:.4f}  |  {cm.fp:<7} |  {cm.fn:<7} |")
		print(f"-------------------------------------------------------")
		g = self.global_cm
		print(f" {'GLOBAL (TOTAL)':<20} |  {g.accuracy:.4f}  |  {g.fp:<7} |  {g.fn:<7} |")
		print(f"=======================================================\n")

@lru_cache(maxsize=131072)
def _fast_norm(s: str) -> str:
	s_clean = clean_string_completely(s).lower().replace("_", "").replace(" ", "")
	nfkd_form = unicodedata.normalize('NFKD', s_clean)
	return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

@lru_cache(maxsize=65536)
def normalize_entity(e: str) -> str:
	if not e: return ""
	return clean_string_completely(e).replace(" ", "_")

@lru_cache(maxsize=65536)
def entity_aliases(entity: str) -> list:
	if not entity: return []
	base = clean_string_completely(entity)
	return list({base, base.replace(" ", "_"), base.replace("_", " ")})

@lru_cache(maxsize=32768)
def _clean_predicate_fast(rel: str) -> str:
	clean = rel.split(":")[-1].split("/")[-1]
	return re.sub(r'([a-z])([A-Z])', r'\1 \2', clean).lower().replace("_", " ")

def extract_keywords_fast(claim: str, entities: list) -> tuple:
	claim_clean = clean_string_completely(claim)
	raw_words = re.findall(r'\b\w+\b', claim_clean.lower())
	words = {w for w in raw_words if w not in _STOP_WORDS}
	ent_words = set()
	for e in entities:
		ent_words.update(clean_string_completely(e).lower().replace("_", " ").split())
	return tuple(words - ent_words)

def load_restricted_subgraph(db_path: str, test_instances: list):
	global RAM_GRAPH
	print(f"-> Selected topological extraction (Degree Threshold: {MAX_NODE_DEGREE})...")
	
	target_nodes = set()
	for _, meta in test_instances:
		for e in meta.get("Entity_set", []):
			if e:
				e_clean = clean_string_completely(e)
				if not _LITERAL_RE.match(e_clean):
					target_nodes.update(entity_aliases(normalize_entity(e_clean)))
				
	if not target_nodes: return

	t_start = time.time()
	conn = sqlite3.connect(db_path)
	bad_rels = {"type", "label", "wikipagewikilink", "subject", "wikiPageRedirects", "wikiPageDisambiguates"}
	
	node_list = list(target_nodes)
	chunk_size = 250
	triples_loaded = 0
	
	for i in range(0, len(node_list), chunk_size):
		chunk = node_list[i:i+chunk_size]
		ph = ",".join("?" * len(chunk))
		sql = f"SELECT subject, relation, object FROM triples WHERE subject IN ({ph}) OR object IN ({ph})"
		rows = conn.execute(sql, tuple(chunk) + tuple(chunk)).fetchall()
		
		local_degree = defaultdict(int)
		valid_rows = []
		for s, r, o in rows:
			s_c, o_c = clean_string_completely(s), clean_string_completely(o)
			rel_short = r.split("/")[-1].split("#")[-1].lower()
			if rel_short in bad_rels: continue
			local_degree[s_c] += 1
			local_degree[o_c] += 1
			valid_rows.append((s_c, r, o_c))
			
		for s, r, o in valid_rows:
			if local_degree[s] > MAX_NODE_DEGREE or local_degree[o] > MAX_NODE_DEGREE: continue
			
			r_clean = _clean_predicate_fast(r)
			s_id = get_node_id(s)
			o_id = get_node_id(o)
			
			RAM_GRAPH[s_id].append((o_id, r_clean, r, 'F'))
			RAM_GRAPH[o_id].append((s_id, r_clean, r, 'B'))
			triples_loaded += 1

	conn.close()
	print(f"--> Subgraph loaded: {triples_loaded} edges in {time.time() - t_start:.2f}s.")


class StatisticalTrainer:
	def __init__(self):
		self.keyword_type_distribution = defaultdict(lambda: defaultdict(int))
		self.type_occurrences = defaultdict(int)
		self.amie_transitions = defaultdict(lambda: defaultdict(float))
		self.amie_transitions_inv = defaultdict(lambda: defaultdict(float))
		self.token_polarity_score = defaultdict(lambda: {"true_count": 0, "total": 0})
		self.token_direction_inversion_score = defaultdict(lambda: {"inverted": 0, "standard": 0})
		self.token_to_rel_prob = defaultdict(lambda: defaultdict(int))
		self.rel_total_counts = defaultdict(int)

	def train(self, datasets_paths: list):
		print("-> Statistical analysis of training datasets...")
		for path in datasets_paths:
			if not os.path.exists(path): continue
			with open(path, "rb") as f: data = pickle.load(f)
			for claim, meta in data.items():
				types = meta.get("types", [])
				evidence = meta.get("Evidence", {})
				lr = meta.get("Label", [False])
				is_true = 1 if (lr[0] if isinstance(lr, list) else lr) in [True, "True", 1] else 0
				
				claim_clean = clean_string_completely(claim)
				claim_lower = claim_clean.lower()
				raw_words = re.findall(r'\b\w+\b', claim_lower)
				words = {w for w in raw_words if w not in _STOP_WORDS and len(w) > 2}
				
				for w in words:
					self.token_polarity_score[w]["total"] += 1
					if is_true: self.token_polarity_score[w]["true_count"] += 1

				for t in types:
					self.type_occurrences[t] += 1
					for w in words:
						if len(w) > 3: self.keyword_type_distribution[w][t] += 1
				
				if is_true:
					for _, paths in evidence.items():
						for path_chain in paths:
							for r_raw in path_chain:
								r_clean = str(r_raw).replace("~", "")
								r_short = _clean_predicate_fast(r_clean)
								self.rel_total_counts[r_short] += 1
								for w in words:
									self.token_to_rel_prob[w][r_short] += 1
				
				entities_raw = [clean_string_completely(e) for e in meta.get("Entity_set", []) if e]
				if len(entities_raw) >= 2 and is_true:
					e1_clean = entities_raw[0].lower().replace("_", " ")
					e2_clean = entities_raw[-1].lower().replace("_", " ")
					idx_e1 = claim_lower.find(e1_clean)
					idx_e2 = claim_lower.find(e2_clean)
					
					spatial_order_inverted = (idx_e1 > idx_e2 and idx_e2 != -1)
					
					for _, paths in evidence.items():
						for path_chain in paths:
							if isinstance(path_chain, list) and len(path_chain) >= 1:
								for i in range(len(path_chain) - 1):
									r1 = str(path_chain[i]).replace("~", "")
									r2 = str(path_chain[i+1]).replace("~", "")
									self.amie_transitions[r1][r2] += 1.0
									self.amie_transitions_inv[r2][r1] += 1.0
								
								graph_inverted = str(path_chain[0]).startswith("~")
								for w in words:
									if len(w) > 2:
										if graph_inverted != spatial_order_inverted:
											self.token_direction_inversion_score[w]["inverted"] += 1
										else:
											self.token_direction_inversion_score[w]["standard"] += 1

		for r1 in self.amie_transitions:
			total_weight = sum(self.amie_transitions[r1].values())
			for r2 in self.amie_transitions[r1]: self.amie_transitions[r1][r2] /= max(total_weight, 1.0)
			
		for r1 in self.amie_transitions_inv:
			total_weight = sum(self.amie_transitions_inv[r1].values())
			for r2 in self.amie_transitions_inv[r1]: self.amie_transitions_inv[r2][r1] /= max(total_weight, 1.0)

	def infer_structural_types(self, claim: str, entities: list) -> str:
		claim_clean = clean_string_completely(claim)
		cleaned_ents = [normalize_entity(clean_string_completely(e)) for e in entities if e]
		se = [e for e in cleaned_ents if not _LITERAL_RE.match(e)]
		le = [e for e in cleaned_ents if _LITERAL_RE.match(e)]
		
		if le and len(se) == 1:
			return "num1"
		if len(se) == 1 and not le:
			return "existence"
		if any(f" {word} " in f" {claim_clean.lower()} " for word in _MULTICLAIM_WORDS):
			return "multi claim"
			
		raw_words = re.findall(r'\b\w+\b', claim_clean.lower())
		words = {w for w in raw_words if w not in _STOP_WORDS}
		
		type_scores = defaultdict(float)
		for w in words:
			if w in self.keyword_type_distribution:
				for t, freq in self.keyword_type_distribution[w].items():
					type_scores[t] += freq / max(self.type_occurrences[t], 1)
		return str(sorted(type_scores.items(), key=lambda x: x[1], reverse=True)[0][0]) if type_scores else "multi hop"

	def check_negation_signals(self, claim: str, predicted_type: str) -> bool:
		claim_lower = clean_string_completely(claim).lower()
		if any(f" {tok} " in f" {claim_lower} " or claim_lower.startswith(tok) for tok in _NEGATION_WORDS):
			return True
		if predicted_type == "negation":
			return True
		raw_words = re.findall(r'\b\w+\b', claim_lower)
		words = {w for w in raw_words if w not in _STOP_WORDS}
		inversion_votes, total_signals = 0, 0
		for w in words:
			if w in self.token_polarity_score and self.token_polarity_score[w]["total"] > 15:
				stats = self.token_polarity_score[w]
				if (stats["true_count"] / stats["total"]) < 0.20: 
					inversion_votes += 1
				total_signals += 1
		return (inversion_votes / total_signals) > 0.50 if total_signals > 0 else False

	def has_multiclaim_signals(self, claim: str) -> bool:
		claim_lower = clean_string_completely(claim).lower()
		return any(f" {word} " in f" {claim_lower} " or word in claim_lower for word in _MULTICLAIM_WORDS)

	def is_claim_semantically_inverted(self, claim: str) -> bool:
		raw_words = re.findall(r'\b\w+\b', clean_string_completely(claim).lower())
		words = {w for w in raw_words if w not in _STOP_WORDS and len(w) > 2}
		
		inversion_votes = 0
		valid_signals = 0
		for w in words:
			if w in self.token_direction_inversion_score:
				counts = self.token_direction_inversion_score[w]
				total = counts["inverted"] + counts["standard"]
				if total > 12:
					valid_signals += 1
					if counts["inverted"] > (counts["standard"] * 1.5):
						inversion_votes += 1
		return (inversion_votes / valid_signals) > 0.50 if valid_signals > 0 else False

	def score_relation_by_keywords(self, rel_clean: str, keywords: tuple) -> float:
		if not keywords: return 1.0
		score = 0.0
		for kw in keywords:
			if kw in rel_clean:
				score += 1.0
			if kw in self.token_to_rel_prob:
				total_rel_instances = max(self.rel_total_counts[rel_clean], 1)
				score += (self.token_to_rel_prob[kw][rel_clean] / total_rel_instances) * 2.5
		return score


def semantic_bidirectional_beam_search(start_entities: list, goal_entities: list, keywords: tuple, trainer: StatisticalTrainer, max_steps: int) -> bool:
	start_ids = [NODE_TO_ID[a] for e in start_entities for a in entity_aliases(clean_string_completely(e)) if a in NODE_TO_ID]
	goal_ids  = [NODE_TO_ID[a] for e in goal_entities for a in entity_aliases(clean_string_completely(e)) if a in NODE_TO_ID]
	if not start_ids or not goal_ids: return False

	forward_beam = {node_id: (0.0, None, None, 0) for node_id in start_ids}  
	backward_beam = {node_id: (0.0, None, None, 0) for node_id in goal_ids}
	
	forward_history = {}
	backward_history = {}

	for step in range(max_steps - 1):
		if forward_beam:
			next_f_beam = defaultdict(lambda: -999999.0)
			next_f_meta = {}
			for active_id, (score, last_rel, last_dir, p_len) in forward_beam.items():
				for next_id, rel_clean, rel_raw, direction in RAM_GRAPH.get(active_id, []):
					dir_penalty = 0.0 if (last_dir is None or last_dir == direction) else -0.35
					match_factor = sum(1.0 for kw in keywords if kw in rel_clean) / max(len(keywords), 1)
					local_score = math.log(0.35) if match_factor == 0 else math.log(match_factor + 0.25)
					
					bonus_prob = trainer.amie_transitions.get(last_rel, {}).get(rel_raw, 0.0)
					bonus = math.log(1.0 + bonus_prob * 6.0)
					
					cum_score = score + local_score + bonus + dir_penalty - (0.05 * p_len)
					if cum_score > next_f_beam[next_id]:
						next_f_beam[next_id] = cum_score
						next_f_meta[next_id] = (rel_raw, direction, p_len + 1)
			
			sorted_f = sorted(next_f_beam.items(), key=lambda x: x[1], reverse=True)[:BEAM_WIDTH]
			forward_beam = {nid: (sc, next_f_meta[nid][0], next_f_meta[nid][1], next_f_meta[nid][2]) for nid, sc in sorted_f}
			for nid, (sc, r, d, pl) in forward_beam.items():
				if nid not in forward_history or sc > forward_history[nid][0]:
					forward_history[nid] = (sc, r, d)

		if backward_beam:
			next_b_beam = defaultdict(lambda: -999999.0)
			next_b_meta = {}
			for active_id, (score, last_rel, last_dir, p_len) in backward_beam.items():
				for next_id, rel_clean, rel_raw, direction in RAM_GRAPH.get(active_id, []):
					dir_penalty = 0.0 if (last_dir is None or last_dir == direction) else -0.35
					match_factor = sum(1.0 for kw in keywords if kw in rel_clean) / max(len(keywords), 1)
					local_score = math.log(0.35) if match_factor == 0 else math.log(match_factor + 0.25)
					
					bonus_prob = trainer.amie_transitions_inv.get(last_rel, {}).get(rel_raw, 0.0)
					bonus = math.log(1.0 + bonus_prob * 6.0)
					
					cum_score = score + local_score + bonus + dir_penalty - (0.05 * p_len)
					if cum_score > next_b_beam[next_id]:
						next_b_beam[next_id] = cum_score
						next_b_meta[next_id] = (rel_raw, direction, p_len + 1)

			sorted_b = sorted(next_b_beam.items(), key=lambda x: x[1], reverse=True)[:BEAM_WIDTH]
			backward_beam = {nid: (sc, next_b_meta[nid][0], next_b_meta[nid][1], next_b_meta[nid][2]) for nid, sc in sorted_b}
			for nid, (sc, r, d, pl) in backward_beam.items():
				if nid not in backward_history or sc > backward_history[nid][0]:
					backward_history[nid] = (sc, r, d)

		common_nodes = set(forward_history.keys()) & set(backward_history.keys())
		for node_id in common_nodes:
			f_rel = forward_history[node_id][1]
			b_rel = backward_history[node_id][1]
			if f_rel is None or b_rel is None: return True
			if trainer.amie_transitions.get(f_rel, {}).get(b_rel, 0) > 0.01 or trainer.amie_transitions_inv.get(b_rel, {}).get(f_rel, 0) > 0.01:
				return True
			
			node_name_lower = ID_TO_NODE[node_id].lower()
			if any(kw in node_name_lower for kw in keywords if len(kw) > 2):
				return True

	return False

def amie_guided_beam_search_ram(start_entities: list, goal_entities: list, keywords: tuple, amie_rules: dict, estimated_steps: int, predicted_type: str) -> bool:
	start_ids = [NODE_TO_ID[a] for e in start_entities for a in entity_aliases(clean_string_completely(e)) if a in NODE_TO_ID]
	if not start_ids: return False
	
	goal_norms = {_fast_norm(a) for e in goal_entities for a in entity_aliases(clean_string_completely(e))}
	beam_dict = {node_id: (0.0, None, None, 0) for node_id in start_ids} 
	visited = set(start_ids)

	for step in range(estimated_steps):
		if not beam_dict: break
		candidates_pool = defaultdict(lambda: -999999.0)
		candidate_meta = {}

		for active_id, (b_score, last_rel, last_dir, p_len) in beam_dict.items():
			neighbors = RAM_GRAPH.get(active_id, [])
			if not neighbors: continue
			
			for next_id, rel_clean, rel_raw, direction in neighbors:
				if last_dir is not None and last_dir != direction and predicted_type in ["num1", "existence"]:
					continue
				
				match_factor = sum(1.0 for kw in keywords if kw in rel_clean) / max(len(keywords), 1)
				local_score = math.log(0.35) if (step < (estimated_steps - 1) and match_factor == 0) else math.log(match_factor + 0.20)
				
				bonus_prob = amie_rules.get(last_rel, {}).get(rel_raw, 0.0)
				amie_bonus = math.log(1.0 + bonus_prob * 4.5)
				
				cum_score = b_score + local_score + amie_bonus - (0.04 * p_len)
				
				if cum_score > candidates_pool[next_id]:
					candidates_pool[next_id] = cum_score
					candidate_meta[next_id] = (rel_raw, direction, p_len + 1)

		if not candidates_pool: break
		sorted_cands = sorted(candidates_pool.items(), key=lambda x: x[1], reverse=True)[:BEAM_WIDTH]
		
		beam_dict.clear()
		for node_id, score in sorted_cands:
			if _fast_norm(ID_TO_NODE[node_id]) in goal_norms: return True
			
			node_name_clean = ID_TO_NODE[node_id].lower().replace("_", " ")
			if any(gn in node_name_clean for gn in goal_norms if len(gn) > 3): return True
			
			if node_id not in visited:
				visited.add(node_id)
				beam_dict[node_id] = (score, candidate_meta[node_id][0], candidate_meta[node_id][1], candidate_meta[node_id][2])
				
	return False


def universal_pipeline(entities: list, claim: str, predicted_type: str, trainer: StatisticalTrainer) -> bool:
	cleaned_ents = [normalize_entity(clean_string_completely(e)) for e in entities if e]
	se = [e for e in cleaned_ents if not _LITERAL_RE.match(e)]
	le = [e for e in cleaned_ents if _LITERAL_RE.match(e)]
	keywords = extract_keywords_fast(claim, cleaned_ents)

	if trainer.has_multiclaim_signals(claim) and len(se) >= 3:
		predicted_type = "multi claim"

	steps = 5 if predicted_type == "multi hop" else (2 if "num" in predicted_type else 3)

	if len(se) == 1 and le:
		target_lit = le[0].lower()
		start_nodes = [NODE_TO_ID[a] for a in entity_aliases(se[0]) if a in NODE_TO_ID]
		
		best_match_score = 0.0
		threshold = 0.85
		
		for s_id in start_nodes:
			for n1_id, rel_clean, _, _ in RAM_GRAPH.get(s_id, []):
				n1_name = ID_TO_NODE[n1_id].lower()
				if target_lit == n1_name or f"_{target_lit}_" in f"_{n1_name}_" or n1_name.endswith(f"_{target_lit}"):
					if not keywords: return True
					rel_score = trainer.score_relation_by_keywords(rel_clean, keywords)
					if rel_score > best_match_score:
						best_match_score = rel_score

			for n1_id, _, _, _ in RAM_GRAPH.get(s_id, []):
				for n2_id, rel2_clean, _, _ in RAM_GRAPH.get(n1_id, []):
					n2_name = ID_TO_NODE[n2_id].lower()
					if target_lit == n2_name or f"_{target_lit}_" in f"_{n2_name}_":
						if not keywords: return True
						rel_score = trainer.score_relation_by_keywords(rel2_clean, keywords)
						if rel_score > best_match_score:
							best_match_score = rel_score
							
		return best_match_score >= threshold

	if not se: return False

	if predicted_type == "existence" or len(se) == 1:
		start_nodes = [NODE_TO_ID[a] for a in entity_aliases(se[0]) if a in NODE_TO_ID]
		if not keywords: 
			return len(start_nodes) > 0
			
		best_exist_score = 0.0
		for s_id in start_nodes:
			for _, rc, _, _ in RAM_GRAPH.get(s_id, []):
				score = trainer.score_relation_by_keywords(rc, keywords)
				if score > best_exist_score:
					best_exist_score = score
					
		return best_exist_score >= 1.2

	if predicted_type == "multi claim" and len(se) >= 3:
		def get_degree(ent):
			return sum(len(RAM_GRAPH.get(NODE_TO_ID.get(a, -1), [])) for a in entity_aliases(ent))
		
		sorted_se = sorted(se, key=get_degree, reverse=True)
		hub_entity = sorted_se[0]
		
		hub_id = NODE_TO_ID.get(normalize_entity(hub_entity), -1)
		if hub_id != -1 and len(RAM_GRAPH.get(hub_id, [])) > 1:
			leaves = [e for e in se if e != hub_entity]
			sub_results = [amie_guided_beam_search_ram([hub_entity], [leaf], keywords, trainer.amie_transitions, 2, predicted_type) for leaf in leaves]
			if sum(sub_results) >= (len(sub_results) * 0.75): return True

		sub_results_orig = [amie_guided_beam_search_ram([se[0]], [leaf], keywords, trainer.amie_transitions, 2, predicted_type) for leaf in se[1:]]
		return sum(sub_results_orig) >= (len(sub_results_orig) * 0.75)

	if len(se) >= 2:
		claim_lower = claim.lower()
		idx_e1 = claim_lower.find(se[0].lower().replace("_", " "))
		idx_e2 = claim_lower.find(se[-1].lower().replace("_", " "))
		
		src_ents, tgt_ents = [se[0]], [se[-1]]
		if idx_e1 > idx_e2 and idx_e2 != -1:
			src_ents, tgt_ents = [se[-1]], [se[0]]
			
		if trainer.is_claim_semantically_inverted(claim):
			src_ents, tgt_ents = tgt_ents, src_ents

		if predicted_type == "multi hop":
			if semantic_bidirectional_beam_search(src_ents, tgt_ents, keywords, trainer, steps): return True

		if amie_guided_beam_search_ram(src_ents, tgt_ents, keywords, trainer.amie_transitions, steps, predicted_type): return True
		return amie_guided_beam_search_ram(tgt_ents, src_ents, keywords, trainer.amie_transitions_inv, steps, predicted_type)
		
	return False


def main():
	with open(TEST_SET_PATH, "rb") as f: test_set = pickle.load(f)
	instances = list(test_set.items())

	load_restricted_subgraph(DB_PATH, instances)
	
	trainer = StatisticalTrainer()
	trainer.train([TRAIN_SET_PATH, DEV_SET_PATH])

	eval_manager = EvaluationManager()
	output_records = {}
	total_len, t0 = len(instances), time.time()

	print(f"Optimized Global Processing on {total_len} instances...")

	for idx, (claim, meta) in enumerate(instances):
		raw_entities = meta.get("Entity_set", [])
		
		predicted_type = trainer.infer_structural_types(claim, raw_entities)
		is_negated = trainer.check_negation_signals(claim, predicted_type)
		
		fact_exists = universal_pipeline(raw_entities, claim, predicted_type, trainer)
		
		if is_negated:
			pred = 0 if fact_exists else 1
		else:
			pred = 1 if fact_exists else 0
		
		lr = meta.get("Label", [False])
		gold = 1 if (lr[0] if isinstance(lr, list) else lr) in [True, "True", 1] else 0

		eval_manager.update(gold, pred, meta.get("types", []))
		output_records[str(idx)] = {"claim": claim, "label": gold, "pred": pred}

		if (idx + 1) % 10 == 0 or (idx + 1) == total_len:
			print(f"\rProcessed: {idx+1}/{total_len} | Global Acc: {eval_manager.global_cm.accuracy:.4f} | Speed: {(idx+1)/(time.time()-t0):.1f} claims/sec", end="", flush=True)

	eval_manager.print_report()
	with open(OUT_PATH, "w", encoding="utf-8") as f: json.dump(output_records, f, indent=2)

if __name__ == "__main__":
	main()