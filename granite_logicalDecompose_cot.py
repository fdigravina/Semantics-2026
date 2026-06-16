import os, sys, re, json, time, pickle, math
from llama_cpp import Llama

DATA_DIR        = "./data"
TEST_SET_PATH   = os.path.join(DATA_DIR, "factkg_test.pickle")
OUT_PATH        = os.path.join(DATA_DIR, "decomposed_claims.json")
STATE_FILE      = "decompose_checkpoint.json"

MODEL_PATH      = "/home/filippo/Scrivania/models/granite-4.1-3b-BF16.gguf"

N_CTX           = 512
N_GPU_LAYERS    = 0
N_THREADS       = 4
MAX_TOKENS      = 256
TEMPERATURE     = 0.0
RESET_STATE     = False
LOG_EVERY       = 50

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

SYSTEM_PROMPT = (
	"You are a logical analyst. "
	"When given a factual claim, you decompose it into its atomic propositions "
	"and identify the logical connective that links them. "
	"Respond ONLY with valid JSON, no markdown fences, no extra text."
)

def build_decomposition_prompt(claim: str) -> str:
	"""
	Returns the full prompt string for completion-mode inference.
	The model must output a JSON object with keys:
	  atomic_propositions: list[str]   - one or more minimal sub-claims
	  connectives:         str         - AND | OR | NOT | IF-THEN | SINGLE
	  negated:             bool        - true if the overall claim is negated
	"""
	example = (
		'{"atomic_propositions":["Marie Curie was born in Poland",'
		'"Marie Curie won a Nobel Prize"],'
		'"connectives":"AND","negated":false}'
	)
	return (
		f"{SYSTEM_PROMPT}\n\n"
		f"Example output format (do not copy content):\n{example}\n\n"
		f"Claim: {claim}\n"
		f"JSON:"
	)

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)
_CONNECTIVE_VALUES = {"AND", "OR", "NOT", "IF-THEN", "SINGLE"}

def parse_decomposition(raw: str) -> dict | None:
	"""
	Extract and validate the JSON blob from the model's raw output.
	Returns None on failure.
	"""
	m = _JSON_RE.search(raw)
	if not m:
		return None
	try:
		obj = json.loads(m.group())
	except json.JSONDecodeError:
		return None

	# Validate / normalise
	props = obj.get("atomic_propositions")
	if not isinstance(props, list) or not props:
		return None
	conn = str(obj.get("connectives", "SINGLE")).upper().strip()
	if conn not in _CONNECTIVE_VALUES:
		# try to salvage
		for cv in _CONNECTIVE_VALUES:
			if cv in conn:
				conn = cv
				break
		else:
			conn = "SINGLE" if len(props) == 1 else "AND"
	neg = bool(obj.get("negated", False))
	return {
		"atomic_propositions": [str(p).strip() for p in props if str(p).strip()],
		"connectives":         conn,
		"negated":             neg,
	}

def fallback_decomposition(claim: str) -> dict:
	"""
	Rule-based fallback when the model fails to produce valid JSON.
	Splits on 'and', 'but', 'while', 'or'; detects surface negation.
	"""
	neg = bool(re.search(r'\b(not|never|no|neither|nor)\b', claim, re.I))
	parts = re.split(r'\b(and|but|while|;)\b', claim, flags=re.I)
	props = [p.strip() for p in parts if p.strip() and p.lower() not in ('and','but','while',';')]
	if not props:
		props = [claim.strip()]
	conn = "SINGLE" if len(props) == 1 else "AND"
	return {"atomic_propositions": props, "connectives": conn, "negated": neg}

def save_checkpoint(filepath, last_idx, output, elapsed):
	with open(filepath, 'w', encoding='utf-8') as f:
		json.dump({"last_index": last_idx, "output": output, "elapsed": elapsed}, f)

def load_checkpoint(filepath):
	if os.path.exists(filepath):
		with open(filepath, 'r', encoding='utf-8') as f:
			return json.load(f)
	return None

def normalize_entity(e: str) -> str:
	return e.replace(" ", "_").replace('"', "").replace("'", "")

def main():
	_hdr("STEP 1 – CLAIM DECOMPOSER")

	if not os.path.exists(TEST_SET_PATH):
		sys.exit(f"[✗] Test set not found: {TEST_SET_PATH}")

	print(f"  {GRN}[✓]{RST} Test set: {CYN}{TEST_SET_PATH}{RST}")
	print(f"  {GRN}[✓]{RST} Model:    {CYN}{MODEL_PATH}{RST}")

	with open(TEST_SET_PATH, "rb") as f:
		test_set = pickle.load(f)
	instances = list(test_set.items())
	total     = len(instances)
	print(f"  {GRN}[✓]{RST} Instances: {BLD}{total}{RST}")

	start_idx        = 0
	output: dict     = {}
	previous_elapsed = 0.0

	ckpt = load_checkpoint(STATE_FILE)
	if ckpt and not RESET_STATE:
		start_idx        = ckpt["last_index"] + 1
		output           = {k: v for k, v in ckpt["output"].items()}
		previous_elapsed = ckpt.get("elapsed", 0.0)
		if start_idx >= total:
			print(f"  {YLW}[i]{RST} Checkpoint says processing is complete – resetting.")
			start_idx = 0; output = {}; previous_elapsed = 0.0
		else:
			print(f"  {YLW}[→]{RST} Resuming from index {BLD}{start_idx}{RST}")
	else:
		if RESET_STATE:
			print(f"  {YLW}[i]{RST} RESET_STATE=True – ignoring any checkpoint.")
		else:
			print(f"  {GRN}[i]{RST} No checkpoint found – starting from scratch.")

	_sec("Loading model")
	t0 = time.time()
	llm = Llama(
		model_path   = MODEL_PATH,
		n_ctx        = N_CTX,
		n_gpu_layers = N_GPU_LAYERS,
		n_threads    = N_THREADS,
		verbose      = False,
	)
	print(f"  {GRN}[✓]{RST} Model ready in {time.time()-t0:.1f}s")

	_hdr("DECOMPOSING CLAIMS")
	print(f"{DIM}Press CTRL+C at any time to stop and save progress.{RST}\n")

	fail_count = 0
	start_time = time.time()

	try:
		for idx in range(start_idx, total):
			claim, meta = instances[idx]

			lr        = meta.get("Label", [False])
			label_val = 1 if (lr[0] if isinstance(lr, list) else lr) in [True, "True", 1] else 0
			raw_ents  = meta.get("Entity_set", [])
			entities  = [normalize_entity(e) for e in raw_ents if e]

			prompt = build_decomposition_prompt(claim)
			out    = llm.create_completion(
				prompt       = prompt,
				max_tokens   = MAX_TOKENS,
				temperature  = TEMPERATURE,
				echo         = False,
			)
			raw_text = out["choices"][0]["text"]
			parsed   = parse_decomposition(raw_text)

			if parsed is None:
				fail_count += 1
				parsed = fallback_decomposition(claim)

			output[str(idx)] = {
				"claim":               claim,
				"label":               label_val,
				"entities":            entities,
				"atomic_propositions": parsed["atomic_propositions"],
				"connectives":         parsed["connectives"],
				"negated":             parsed["negated"],
			}

			elapsed = (time.time() - start_time) + previous_elapsed
			done    = idx - start_idx + 1
			speed   = done / max(time.time() - start_time, 1e-9)
			eta     = (total - idx - 1) / max(speed, 1e-9)
			bar     = _bar(idx + 1, total)
			print(
				f"\r{bar} {BLD}{idx+1:>5}/{total}{RST}  "
				f"{speed:.1f} it/s  ETA {CYN}{_eta(eta)}{RST}  "
				f"Fallbacks {YLW}{fail_count}{RST}   ",
				end="", flush=True,
			)

			if (idx + 1) % LOG_EVERY == 0:
				print()
				_sec(f"SAMPLE #{idx+1}", MGT)
				item = output[str(idx)]
				print(f"  {BLD}Claim:{RST}       {item['claim']}")
				print(f"  {BLD}Propositions:{RST}")
				for i, p in enumerate(item["atomic_propositions"], 1):
					print(f"    {CYN}{i}.{RST} {p}")
				print(f"  {BLD}Connective:{RST}  {item['connectives']}")
				print(f"  {BLD}Negated:{RST}     {item['negated']}")
				print()

			if (idx + 1) % 100 == 0:
				save_checkpoint(STATE_FILE, idx, output,
								(time.time() - start_time) + previous_elapsed)

	except KeyboardInterrupt:
		print(f"\n\n{RED}[!] Interrupted – saving checkpoint…{RST}")
		save_checkpoint(STATE_FILE, idx,  # type: ignore[name-defined]
						output, (time.time() - start_time) + previous_elapsed)
		print(f"{YLW}Resume by re-running with RESET_STATE = False.{RST}")
		sys.exit(0)

	elapsed_tot = (time.time() - start_time) + previous_elapsed
	print(f"\n\n{GRN}[✓]{RST} Decomposition complete in {YLW}{elapsed_tot:.1f}s{RST}  "
		  f"({total/elapsed_tot:.1f} it/s)  Fallbacks: {YLW}{fail_count}/{total}{RST}")

	os.makedirs(DATA_DIR, exist_ok=True)
	print(f"\nSaving → {CYN}{OUT_PATH}{RST}…", end=" ", flush=True)
	with open(OUT_PATH, "w", encoding="utf-8") as f:
		json.dump(output, f, ensure_ascii=False, indent=None, separators=(",", ":"))
	size_mb = os.path.getsize(OUT_PATH) / 1_048_576
	print(f"{GRN}OK{RST}  ({size_mb:.2f} MB)")

	if os.path.exists(STATE_FILE):
		os.remove(STATE_FILE)


if __name__ == "__main__":
	main()