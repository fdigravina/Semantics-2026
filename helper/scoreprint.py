import json
import os

OUT_PATH = "./data/kg_contexts.json"

R = "\033[0m"; B = "\033[1m"
G = "\033[92m"; RE = "\033[91m"; Y = "\033[93m"; C = "\033[96m"


def inspect_top_claims(num_claims: int = 10):
	if not os.path.exists(OUT_PATH):
		print(
			f"{RE}[X] Errore: Il file {OUT_PATH} non esiste.{R} Esegui prima il retrieval principale."
		)
		return

	print(f"Caricamento dei contesti da {C}{OUT_PATH}{R}...\n")
	with open(OUT_PATH, "r", encoding="utf-8") as f:
		data = json.load(f)

	sorted_keys = sorted(data.keys(), key=int)[:num_claims]

	for key in sorted_keys:
		item = data[key]
		claim = item["claim"]
		label = item["label"]
		entities = item["entities"]
		kg_lines = item["kg_lines"]

		lbl_str = f"{G}TRUE{R}" if label == 1 else f"{RE}FALSE{R}"

		print(f"{B}ID Claim:{R} {C}{key}{R} | {B}Label originale:{R} {lbl_str}")
		print(f"{B}Text:{R} \"{claim}\"")
		print(f"{B}Entities estratte:{R} {entities}")

		print(f"{B}KG Lines trovate ({len(kg_lines)}):{R}")
		if not kg_lines:
			print(f"  {Y}[Contesto vuoto.]{R}")
		else:
			for line in kg_lines:
				formatted_line = (
					line.replace("subject:", f"{B}sub:{R}")
					.replace("property:", f"{B}prop:{R}")
					.replace("object:", f"{B}obj:{R}")
				)
				print(f"  → {formatted_line}")

		print("─" * 60)


if __name__ == "__main__":
	inspect_top_claims(num_claims=45)