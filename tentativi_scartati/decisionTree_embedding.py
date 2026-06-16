import os
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

DATA_DIR         = "./data"
TEST_SET_PATH    = os.path.join(DATA_DIR, "factkg_test.pickle")
KG_CONTEXTS_PATH = os.path.join(DATA_DIR, "kg_contexts.json")

with open(KG_CONTEXTS_PATH, encoding="utf-8") as f:
	kg_data = json.load(f)
with open(TEST_SET_PATH, "rb") as f:
	test_set = pickle.load(f)

X = []
y = []

for idx, (claim, meta) in enumerate(test_set.items()):
	str_idx = str(idx)
	if str_idx not in kg_data:
		continue
		
	kg_item = kg_data[str_idx]
	label = kg_item["label"] # 0 o 1
	kg_lines = kg_item["kg_lines"]
	types = meta.get("types", [])
	
	f_substitution = 1 if "substitution" in types else 0
	f_kg_count = len(kg_lines)
	f_claim_len = len(claim.split())
	
	claim_words = set(claim.lower().replace(".", "").split())
	kg_blob = " ".join(kg_lines).lower()
	f_overlap = sum(1 for w in claim_words if w in kg_blob)
	
	X.append([f_substitution, f_kg_count, f_claim_len, f_overlap])
	y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)

print("\n=== METRICHE CLASSIFICATORE ML LEGGERO ===")
print(classification_report(y_test, preds, target_names=["FALSE", "TRUE"]))

print("=== IMPORTANZA DELLE FEATURE ===")
features_names = ["Has Substitution", "KG Lines Count", "Claim Length", "Lexical Overlap"]
for name, importance in zip(features_names, clf.feature_importances_):
	print(f"  {name:<20}: {importance:.4f}")