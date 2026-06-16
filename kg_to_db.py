import pickle
import sqlite3
import os

PICKLE_PATH = "./data/dbpedia_2015_undirected_light.pickle"
DB_PATH = "./data/dbpedia_light.db"

with open(PICKLE_PATH, 'rb') as f:
	kg_dict = pickle.load(f)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
	CREATE TABLE IF NOT EXISTS triples (
		subject TEXT,
		relation TEXT,
		object TEXT
	)
''')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject)')

buffer = []
for subj, neighbors in kg_dict.items():
	if isinstance(neighbors, dict):
		for rel, objs in neighbors.items():
			if isinstance(objs, list):
				for obj in objs:
					buffer.append((str(subj), str(rel), str(obj)))
			else:
				buffer.append((str(subj), str(rel), str(objs)))
				
	if len(buffer) > 100000:
		cursor.executemany('INSERT INTO triples VALUES (?, ?, ?)', buffer)
		buffer = []

if buffer:
	cursor.executemany('INSERT INTO triples VALUES (?, ?, ?)', buffer)

conn.commit()
conn.close()