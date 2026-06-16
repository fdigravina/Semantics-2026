from nltk.corpus import wordnet as wn

wife = wn.synset('wife.n.01')
spouse = wn.synset('spouse.n.01')

hypernyms = wife.hypernyms()
print("Iperonimi di 'wife' (concetti più generali):")
for h in hypernyms:
    print(f" - {h.name()}: {h.definition()}")

hyponyms = spouse.hyponyms()
print("\nIponimi di 'spouse' (concetti più specifici):")
for h in hyponyms:
    if h == wife:
        print(f" - Trovato! '{wife.name()}' è un iponimo diretto di '{spouse.name()}'")

lcs = wife.lowest_common_hypernyms(spouse)
print(f"\nAntenato comune più vicino: {lcs[0].name()}")