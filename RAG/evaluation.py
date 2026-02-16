from rag_engine import search

test_queries = [
    {"query": "responsabilidade civil", "expected_doc": "Codigo_Civil.pdf"},
    {"query": "direitos trabalhistas", "expected_doc": "CLT.pdf"}
]

correct = 0

for test in test_queries:
    results = search(test["query"], k=1)

    if results and results[0]["document"] == test["expected_doc"]:
        correct += 1

accuracy = correct / len(test_queries)

print(f"Acurácia: {accuracy}")
