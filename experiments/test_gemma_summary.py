from gemma_summary import summarize_captions_with_gemma

with open("captions.txt", "r", encoding="utf-8") as f:
    captions_text = f.read()

result = summarize_captions_with_gemma(captions_text)

print("Résumé final :")
print(result)