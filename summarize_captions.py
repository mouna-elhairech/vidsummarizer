from transformers import pipeline

descriptions = []

with open("captions.txt", "r", encoding="utf-8") as f:
    for line in f:
        if " : " in line:
            description = line.split(" : ", 1)[1].strip()
            descriptions.append(description)

clean_text = ". ".join(descriptions)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

summary = summarizer(clean_text, max_length=100, min_length=40, do_sample=False)

final_summary = summary[0]["summary_text"].strip()

if not final_summary.lower().startswith("the video"):
    final_summary = "The video shows " + final_summary[0].lower() + final_summary[1:]

print("Résumé final :")
print(final_summary)