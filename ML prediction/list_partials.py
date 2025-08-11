import json, glob

partials = []
for file in glob.glob("partials/*.json"):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        partials.append((file, data.get("success_steps", 0)))

# Sort by steps (descending)
partials.sort(key=lambda x: x[1], reverse=True)

for file, steps in partials:
    print(f"{steps}/10 -> {file}")
