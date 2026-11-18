import pandas as pd

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print("Fake news data loaded:", len(fake))
print("Genuine news data loaded:", len(true))


fake["label"] = 0
true["label"] = 1


data = pd.concat([fake, true], axis=0)


data = data[["text", "label"]]


data = data.dropna()
data = data.drop_duplicates()


data = data.sample(frac=1, random_state=42)

print("\nDataset loaded successfully!")
print(data.head())

print("\nLabel distribution:")
print(data["label"].value_counts())