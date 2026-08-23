import joblib
from pathlib import Path

# Load trained model
model_path = Path("models/customer_support_model.pkl")
model = joblib.load(model_path)

print("===== CUSTOMER SUPPORT AI =====")
print("Type 'exit' to stop.\n")

while True:
    text = input("Customer message: ")

    if text.lower() == "exit":
        print("Goodbye!")
        break

    if not text.strip():
        print("Please enter a message.\n")
        continue

    prediction = model.predict([text])[0]

    print(f"Predicted Intent: {prediction}\n")