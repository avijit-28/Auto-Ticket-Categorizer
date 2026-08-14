import pandas as pd
import random
import os

random.seed(42)

billing = [
    "I was charged twice for my subscription",
    "My invoice shows an incorrect amount",
    "I want to update my billing information",
    "Why was I charged this month?",
    "I need a copy of my invoice",
    "My payment was declined",
    "There is an unexpected charge on my account",
    "How can I change my payment method?",
    "I was charged after cancelling my subscription",
    "My credit card payment failed",
    "Please explain this transaction on my account",
    "I need help with my monthly bill",
    "The subscription price is incorrect",
    "Can I get a refund for this payment?",
    "My invoice has the wrong billing address",
    "I want to cancel my paid subscription",
    "Why did my subscription renewal cost more?",
    "I have a question about my payment",
    "The payment is showing twice",
    "Please help me with my refund"
]

technical = [
    "The application crashes when I try to login",
    "The website is not loading",
    "I cannot reset my password",
    "The mobile app keeps crashing",
    "The dashboard is showing an error",
    "The server appears to be down",
    "I cannot upload my document",
    "The application is extremely slow",
    "I am getting an error when opening the app",
    "The login page is not working",
    "My account page keeps freezing",
    "The system stopped working suddenly",
    "I cannot access the dashboard",
    "The website gives me a 500 error",
    "The app is not responding",
    "The API is returning an error",
    "I cannot download my report",
    "The page keeps timing out",
    "There is a technical problem with my account",
    "The software stopped working after the update"
]

hr = [
    "I want to know about the leave policy",
    "How many vacation days do employees receive?",
    "I need a copy of my employment letter",
    "How can I update my personal information?",
    "I have a question about my salary slip",
    "Where can I find the employee handbook?",
    "I need information about health benefits",
    "How do I apply for parental leave?",
    "I want to update my emergency contact",
    "Can you explain the company leave policy?",
    "I need help with my employee benefits",
    "How can I request time off?",
    "I have a question about my payslip",
    "Where can I download my HR documents?",
    "I need to contact the HR department",
    "How do I update my bank details?",
    "Can you help with my employment records?",
    "What is the holiday policy?",
    "I have a question regarding employee insurance",
    "How can I change my employee information?"
]

general = [
    "I need help with my account",
    "Can you tell me more about your services?",
    "Where can I find more information?",
    "I have a general question",
    "How can I contact customer support?",
    "I need assistance with something",
    "Can someone help me?",
    "I would like more information about the platform",
    "Where can I find the documentation?",
    "I have a question about the service",
    "How do I get started?",
    "Can you explain how the platform works?",
    "I need some assistance",
    "Who should I contact for help?",
    "Where can I find the user guide?",
    "I would like to know more about the product",
    "Can you help me understand the service?",
    "I have a question",
    "Please provide more information",
    "I need support with my account"
]

data = []
for text in billing:
    data.append(["Billing", text])
for text in technical:
    data.append(["Technical", text])
for text in hr:
    data.append(["HR", text])
for text in general:
    data.append(["General", text])

variations = [
    " Please help.",
    " Can you assist?",
    " This is urgent.",
    " I need assistance.",
    " Please resolve this.",
    " Can you look into this?"
]

original_data = data.copy()
for label, text in original_data:
    for variation in random.sample(variations, 2):
        data.append([label, text + variation])

random.shuffle(data)

df = pd.DataFrame(data, columns=["category", "text"])
os.makedirs("data", exist_ok=True)
df.to_csv("data/tickets.csv", index=False)

print(f"Dataset created with {len(df)} tickets.")
print(df["category"].value_counts())
