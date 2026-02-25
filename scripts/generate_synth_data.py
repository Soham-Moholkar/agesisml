"""Generate synthetic datasets for AegisML demos.

SYNTHETIC DATA JUSTIFICATION:
These datasets are synthetic (artificially generated) for the following reasons:
1. Avoid copyright/licensing issues with real-world datasets
2. Ensure reproducibility with fixed random seeds
3. Enable offline demos without internet access
4. Control feature distributions to showcase all model capabilities

The synthetic data mimics realistic patterns found in:
- Credit card fraud detection (imbalanced binary classification)
- Sentiment analysis (text classification)
- Healthcare triage (multiclass classification)
"""
import numpy as np
import pandas as pd
import os

SEED = 42
np.random.seed(SEED)

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)


def generate_tabular_fraud():
    """Generate synthetic fraud detection dataset.
    Features mimic credit/loan application data.
    ~10% fraud rate (imbalanced).
    """
    n = 2000
    np.random.seed(SEED)

    data = {
        "age": np.random.randint(18, 75, n),
        "income": np.random.lognormal(10.5, 0.8, n).astype(int),
        "credit_score": np.random.normal(680, 80, n).clip(300, 850).astype(int),
        "loan_amount": np.random.lognormal(9, 1.2, n).astype(int),
        "employment_years": np.random.exponential(5, n).clip(0, 40).round(1),
        "num_accounts": np.random.poisson(3, n),
        "num_late_payments": np.random.poisson(1, n),
        "debt_to_income": np.random.uniform(0.05, 0.8, n).round(3),
        "has_mortgage": np.random.binomial(1, 0.4, n),
        "has_dependents": np.random.binomial(1, 0.5, n),
        "education_level": np.random.choice(["high_school", "bachelors", "masters", "phd"], n, p=[0.3, 0.4, 0.2, 0.1]),
        "region": np.random.choice(["north", "south", "east", "west"], n),
    }

    # Create fraud label with realistic pattern
    fraud_score = (
        -0.02 * data["credit_score"]
        + 0.5 * data["num_late_payments"]
        + 2.0 * data["debt_to_income"]
        - 0.1 * data["employment_years"]
        + 0.3 * (data["loan_amount"] / data["income"])
        + np.random.normal(0, 0.5, n)
    )
    threshold = np.percentile(fraud_score, 90)
    data["is_fraud"] = (fraud_score > threshold).astype(int)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATASETS_DIR, "tabular_sample.csv"), index=False)
    print(f"✓ tabular_sample.csv: {len(df)} rows, fraud rate: {df['is_fraud'].mean():.1%}")
    return df


def generate_text_sentiment():
    """Generate synthetic sentiment dataset."""
    np.random.seed(SEED)

    positive_templates = [
        "This product is amazing and I love it!",
        "Great service, very satisfied with the quality.",
        "Excellent experience, would recommend to everyone.",
        "The team did a fantastic job, very professional.",
        "I'm really happy with my purchase, exceeded expectations.",
        "Outstanding customer support, they resolved my issue quickly.",
        "Best value for money, this is exactly what I needed.",
        "Really impressed by the quality and attention to detail.",
        "Love the design and functionality, perfect product.",
        "Wonderful experience from start to finish, highly recommended.",
        "The quality is superb and the delivery was fast.",
        "Very pleased with the results, will buy again.",
        "Incredible performance, this exceeded all my expectations.",
        "Friendly staff and amazing ambiance, truly enjoyable.",
        "Everything was perfect, couldn't ask for more.",
    ]

    negative_templates = [
        "Terrible product, waste of money, avoid at all costs.",
        "Very disappointed with the quality, fell apart quickly.",
        "Poor customer service, nobody responded to my complaint.",
        "The worst experience I've ever had, completely unacceptable.",
        "Not worth the price, misleading description and bad quality.",
        "Extremely frustrated with delayed delivery and damaged product.",
        "Horrible quality control, this is defective and unusable.",
        "Never buying from them again, total scam operation.",
        "The product broke after one use, cheap and fragile.",
        "Rude staff and terrible food, would not recommend to anyone.",
        "Complete waste of time and money, very disappointing.",
        "False advertising, the product looks nothing like the picture.",
        "Overpriced and underwhelming, expected much better quality.",
        "Awful experience, long wait times and incompetent staff.",
        "Product arrived broken and customer service was unhelpful.",
    ]

    modifiers = [
        "Honestly, ", "To be fair, ", "I have to say, ", "In my experience, ",
        "After careful consideration, ", "Without a doubt, ", "", "",
        "Update: ", "Just wanted to say, ", "For what it's worth, ",
    ]

    rows = []
    for _ in range(500):
        mod = np.random.choice(modifiers)
        text = mod + np.random.choice(positive_templates)
        # Add some noise words
        if np.random.random() < 0.3:
            text += " " + np.random.choice(["Overall good.", "Worth it.", "5 stars.", "Thumbs up."])
        rows.append({"text": text, "label": "positive"})

    for _ in range(500):
        mod = np.random.choice(modifiers)
        text = mod + np.random.choice(negative_templates)
        if np.random.random() < 0.3:
            text += " " + np.random.choice(["Never again.", "0 stars.", "Avoid.", "Terrible."])
        rows.append({"text": text, "label": "negative"})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv(os.path.join(DATASETS_DIR, "text_sample.csv"), index=False)
    print(f"✓ text_sample.csv: {len(df)} rows, balanced sentiment")
    return df


def generate_healthcare():
    """Generate synthetic healthcare triage dataset.
    Multiclass: Low Risk, Medium Risk, High Risk, Critical.
    """
    n = 1500
    np.random.seed(SEED)

    data = {
        "age": np.random.randint(1, 95, n),
        "heart_rate": np.random.normal(80, 20, n).clip(40, 180).astype(int),
        "blood_pressure_sys": np.random.normal(120, 25, n).clip(70, 220).astype(int),
        "blood_pressure_dia": np.random.normal(80, 15, n).clip(40, 130).astype(int),
        "temperature": np.random.normal(98.6, 1.5, n).round(1),
        "oxygen_saturation": np.random.normal(96, 3, n).clip(70, 100).round(1),
        "respiratory_rate": np.random.normal(16, 4, n).clip(8, 40).astype(int),
        "pain_level": np.random.randint(0, 11, n),
        "has_chronic_condition": np.random.binomial(1, 0.3, n),
        "bmi": np.random.normal(26, 5, n).clip(15, 50).round(1),
        "smoker": np.random.binomial(1, 0.2, n),
        "diabetes": np.random.binomial(1, 0.15, n),
    }

    # Risk scoring
    risk_score = (
        0.02 * data["age"]
        + 0.01 * np.abs(data["heart_rate"] - 75)
        + 0.015 * np.abs(data["blood_pressure_sys"] - 120)
        + 0.1 * np.abs(data["temperature"] - 98.6)
        + 0.05 * (100 - data["oxygen_saturation"])
        + 0.03 * np.abs(data["respiratory_rate"] - 16)
        + 0.1 * data["pain_level"]
        + 0.5 * data["has_chronic_condition"]
        + 0.3 * data["smoker"]
        + 0.4 * data["diabetes"]
        + np.random.normal(0, 0.5, n)
    )

    conditions = [
        risk_score < np.percentile(risk_score, 30),
        (risk_score >= np.percentile(risk_score, 30)) & (risk_score < np.percentile(risk_score, 60)),
        (risk_score >= np.percentile(risk_score, 60)) & (risk_score < np.percentile(risk_score, 85)),
        risk_score >= np.percentile(risk_score, 85),
    ]
    choices = ["low_risk", "medium_risk", "high_risk", "critical"]
    data["risk_level"] = np.select(conditions, choices, default="medium_risk")

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATASETS_DIR, "healthcare_synth.csv"), index=False)
    print(f"✓ healthcare_synth.csv: {len(df)} rows, {df['risk_level'].value_counts().to_dict()}")
    return df


def generate_campus_feedback():
    """Generate synthetic campus feedback dataset for transfer learning demo."""
    np.random.seed(SEED + 1)

    positive = [
        "The campus library is very well-maintained and quiet.",
        "Professors are very supportive and approachable.",
        "Great extracurricular activities and student clubs.",
        "The food court has excellent variety and quality.",
        "Lab facilities are modern and well-equipped.",
        "The placement cell does a great job connecting with companies.",
        "Sports facilities are top-notch and well-organized.",
        "The campus is beautiful with lots of green spaces.",
        "WiFi connectivity has improved significantly this semester.",
        "The mentoring program really helped me grow academically.",
    ]
    negative = [
        "The hostel conditions are terrible and unhygienic.",
        "Too much academic pressure with no proper guidance.",
        "Administration is slow and unresponsive to complaints.",
        "The canteen food quality has deteriorated significantly.",
        "Lack of industry-relevant courses in the curriculum.",
        "Parking is a nightmare, no space available for students.",
        "The labs are outdated and poorly maintained.",
        "Class sizes are too large for effective learning.",
        "The fee structure is unreasonable and non-transparent.",
        "No proper mental health support for students.",
    ]

    rows = []
    for _ in range(150):
        rows.append({"text": np.random.choice(positive), "label": "positive"})
    for _ in range(150):
        rows.append({"text": np.random.choice(negative), "label": "negative"})

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv(os.path.join(DATASETS_DIR, "campus_feedback.csv"), index=False)
    print(f"✓ campus_feedback.csv: {len(df)} rows (transfer learning target)")
    return df


if __name__ == "__main__":
    print("Generating synthetic datasets for AegisML...\n")
    generate_tabular_fraud()
    generate_text_sentiment()
    generate_healthcare()
    generate_campus_feedback()
    print("\n✓ All datasets generated successfully!")
