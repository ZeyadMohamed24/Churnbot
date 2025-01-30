from gliner import GLiNER
from .utils import clean_numeric_value
import re


def extract_specific_features(user_text):
    """Extracts key customer features from text using GLiNER model with enhanced performance and accuracy."""

    SUBSCRIPTION_PATTERN = re.compile(r"\b(standard|basic|premium)\b", re.IGNORECASE)
    SPEND_PATTERN = re.compile(
        r"(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\s*(?:dollars?|usd|€|\$|euro|a year|per year)",
        re.IGNORECASE,
    )
    CONTRACT_PATTERN = re.compile(r"\b(annual|monthly|quarterly)\b", re.IGNORECASE)

    ACCEPTED_VALUES = {
        "Age": "numeric",
        "Gender": {"male", "female"},
        "Tenure": "numeric",
        "Usage Frequency": "numeric",
        "Support Calls": "numeric",
        "Payment Delay": "numeric",
        "Subscription Type": {"standard", "basic", "premium"},
        "Contract Length": {"annual", "monthly", "quarterly"},
        "Total Spend": "numeric",
        "Last Interaction": "numeric",
    }
    model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
    labels = list(ACCEPTED_VALUES.keys())
    truncated_text = user_text[:512]  
    entities = model.predict_entities(truncated_text, labels, threshold=0.3)

    features = {label: "none" for label in labels}

    for entity in entities:
        feature_name = entity["label"]
        feature_value = entity["text"].strip().lower()

        if feature_name in ACCEPTED_VALUES:
            accepted = ACCEPTED_VALUES[feature_name]
            if accepted == "numeric":
                cleaned_value = clean_numeric_value(feature_value)
                if cleaned_value != "none":
                    features[feature_name] = cleaned_value
            elif feature_value in accepted:
                features[feature_name] = feature_value.capitalize()

    if features["Subscription Type"] == "none":
        match = SUBSCRIPTION_PATTERN.search(user_text)
        if match:
            features["Subscription Type"] = match.group().capitalize()

    if features["Total Spend"] == "none":
        match = SPEND_PATTERN.search(user_text)
        if match:
            features["Total Spend"] = float(match.group(1).replace(",", ""))

    if features["Contract Length"] == "none":
        match = CONTRACT_PATTERN.search(user_text)
        if match:
            features["Contract Length"] = match.group().capitalize()

    return features
