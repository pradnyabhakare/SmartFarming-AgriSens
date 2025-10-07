import numpy as np

def recommend_crops(model, sample_df, label_encoder, features, top_n_high=3, top_n_mod=2):
    """
    Recommend crops based on model prediction probabilities.
    """
    # Ensure correct feature order
    sample_df = sample_df[features]

    # Predict probabilities
    probs = model.predict_proba(sample_df)[0]
    class_order = model.classes_.astype(int)

    # Sort by probability
    sorted_pos = probs.argsort()[::-1]

    # Top highly and moderately recommended
    high_pos = sorted_pos[:top_n_high]
    mod_pos = sorted_pos[top_n_high: top_n_high + top_n_mod]

    high_labels = class_order[high_pos]
    mod_labels = class_order[mod_pos]

    high_crops = label_encoder.inverse_transform(high_labels)
    mod_crops = label_encoder.inverse_transform(mod_labels)

    high_probs = probs[high_pos]
    mod_probs = probs[mod_pos]

    result = {
        "Highly Recommended": list(zip(high_crops, high_probs)),
        "Moderately Recommended": list(zip(mod_crops, mod_probs))
    }

    # ✅ Debug: print result in console
    print("🔹 Recommend Crops Result:", result)

    # Optional: you can also return it normally
    return result
