"""
clinical/sdoh/train_risk_model.py — One-shot training script.

1. Generates a 500-patient × 12-month synthetic dataset.
2. Optionally seeds the patient_store SQLite DB with the synthetic records.
3. Trains the GradientBoostingClassifier and saves it to risk_model.joblib.
4. Prints a classification report.

Usage:
    python -m clinical.sdoh.train_risk_model
    python -m clinical.sdoh.train_risk_model --no-seed-db
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SDOH risk trajectory model.")
    parser.add_argument("--no-seed-db", action="store_true",
                        help="Skip seeding the patient_store SQLite DB.")
    parser.add_argument("--n-patients", type=int, default=500)
    parser.add_argument("--n-months",   type=int, default=12)
    args = parser.parse_args()

    print("[*] Generating synthetic patient dataset ...")
    from clinical.sdoh.patient_simulator import generate_dataset
    df = generate_dataset(n_patients=args.n_patients, n_months=args.n_months)

    total_records = len(df)
    n_patients    = df["patient_id"].nunique()
    print(f"  Generated {total_records:,} visit records for {n_patients} patients.")
    print("  Label distribution:")
    for label, count in df["sdoh_risk_label"].value_counts().items():
        pct = count / total_records * 100
        print(f"    {label:<10} {count:>6,}  ({pct:.1f}%)")

    if not args.no_seed_db:
        print("\n[*] Seeding patient_store SQLite DB ...")
        from clinical.sdoh.patient_store import init_db, bulk_save
        init_db()
        saved = bulk_save(df.to_dict(orient="records"))
        print(f"  Saved {saved:,} records to sdoh_patients.db")

    print("\n[*] Training GradientBoostingClassifier ...")
    from clinical.sdoh.risk_model import train
    bundle = train(df)

    # Print a full sklearn classification report
    from sklearn.metrics import classification_report

    clf = bundle["model"]
    le  = bundle["label_encoder"]
    FEATURES = bundle["features"]
    X = df[FEATURES].fillna(0).values
    y_true = le.transform(df["sdoh_risk_label"].values)
    y_pred = clf.predict(X)

    print("\n[*] Classification Report (in-sample):")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    print("\n[OK] Training complete. Model saved to clinical/sdoh/risk_model.joblib")


if __name__ == "__main__":
    main()
