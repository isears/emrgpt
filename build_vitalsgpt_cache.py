import sqlalchemy
import pandas as pd
import psycopg2


SAMPLE_SIZE = 1e6
GRANULARITY_SECONDS = 60 * 60

con = sqlalchemy.create_engine(
    "postgresql+psycopg2://",
    creator=lambda: psycopg2.connect(database="mimiciv", user="readonly"),
)

icustays_df = pd.read_sql(
    "SELECT stay_id, icu_intime FROM mimiciv_derived.icustay_detail;",
    con=con,
)

if SAMPLE_SIZE is not None:

    vitals_df = pd.read_sql(
        f"SELECT * FROM mimiciv_derived.vitalsign LIMIT {SAMPLE_SIZE};", con=con
    )
else:
    vitals_df = pd.read_sql(f"SELECT * FROM mimiciv_derived.vitalsign;", con=con)

vitals_df = (
    vitals_df[
        [
            "subject_id",
            "stay_id",
            "charttime",
            "heart_rate",
            "sbp",
            "dbp",
            "resp_rate",
            "temperature",
            "spo2",
            "glucose",
        ]
    ]
    .melt(
        id_vars=["subject_id", "stay_id", "charttime"],
        var_name="event_type",
        value_name="event_value",
    )
    .dropna(subset="event_value")
)

vitals_df = pd.merge(vitals_df, icustays_df, on="stay_id", how="left")
vitals_df["tidx"] = vitals_df.apply(
    lambda r: (r["charttime"] - r["icu_intime"]).total_seconds() // GRANULARITY_SECONDS,
    axis=1,
).astype(int)
vitals_df = vitals_df[vitals_df["tidx"] >= 0]
vitals_df = vitals_df.drop(columns=["charttime", "icu_intime"])
vitals_df = vitals_df.sort_values(
    ["subject_id", "stay_id", "tidx", "event_type"], ascending=True
)

print(vitals_df.info())

vitals_df.to_parquet("cache/vitals.parquet")
