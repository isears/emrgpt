DROP TABLE IF EXISTS mimiciv_local.tidx_encoding;
CREATE TABLE mimiciv_local.tidx_encoding AS(
    WITH charthours AS (
        SELECT stay_id,
            generate_series(
                time_bucket('1 hour', icustays.icu_intime),
                time_bucket('1 hour', icustays.icu_outtime),
                INTERVAL '1 hour'
            ) AS charthour,
            time_bucket('1 hour', icu_intime) AS firsthour
        FROM mimiciv_derived.icustay_detail icustays
    )
    SELECT stay_id,
        charthour,
        cast(
            extract(
                EPOCH
                FROM (charthour - firsthour)
            ) / 3600 AS INT
        ) AS tidx
    FROM charthours
    ORDER BY stay_id,
        charthour
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_stay_id_tidx ON mimiciv_local.tidx_encoding(stay_id, tidx);