DROP TABLE IF EXISTS mimiciv_local.tidx_encoding;
CREATE TABLE mimiciv_local.tidx_encoding AS(
    WITH charthours AS (
        SELECT stay_id,
            time_bucket('1 hour', charttime) AS charthour
        FROM mimiciv_icu.chartevents
        GROUP BY stay_id,
            time_bucket('1 hour', charttime)
    )
    SELECT charthours.stay_id,
        -- Some events occur before the beginning of the ICU stay
        -- Corrected here by just setting the event time to the icu_intime
        CASE
            WHEN charthours.charthour >= time_bucket('1 hour', icustays.icu_intime) THEN charthours.charthour
            ELSE time_bucket('1 hour', icustays.icu_intime)
        END AS charthour,
        cast(
            extract(
                epoch
                FROM (
                        charthours.charthour - time_bucket('1 hour', icustays.icu_intime)
                    ) / 3600
            ) AS INT
        ) AS tidx
    FROM charthours
        LEFT JOIN mimiciv_derived.icustay_detail icustays ON icustays.stay_id = charthours.stay_id
    ORDER BY stay_id,
        charthour
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_stay_id_tidx ON mimiciv_local.tidx_encoding(stay_id, tidx);