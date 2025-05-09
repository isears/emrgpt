DROP TABLE IF EXISTS mimiciv_local.sequences;
CREATE TABLE mimiciv_local.sequences AS (
    WITH encoded_events AS (
        SELECT ce.stay_id,
            CASE
                -- Have to handle special case where charttime < icu_intime or > icu_outtime:
                -- Set all events prior to beginning of ICU stay to tidx 0
                WHEN time_bucket('1 hour', ce.charttime) < time_bucket('1 hour', icustays.icu_intime) THEN 0 -- Set all events after the end of the ICU stay to max tidx
                WHEN time_bucket('1 hour', ce.charttime) > time_bucket('1 hour', icustays.icu_outtime) THEN cast(
                    extract(
                        epoch
                        FROM (
                                time_bucket('1 hour', icustays.icu_outtime) - time_bucket('1 hour', icustays.icu_intime)
                            ) / 3600
                    ) AS INT
                )
                ELSE cast(
                    extract(
                        epoch
                        FROM (
                                time_bucket('1 hour', ce.charttime) - time_bucket('1 hour', icustays.icu_intime)
                            ) / 3600
                    ) AS INT
                )
            END AS tidx,
            enc.encoding,
            coalesce(ce.valuenum, 1.0) AS valuenum
        FROM mimiciv_icu.chartevents ce
            LEFT JOIN mimiciv_local.item_encoding enc ON ce.itemid = enc.itemid
            AND (
                ce.value = enc.value
                OR enc.value = 'NUMERIC'
            )
            LEFT JOIN mimiciv_derived.icustay_detail icustays ON ce.stay_id = icustays.stay_id
        ORDER BY stay_id,
            tidx
    ),
    -- Need to merge with tidx_encoding to ensure there are no tidx gaps
    encoded_events_complete AS (
        SELECT te.stay_id,
            te.tidx,
            -- Special event_id '0' (with value 0) for no events happened during tidx
            coalesce(ee.encoding, 0) AS encoding,
            coalesce(ee.valuenum, 0.0) AS valuenum
        FROM encoded_events ee
            FULL OUTER JOIN mimiciv_local.tidx_encoding te ON (
                ee.stay_id = te.stay_id
                AND ee.tidx = te.tidx
            )
    ),
    aggregations AS (
        SELECT stay_id,
            array_agg(encoding) AS encodings,
            array_agg(valuenum) AS
        values,
            array_agg(tidx) AS tidx
        FROM encoded_events_complete
            /*
             Filter out encodings b/c low-prevalence events never received an encoding
             Filter out null stay_ids, which are stay_ids that appeared in tidx_encoding but not events
             i.e. stays with entry in icustays but no chartevents recorded
             */
        WHERE encoding IS NOT NULL
            AND stay_id IS NOT NULL
        GROUP BY stay_id
    )
    SELECT stay_id,
        encodings,
        values,
        tidx,
        (
            SELECT array_agg(i - 1)
            FROM generate_subscripts(tidx, 1) AS i
            WHERE i = 1
                OR tidx [i] <> tidx [i-1]
        ) AS offsets,
        random() * 100 > 90 AS testset
    FROM aggregations
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_stay_id ON mimiciv_local.sequences(stay_id);