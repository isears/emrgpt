DROP TABLE IF EXISTS mimiciv_local.sequences;
CREATE TABLE mimiciv_local.sequences AS (
    WITH tidx_min AS (
        SELECT stay_id,
            min(time_bucket('1 hour', charttime)) AS min_charthour
        FROM mimiciv_icu.chartevents
        GROUP BY stay_id
    ),
    encoded_events AS (
        SELECT ce.stay_id,
            cast(
                extract(
                    epoch
                    FROM (
                            time_bucket('1 hour', ce.charttime) - tm.min_charthour
                        ) / 3600
                ) AS INT
            ) AS tidx,
            enc.encoding,
            coalesce(ce.valuenum, 1.0) AS valuenum
        FROM mimiciv_icu.chartevents ce
            LEFT JOIN mimiciv_local.item_encoding enc ON ce.itemid = enc.itemid
            AND (
                ce.value = enc.value
                OR enc.value = 'NUMERIC'
            )
            LEFT JOIN tidx_min tm ON ce.stay_id = tm.stay_id
    ),
    aggregations AS (
        SELECT stay_id,
            array_agg(encoding) AS encodings,
            array_agg(valuenum) AS
        values,
            array_agg(tidx) AS tidx -- Need to filter out null encodings b/c low-prevalence events never received an encoding
        FROM encoded_events
        WHERE encoding IS NOT NULL
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