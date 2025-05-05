DROP TABLE IF EXISTS mimiciv_local.tidx_encoding;
CREATE TABLE mimiciv_local.tidx_encoding AS(
    WITH tidx_min AS (
        SELECT stay_id,
            min(time_bucket('1 hour', charttime)) AS min_charthour
        FROM mimiciv_icu.chartevents
        GROUP BY stay_id
    ),
    charthours AS (
        SELECT stay_id,
            time_bucket('1 hour', charttime) AS charthour
        FROM mimiciv_icu.chartevents
        GROUP BY stay_id,
            time_bucket('1 hour', charttime)
    )
    SELECT charthours.stay_id,
        charthours.charthour,
        cast(
            extract(
                epoch
                FROM (
                        charthours.charthour - tidx_min.min_charthour
                    ) / 3600
            ) AS INT
        ) AS tidx
    FROM charthours
        LEFT JOIN tidx_min ON tidx_min.stay_id = charthours.stay_id
    ORDER BY stay_id,
        charthour
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_stay_id_tidx ON mimiciv_local.tidx_encoding(stay_id, tidx);