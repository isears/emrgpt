DROP TABLE IF EXISTS mimiciv_local.val_ventilation;
CREATE TABLE mimiciv_local.val_ventilation AS (
    WITH first_vent AS(
        SELECT stay_id,
            min(time_bucket('1 hour', starttime)) AS charthour
        FROM mimiciv_derived.ventilation
        WHERE ventilation_status = 'InvasiveVent'
            OR ventilation_status = 'Tracheostomy'
        GROUP BY stay_id
    ),
    sid_firstlasthour AS (
        SELECT stay_id,
            min(charthour) AS firsthour,
            max(tidx) AS max_tidx
        FROM mimiciv_local.tidx_encoding
        GROUP BY stay_id
    ),
    included_sid AS (
        SELECT sid_firstlasthour.stay_id,
            sid_firstlasthour.firsthour AS firsthour,
            sid_firstlasthour.max_tidx AS max_tidx,
            first_vent.charthour AS firstventhour
        FROM sid_firstlasthour
            LEFT JOIN mimiciv_local.sequences ON sid_firstlasthour.stay_id = mimiciv_local.sequences.stay_id
            LEFT JOIN first_vent ON first_vent.stay_id = sid_firstlasthour.stay_id -- Inclusion criteria:
            --  - In test set
            --  - Stay longer than 24 hrs
            --  - Either no ventilation or ventilation happened after 24 hrs into ICU stay
        WHERE mimiciv_local.sequences.testset = true
            AND (
                first_vent.charthour > (
                    sid_firstlasthour.firsthour + INTERVAL '24 hours'
                )
                OR first_vent.charthour IS NULL
            )
            AND sid_firstlasthour.max_tidx >= 24
    )
    SELECT included_sid.stay_id,
        mimiciv_local.tidx_encoding.charthour,
        mimiciv_local.tidx_encoding.tidx AS base_tidx,
        included_sid.firstventhour AS firstventhour,
        coalesce(
            -- Label points that are within 36 hrs of intubation as 'positive' b/c the model will have 24 of those hours
            mimiciv_local.tidx_encoding.charthour >= (included_sid.firstventhour - INTERVAL '36 hours'),
            false
        ) AS vent_initiation_36h
    FROM mimiciv_local.tidx_encoding
        RIGHT JOIN included_sid ON included_sid.stay_id = mimiciv_local.tidx_encoding.stay_id
    WHERE (
            mimiciv_local.tidx_encoding.charthour < (included_sid.firstventhour - INTERVAL '24 hours')
            OR included_sid.firstventhour IS NULL
        ) -- Drop all timepoints that won't give us a full 24 hr block
        AND mimiciv_local.tidx_encoding.tidx < (included_sid.max_tidx - 24)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_stay_id ON mimiciv_local.val_ventilation(stay_id, base_tidx);