DROP TABLE IF EXISTS mimiciv_local.timebuckets;
-- Base table: patient_hours
CREATE TABLE mimiciv_local.timebuckets AS WITH patient_hours AS (
    SELECT stay_id,
        gs.tidx
    FROM mimiciv_derived.icustay_times,
        LATERAL generate_series(
            date_trunc('hour', intime_hr),
            date_trunc('hour', outtime_hr) - interval '1 hour',
            interval '1 hour'
        ) AS gs(tidx)
),
-- Vitals timebuckets
tb_vitals AS (
    SELECT time_bucket('1 hour', charttime) AS tidx,
        stay_id,
        avg(heart_rate) AS heart_rate,
        avg(sbp) AS sbp,
        avg(dbp) AS dbp,
        avg(resp_rate) as resp_rate,
        avg(temperature) as temperature,
        avg(spo2) as spo2,
        avg(glucose) as glucose
    FROM mimiciv_derived.vitalsign
    GROUP BY stay_id,
        tidx
),
-- Norepi equivalent dose timebuckets
tb_norepi_eq AS (
    SELECT n.stay_id,
        gs.tidx,
        avg(
            n.norepinephrine_equivalent_dose / EXTRACT(
                epoch
                FROM (n.endtime - n.starttime)
            )
        ) AS norepi_eq_rate
    FROM mimiciv_derived.norepinephrine_equivalent_dose n,
        LATERAL generate_series(
            date_trunc('hour', n.starttime),
            date_trunc('hour', n.endtime) - interval '1 hour',
            interval '1 hour'
        ) AS gs(tidx)
    GROUP BY stay_id,
        tidx
),
-- Ventilation timebuckets
tb_vent AS (
    SELECT stay_id,
        gs.tidx,
        avg(
            CASE
                WHEN ventilation_status = 'HFNC' THEN 1
                ELSE 0
            END
        ) AS vent_hfnc,
        avg(
            CASE
                WHEN ventilation_status = 'SupplementalOxygen' THEN 1
                ELSE 0
            END
        ) AS vent_suppo2,
        avg(
            CASE
                WHEN ventilation_status = 'NonInvasiveVent' THEN 1
                ELSE 0
            END
        ) AS vent_noninvasive,
        avg(
            CASE
                WHEN ventilation_status = 'InvasiveVent' THEN 1
                ELSE 0
            END
        ) AS vent_invasive,
        avg(
            CASE
                WHEN ventilation_status = 'Tracheostomy' THEN 1
                ELSE 0
            END
        ) AS vent_trach
    FROM mimiciv_derived.ventilation vent,
        LATERAL generate_series(
            date_trunc('hour', starttime),
            date_trunc('hour', endtime) - interval '1 hour',
            interval '1 hour'
        ) AS gs(tidx)
    GROUP BY stay_id,
        gs.tidx
),
-- Ventilator settings timebuckets
tb_vent_setting AS (
    SELECT time_bucket('1 hour', charttime) AS tidx,
        stay_id,
        avg(respiratory_rate_set) AS respiratory_rate_set,
        avg(respiratory_rate_total) AS respiratory_rate_total,
        avg(respiratory_rate_spontaneous) AS respiratory_rate_spontaneous,
        avg(minute_volume) AS minute_volume,
        avg(tidal_volume_set) as tidal_volume_set,
        avg(tidal_volume_observed) as tidal_volume_observed,
        avg(tidal_volume_spontaneous) as tidal_volume_spontaneous,
        avg(plateau_pressure) as plateau_pressure,
        avg(peep) as peep,
        avg(fio2) as fio2,
        avg(flow_rate) as flow_rate
    FROM mimiciv_derived.ventilator_setting
    GROUP BY stay_id,
        tidx
),
-- Chem linked to stay_ids
linked_chem AS (
    SELECT i.stay_id,
        c.charttime,
        c.albumin,
        c.globulin,
        c.total_protein,
        c.aniongap,
        c.bicarbonate,
        c.bun,
        c.calcium,
        c.chloride,
        c.creatinine,
        c.glucose,
        c.sodium,
        c.potassium
    FROM mimiciv_derived.icustay_detail i
        RIGHT JOIN mimiciv_derived.chemistry c ON i.subject_id = c.subject_id
        AND c.charttime < i.icu_outtime
        AND c.charttime > i.icu_intime
    WHERE stay_id IS NOT NULL
),
tb_chem AS (
    SELECT time_bucket('1 hour', charttime) AS tidx,
        stay_id,
        avg(albumin) AS albumin,
        avg(globulin) AS globulin,
        avg(total_protein) AS total_protein,
        avg(aniongap) AS aniongap,
        avg(bicarbonate) AS bicarbonate,
        avg(bun) AS bun,
        avg(calcium) AS calcium,
        avg(chloride) AS chloride,
        avg(creatinine) AS creatinine,
        avg(glucose) AS glucose,
        avg(sodium) AS sodium,
        avg(potassium) AS potassium
    FROM linked_chem
    GROUP BY stay_id,
        tidx
)
SELECT p.stay_id,
    p.tidx,
    tb_vitals.heart_rate,
    tb_vitals.sbp,
    tb_vitals.dbp,
    tb_vitals.resp_rate,
    cast(tb_vitals.temperature AS DOUBLE PRECISION),
    tb_vitals.spo2,
    -- Glucose may appear in either vitals (assume fingerstick) or chem
    -- Take chem if it's available, otherwise fingerstick
    coalesce(tb_chem.glucose, tb_vitals.glucose) AS glucose,
    cast(
        coalesce(tb_norepi_eq.norepi_eq_rate, 0.0) AS DOUBLE PRECISION
    ) AS norepi_eq_rate,
    cast(
        coalesce(tb_vent.vent_hfnc, 0.0) AS DOUBLE PRECISION
    ) AS vent_hfnc,
    -- TODO: potentially coalesce this with vent_setting.fio2?
    cast(
        coalesce(tb_vent.vent_suppo2, 0.0) AS DOUBLE PRECISION
    ) AS vent_suppo2,
    cast(
        coalesce(tb_vent.vent_noninvasive, 0.0) AS DOUBLE PRECISION
    ) AS vent_noninvasive,
    cast(
        coalesce(tb_vent.vent_invasive, 0.0) AS DOUBLE PRECISION
    ) AS vent_invasive,
    cast(
        coalesce(tb_vent.vent_trach, 0.0) AS DOUBLE PRECISION
    ) AS vent_trach,
    tb_vent_setting.respiratory_rate_set,
    tb_vent_setting.respiratory_rate_total,
    tb_vent_setting.respiratory_rate_spontaneous,
    tb_vent_setting.minute_volume,
    tb_vent_setting.tidal_volume_set,
    tb_vent_setting.tidal_volume_observed,
    tb_vent_setting.tidal_volume_spontaneous,
    tb_vent_setting.plateau_pressure,
    tb_vent_setting.peep,
    tb_vent_setting.fio2,
    tb_vent_setting.flow_rate,
    tb_chem.albumin,
    tb_chem.globulin,
    tb_chem.total_protein,
    tb_chem.aniongap,
    tb_chem.bicarbonate,
    tb_chem.bun,
    tb_chem.calcium,
    tb_chem.chloride,
    tb_chem.creatinine,
    -- tb_chem.glucose,
    tb_chem.sodium,
    tb_chem.potassium
FROM patient_hours p
    LEFT JOIN tb_vitals ON p.stay_id = tb_vitals.stay_id
    AND p.tidx = tb_vitals.tidx
    LEFT JOIN tb_norepi_eq ON p.stay_id = tb_norepi_eq.stay_id
    AND p.tidx = tb_norepi_eq.tidx
    LEFT JOIN tb_vent ON p.stay_id = tb_vent.stay_id
    AND p.tidx = tb_vent.tidx
    LEFT JOIN tb_vent_setting ON p.stay_id = tb_vent_setting.stay_id
    AND p.tidx = tb_vent_setting.tidx
    LEFT JOIN tb_chem ON p.stay_id = tb_chem.stay_id
    AND p.tidx = tb_chem.tidx;
CREATE UNIQUE INDEX IF NOT EXISTS idx_stay_id_timestamp ON mimiciv_local.timebuckets(stay_id, tidx);