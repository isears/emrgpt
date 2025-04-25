DROP TABLE IF EXISTS mimiciv_local.timebuckets;
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
)
SELECT p.stay_id,
    p.tidx,
    tb_vitals.heart_rate,
    tb_vitals.sbp,
    tb_vitals.dbp,
    tb_vitals.resp_rate,
    cast(tb_vitals.temperature AS DOUBLE PRECISION),
    tb_vitals.spo2,
    tb_vitals.glucose,
    cast(
        coalesce(tb_norepi_eq.norepi_eq_rate, 0.0) AS DOUBLE PRECISION
    ) AS norepi_eq_rate,
    cast(
        coalesce(tb_vent.vent_hfnc, 0.0) AS DOUBLE PRECISION
    ) AS vent_hfnc,
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
    tb_vent_setting.flow_rate
FROM patient_hours p
    LEFT JOIN tb_vitals ON p.stay_id = tb_vitals.stay_id
    AND p.tidx = tb_vitals.tidx
    LEFT JOIN tb_norepi_eq ON p.stay_id = tb_norepi_eq.stay_id
    AND p.tidx = tb_norepi_eq.tidx
    LEFT JOIN tb_vent ON p.stay_id = tb_vent.stay_id
    AND p.tidx = tb_vent.tidx
    LEFT JOIN tb_vent_setting ON p.stay_id = tb_vent_setting.stay_id
    AND p.tidx = tb_vent_setting.tidx;
CREATE UNIQUE INDEX IF NOT EXISTS idx_stay_id_timestamp ON mimiciv_local.timebuckets(stay_id, tidx);