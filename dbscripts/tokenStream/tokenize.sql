WITH vitalsign_tokenized AS (
    SELECT mimiciv_derived.vitalsign.stay_id AS stay_id,
        mimiciv_derived.vitalsign.charttime AS charttime,
        tokens.token_label AS token_label,
        tokens.token_value AS token_value
    FROM mimiciv_derived.vitalsign
        JOIN LATERAL (
            VALUES (
                    'vitalsign.heart_rate',
                    mimiciv_derived.vitalsign.heart_rate,
                    mimiciv_derived.vitalsign.heart_rate IS NULL
                ),
                (
                    'vitalsign.sbp',
                    mimiciv_derived.vitalsign.sbp,
                    mimiciv_derived.vitalsign.sbp IS NULL
                ),
                (
                    'vitalsign.dbp',
                    mimiciv_derived.vitalsign.dbp,
                    mimiciv_derived.vitalsign.dbp IS NULL
                ),
                (
                    'vitalsign.resp_rate',
                    mimiciv_derived.vitalsign.resp_rate,
                    mimiciv_derived.vitalsign.resp_rate IS NULL
                ),
                (
                    'vitalsign.spo2',
                    mimiciv_derived.vitalsign.spo2,
                    mimiciv_derived.vitalsign.spo2 IS NULL
                ),
                (
                    'vitalsign.glucose',
                    mimiciv_derived.vitalsign.glucose,
                    mimiciv_derived.vitalsign.glucose IS NULL
                ),
                (
                    concat(
                        'vitalsign.temperature.',
                        CAST(
                            mimiciv_derived.vitalsign.temperature_site AS VARCHAR
                        )
                    ),
                    mimiciv_derived.vitalsign.temperature,
                    mimiciv_derived.vitalsign.temperature IS NULL
                )
        ) AS tokens (token_label, token_value, token_null) ON true
    WHERE NOT tokens.token_null
),
crrt_tokenized AS (
    SELECT mimiciv_derived.crrt.stay_id AS stay_id,
        mimiciv_derived.crrt.charttime AS charttime,
        tokens.token_label AS token_label,
        tokens.token_value AS token_value
    FROM mimiciv_derived.crrt
        JOIN LATERAL (
            VALUES (
                    'crrt.access_pressure',
                    mimiciv_derived.crrt.access_pressure,
                    mimiciv_derived.crrt.access_pressure IS NULL
                ),
                (
                    'crrt.blood_flow',
                    mimiciv_derived.crrt.blood_flow,
                    mimiciv_derived.crrt.blood_flow IS NULL
                ),
                (
                    'crrt.citrate',
                    mimiciv_derived.crrt.citrate,
                    mimiciv_derived.crrt.citrate IS NULL
                ),
                (
                    'crrt.current_goal',
                    mimiciv_derived.crrt.current_goal,
                    mimiciv_derived.crrt.current_goal IS NULL
                ),
                (
                    'crrt.dialysate_rate',
                    mimiciv_derived.crrt.dialysate_rate,
                    mimiciv_derived.crrt.dialysate_rate IS NULL
                ),
                (
                    'crrt.effluent_pressure',
                    mimiciv_derived.crrt.effluent_pressure,
                    mimiciv_derived.crrt.effluent_pressure IS NULL
                ),
                (
                    'crrt.filter_pressure',
                    mimiciv_derived.crrt.filter_pressure,
                    mimiciv_derived.crrt.filter_pressure IS NULL
                ),
                (
                    'crrt.heparin_dose',
                    mimiciv_derived.crrt.heparin_dose,
                    mimiciv_derived.crrt.heparin_dose IS NULL
                ),
                (
                    'crrt.hourly_patient_fluid_removal',
                    mimiciv_derived.crrt.hourly_patient_fluid_removal,
                    mimiciv_derived.crrt.hourly_patient_fluid_removal IS NULL
                ),
                (
                    'crrt.prefilter_replacement_rate',
                    mimiciv_derived.crrt.prefilter_replacement_rate,
                    mimiciv_derived.crrt.prefilter_replacement_rate IS NULL
                ),
                (
                    'crrt.postfilter_replacement_rate',
                    mimiciv_derived.crrt.postfilter_replacement_rate,
                    mimiciv_derived.crrt.postfilter_replacement_rate IS NULL
                ),
                (
                    'crrt.replacement_rate',
                    mimiciv_derived.crrt.replacement_rate,
                    mimiciv_derived.crrt.replacement_rate IS NULL
                ),
                (
                    'crrt.return_pressure',
                    mimiciv_derived.crrt.return_pressure,
                    mimiciv_derived.crrt.return_pressure IS NULL
                ),
                (
                    'crrt.ultrafiltrate_output',
                    mimiciv_derived.crrt.ultrafiltrate_output,
                    mimiciv_derived.crrt.ultrafiltrate_output IS NULL
                ),
                (
                    concat(
                        'crrt.crrt_mode.',
                        CAST(mimiciv_derived.crrt.crrt_mode AS TEXT)
                    ),
                    NULL,
                    mimiciv_derived.crrt.crrt_mode IS NULL
                ),
                (
                    concat(
                        'crrt.dialysate_fluid.',
                        CAST(mimiciv_derived.crrt.dialysate_fluid AS TEXT)
                    ),
                    NULL,
                    mimiciv_derived.crrt.dialysate_fluid IS NULL
                ),
                (
                    concat(
                        'crrt.heparin_concentration.',
                        CAST(
                            mimiciv_derived.crrt.heparin_concentration AS TEXT
                        )
                    ),
                    NULL,
                    mimiciv_derived.crrt.heparin_concentration IS NULL
                ),
                (
                    concat(
                        'crrt.replacement_fluid.',
                        CAST(mimiciv_derived.crrt.replacement_fluid AS TEXT)
                    ),
                    NULL,
                    mimiciv_derived.crrt.replacement_fluid IS NULL
                ),
                (
                    concat(
                        'crrt.system_active.',
                        CAST(mimiciv_derived.crrt.system_active AS TEXT)
                    ),
                    NULL,
                    mimiciv_derived.crrt.system_active IS NULL
                ),
                (
                    concat(
                        'crrt.clots.',
                        CAST(mimiciv_derived.crrt.clots AS TEXT)
                    ),
                    NULL,
                    mimiciv_derived.crrt.clots IS NULL
                ),
                (
                    concat(
                        'crrt.clots_increasing.',
                        CAST(mimiciv_derived.crrt.clots_increasing AS TEXT)
                    ),
                    NULL,
                    mimiciv_derived.crrt.clots_increasing IS NULL
                ),
                (
                    concat(
                        'crrt.clotted.',
                        CAST(mimiciv_derived.crrt.clotted AS TEXT)
                    ),
                    NULL,
                    mimiciv_derived.crrt.clotted IS NULL
                )
        ) AS tokens (token_label, token_value, token_null) ON true
    WHERE NOT tokens.token_null
),
norepinephrine_equivalent_dose_tokenized AS (
    SELECT mimiciv_derived.norepinephrine_equivalent_dose.stay_id AS stay_id,
        tokens.charttime AS charttime,
        tokens.token_label AS token_label,
        tokens.token_value AS token_value
    FROM mimiciv_derived.norepinephrine_equivalent_dose
        JOIN LATERAL (
            VALUES (
                    mimiciv_derived.norepinephrine_equivalent_dose.starttime,
                    'norepinephrine_equivalent_dose.rate',
                    mimiciv_derived.norepinephrine_equivalent_dose.norepinephrine_equivalent_dose / CAST(
                        (
                            EXTRACT(
                                epoch
                                FROM mimiciv_derived.norepinephrine_equivalent_dose.endtime - mimiciv_derived.norepinephrine_equivalent_dose.starttime
                            ) / CAST(3600 AS NUMERIC)
                        ) AS NUMERIC
                    )
                ),
                (
                    mimiciv_derived.norepinephrine_equivalent_dose.endtime,
                    'norepinephrine_equivalent_dose.rate',
                    0.0
                )
        ) AS tokens (charttime, token_label, token_value) ON true
),
chemistry_aligned AS (
    SELECT mimiciv_derived.icustay_detail.stay_id AS stay_id,
        mimiciv_derived.chemistry.subject_id AS subject_id,
        mimiciv_derived.chemistry.hadm_id AS hadm_id,
        mimiciv_derived.chemistry.charttime AS charttime,
        mimiciv_derived.chemistry.specimen_id AS specimen_id,
        mimiciv_derived.chemistry.albumin AS albumin,
        mimiciv_derived.chemistry.globulin AS globulin,
        mimiciv_derived.chemistry.total_protein AS total_protein,
        mimiciv_derived.chemistry.aniongap AS aniongap,
        mimiciv_derived.chemistry.bicarbonate AS bicarbonate,
        mimiciv_derived.chemistry.bun AS bun,
        mimiciv_derived.chemistry.calcium AS calcium,
        mimiciv_derived.chemistry.chloride AS chloride,
        mimiciv_derived.chemistry.creatinine AS creatinine,
        mimiciv_derived.chemistry.glucose AS glucose,
        mimiciv_derived.chemistry.sodium AS sodium,
        mimiciv_derived.chemistry.potassium AS potassium
    FROM mimiciv_derived.chemistry
        JOIN mimiciv_derived.icustay_detail ON mimiciv_derived.chemistry.hadm_id = mimiciv_derived.icustay_detail.hadm_id
        AND mimiciv_derived.chemistry.charttime >= mimiciv_derived.icustay_detail.icu_intime
        AND mimiciv_derived.chemistry.charttime <= mimiciv_derived.icustay_detail.icu_outtime
    WHERE mimiciv_derived.icustay_detail.stay_id IS NOT NULL
),
chemistry_tokenized AS (
    SELECT chemistry_aligned.stay_id AS stay_id,
        chemistry_aligned.charttime AS charttime,
        tokens.token_label AS token_label,
        tokens.token_value AS token_value
    FROM chemistry_aligned
        JOIN LATERAL (
            VALUES (
                    'chemistry.albumin',
                    chemistry_aligned.albumin,
                    chemistry_aligned.albumin IS NULL
                ),
                (
                    'chemistry.globulin',
                    chemistry_aligned.globulin,
                    chemistry_aligned.globulin IS NULL
                ),
                (
                    'chemistry.total_protein',
                    chemistry_aligned.total_protein,
                    chemistry_aligned.total_protein IS NULL
                ),
                (
                    'chemistry.bicarbonate',
                    chemistry_aligned.bicarbonate,
                    chemistry_aligned.bicarbonate IS NULL
                ),
                (
                    'chemistry.bun',
                    chemistry_aligned.bun,
                    chemistry_aligned.bun IS NULL
                ),
                (
                    'chemistry.calcium',
                    chemistry_aligned.calcium,
                    chemistry_aligned.calcium IS NULL
                ),
                (
                    'chemistry.chloride',
                    chemistry_aligned.chloride,
                    chemistry_aligned.chloride IS NULL
                ),
                (
                    'chemistry.creatinine',
                    chemistry_aligned.creatinine,
                    chemistry_aligned.creatinine IS NULL
                ),
                (
                    'chemistry.glucose',
                    chemistry_aligned.glucose,
                    chemistry_aligned.glucose IS NULL
                ),
                (
                    'chemistry.sodium',
                    chemistry_aligned.sodium,
                    chemistry_aligned.sodium IS NULL
                ),
                (
                    'chemistry.potassium',
                    chemistry_aligned.potassium,
                    chemistry_aligned.potassium IS NULL
                )
        ) AS tokens (token_label, token_value, token_null) ON true
    WHERE NOT tokens.token_null
),
union_tokenized AS (
    SELECT vitalsign_tokenized.stay_id AS stay_id,
        vitalsign_tokenized.charttime AS charttime,
        vitalsign_tokenized.token_label AS token_label,
        vitalsign_tokenized.token_value AS token_value
    FROM vitalsign_tokenized
    UNION ALL
    SELECT crrt_tokenized.stay_id AS stay_id,
        crrt_tokenized.charttime AS charttime,
        crrt_tokenized.token_label AS token_label,
        crrt_tokenized.token_value AS token_value
    FROM crrt_tokenized
    UNION ALL
    SELECT norepinephrine_equivalent_dose_tokenized.stay_id AS stay_id,
        norepinephrine_equivalent_dose_tokenized.charttime AS charttime,
        norepinephrine_equivalent_dose_tokenized.token_label AS token_label,
        norepinephrine_equivalent_dose_tokenized.token_value AS token_value
    FROM norepinephrine_equivalent_dose_tokenized
    UNION ALL
    SELECT chemistry_tokenized.stay_id AS stay_id,
        chemistry_tokenized.charttime AS charttime,
        chemistry_tokenized.token_label AS token_label,
        chemistry_tokenized.token_value AS token_value
    FROM chemistry_tokenized
)
SELECT union_tokenized.stay_id,
    union_tokenized.charttime,
    union_tokenized.token_label,
    union_tokenized.token_value,
    CAST(
        floor(
            percent_rank() OVER (
                PARTITION BY union_tokenized.token_label
                ORDER BY union_tokenized.token_value
            ) * 100
        ) AS INTEGER
    ) AS percentile
FROM union_tokenized
ORDER BY union_tokenized.stay_id,
    union_tokenized.charttime