DROP TABLE IF EXISTS mimiciv_local.d_tokens;
CREATE TABLE mimiciv_local.d_tokens AS (
    WITH tokens AS (
        -- All event types
        SELECT token_label
        FROM mimiciv_local.tokenevents
        GROUP BY token_label
        UNION ALL
        -- 'Magnitude' token for expressing value after a measureable event
        SELECT 'magnitude.' || cast(magnitude_num AS TEXT)
        FROM generate_series(0, 9) AS magnitude_num
        UNION ALL
        -- 'Hour' token
        SELECT 'hr.' || cast(hr_num AS TEXT)
        FROM generate_series(0, 23) AS hr_num
        UNION ALL
        -- 'Special' tokens: death, discharge, admission, etc.
        SELECT v.token_label
        FROM(
                VALUES ('mort'),
                    ('discharge'),
                    ('admission')
            ) AS v(token_label)
    )
    SELECT row_number() over (
            ORDER BY token_label
        ) AS token_id,
        token_label
    FROM tokens
);
CREATE UNIQUE INDEX id_token ON mimiciv_local.d_tokens(token_id);