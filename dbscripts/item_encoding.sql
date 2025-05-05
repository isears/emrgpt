DROP TABLE IF EXISTS mimiciv_local.item_encoding;
CREATE TABLE mimiciv_local.item_encoding AS (
    WITH categorical_items AS (
        SELECT itemid,
            value,
            COUNT(charttime) AS count
        FROM mimiciv_icu.chartevents
        WHERE valuenum IS NULL
        GROUP BY itemid,
            value
    ),
    numeric_items AS (
        SELECT itemid,
            COUNT(charttime) AS count
        FROM mimiciv_icu.chartevents
        WHERE valuenum IS NOT NULL
        GROUP BY itemid
    ),
    combined_items AS (
        SELECT itemid,
            value,
            count
        FROM categorical_items
        WHERE count > 1000
        UNION
        SELECT itemid,
            'NUMERIC',
            count
        FROM numeric_items
        WHERE count > 1000
    )
    SELECT combined_items.itemid,
        mimiciv_icu.d_items.label,
        combined_items.value,
        combined_items.count,
        ROW_NUMBER() OVER (
            ORDER BY combined_items.itemid
        ) AS encoding
    FROM combined_items
        LEFT JOIN mimiciv_icu.d_items ON mimiciv_icu.d_items.itemid = combined_items.itemid
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_encoding ON mimiciv_local.item_encoding (encoding);