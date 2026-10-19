WITH population AS (
    SELECT c.* FROM clean_campaigns c JOIN dataset_splits s USING(record_id)
    WHERE (:split = 'all' OR s.split = :split)
), counts AS (
    SELECT 'job' AS field, COUNT(*) AS observations, SUM(job='unknown') AS unknown_count
    FROM population
    UNION ALL SELECT 'marital', COUNT(*), SUM(marital='unknown') FROM population
    UNION ALL SELECT 'education', COUNT(*), SUM(education='unknown') FROM population
    UNION ALL SELECT 'default', COUNT(*), SUM("default"='unknown') FROM population
    UNION ALL SELECT 'housing', COUNT(*), SUM(housing='unknown') FROM population
    UNION ALL SELECT 'loan', COUNT(*), SUM(loan='unknown') FROM population
)
SELECT field, observations, COALESCE(unknown_count,0) AS unknown_count,
       1.0 * unknown_count / NULLIF(observations,0) AS unknown_rate
FROM counts ORDER BY field;
