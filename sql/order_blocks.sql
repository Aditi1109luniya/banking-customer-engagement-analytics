-- Ten approximately equal blocks of supplied record order; no invented dates.
WITH population AS (
    SELECT c.record_id, c.y,
           1 + CAST((c.record_id - 1) * 10 /
               (SELECT COUNT(*) FROM clean_campaigns) AS INTEGER) AS order_block
    FROM clean_campaigns c JOIN dataset_splits s USING(record_id)
    WHERE (:split = 'all' OR s.split = :split)
)
SELECT order_block, MIN(record_id) AS first_record_id, MAX(record_id) AS last_record_id,
       COUNT(*) AS observations, SUM(y) AS subscriptions,
       1.0 * SUM(y) / COUNT(*) AS subscription_rate
FROM population GROUP BY order_block ORDER BY order_block;
