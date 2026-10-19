-- Call with all only after model/policy freeze; the CLI enforces this gate.
SELECT s.split, MIN(c.record_id) AS first_record_id, MAX(c.record_id) AS last_record_id,
       COUNT(*) AS observations, SUM(c.y) AS subscriptions,
       1.0 * SUM(c.y) / COUNT(*) AS subscription_rate
FROM clean_campaigns c JOIN dataset_splits s USING(record_id)
WHERE (:split = 'all' OR s.split = :split)
GROUP BY s.split ORDER BY first_record_id;
