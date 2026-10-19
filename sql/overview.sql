SELECT COUNT(*) AS observations, COALESCE(SUM(c.y), 0) AS subscriptions,
       1.0 * SUM(c.y) / NULLIF(COUNT(*), 0) AS subscription_rate
FROM clean_campaigns c JOIN dataset_splits s USING(record_id)
WHERE (:split = 'all' OR s.split = :split);
