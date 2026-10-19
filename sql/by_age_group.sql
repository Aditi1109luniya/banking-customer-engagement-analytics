SELECT c.age_group, COUNT(*) AS observations, SUM(c.y) AS subscriptions,
       1.0 * SUM(c.y) / COUNT(*) AS subscription_rate
FROM clean_campaigns c JOIN dataset_splits s USING(record_id)
WHERE (:split = 'all' OR s.split = :split)
GROUP BY c.age_group
ORDER BY CASE c.age_group WHEN 'under 30' THEN 1 WHEN '30–44' THEN 2
         WHEN '45–59' THEN 3 ELSE 4 END;
