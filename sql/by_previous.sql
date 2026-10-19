SELECT CASE WHEN c.previous = 0 THEN '0' WHEN c.previous = 1 THEN '1'
            WHEN c.previous <= 3 THEN '2–3' ELSE '4+' END AS previous_band,
       COUNT(*) AS observations, SUM(c.y) AS subscriptions,
       1.0 * SUM(c.y) / COUNT(*) AS subscription_rate
FROM clean_campaigns c JOIN dataset_splits s USING(record_id)
WHERE (:split = 'all' OR s.split = :split)
GROUP BY previous_band
ORDER BY CASE previous_band WHEN '0' THEN 1 WHEN '1' THEN 2 WHEN '2–3' THEN 3 ELSE 4 END;
