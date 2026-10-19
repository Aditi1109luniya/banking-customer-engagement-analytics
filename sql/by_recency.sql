SELECT CASE WHEN c.pdays = 999 THEN 'not previously contacted (999)'
            WHEN c.pdays <= 7 THEN '0–7 days' WHEN c.pdays <= 30 THEN '8–30 days'
            ELSE '31+ days' END AS recency_band,
       COUNT(*) AS observations, SUM(c.y) AS subscriptions,
       1.0 * SUM(c.y) / COUNT(*) AS subscription_rate
FROM clean_campaigns c JOIN dataset_splits s USING(record_id)
WHERE (:split = 'all' OR s.split = :split)
GROUP BY recency_band
ORDER BY CASE recency_band WHEN '0–7 days' THEN 1 WHEN '8–30 days' THEN 2
         WHEN '31+ days' THEN 3 ELSE 4 END;
