SELECT CASE WHEN c.pdays = 999 THEN 'no previous contact (pdays=999)'
            ELSE 'previously contacted (pdays!=999)' END AS prior_contact,
       COUNT(*) AS observations, SUM(c.y) AS subscriptions,
       1.0 * SUM(c.y) / COUNT(*) AS subscription_rate
FROM clean_campaigns c JOIN dataset_splits s USING(record_id)
WHERE (:split = 'all' OR s.split = :split)
GROUP BY prior_contact ORDER BY observations DESC, prior_contact;
