-- Observed telephone/cellular association, not evidence of channel effectiveness.
SELECT c.contact, COUNT(*) AS observations, SUM(c.y) AS subscriptions,
       1.0 * SUM(c.y) / COUNT(*) AS subscription_rate
FROM clean_campaigns c JOIN dataset_splits s USING(record_id)
WHERE (:split = 'all' OR s.split = :split)
GROUP BY c.contact ORDER BY observations DESC, c.contact;
