-- Historical associations only. Month names cannot reconstruct years or exact dates.
SELECT c.month, COUNT(*) AS observations, SUM(c.y) AS subscriptions,
       1.0 * SUM(c.y) / COUNT(*) AS subscription_rate
FROM clean_campaigns c JOIN dataset_splits s USING(record_id)
WHERE (:split = 'all' OR s.split = :split)
GROUP BY c.month
ORDER BY CASE c.month WHEN 'jan' THEN 1 WHEN 'feb' THEN 2 WHEN 'mar' THEN 3
         WHEN 'apr' THEN 4 WHEN 'may' THEN 5 WHEN 'jun' THEN 6 WHEN 'jul' THEN 7
         WHEN 'aug' THEN 8 WHEN 'sep' THEN 9 WHEN 'oct' THEN 10 WHEN 'nov' THEN 11
         ELSE 12 END;
