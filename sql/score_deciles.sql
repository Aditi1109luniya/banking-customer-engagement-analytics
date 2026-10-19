WITH scored AS (
    SELECT p.model_id, p.split, p.score, p.rank, c.y,
           NTILE(10) OVER (PARTITION BY p.model_id, p.split ORDER BY p.rank) AS score_decile
    FROM prospect_scores p JOIN clean_campaigns c USING(record_id)
    WHERE (:split = 'all' OR p.split = :split)
)
SELECT model_id, split, score_decile, COUNT(*) AS observations,
       SUM(y) AS subscriptions, 1.0 * SUM(y) / COUNT(*) AS subscription_rate,
       MIN(score) AS minimum_score, MAX(score) AS maximum_score
FROM scored GROUP BY model_id, split, score_decile ORDER BY model_id, split, score_decile;
