"""Task 1

Step 1: Modify customer_behavior_analyzer.py to analyze these influencer metrics:

followers_count (1K-1M)
posts_per_week (1-20)
avg_likes_per_post (10-50K)
avg_comments_per_post (5-2K)
engagement_rate (1-15%)

Step 2: Generate 500 influencer profiles with 3 natural segments:
1.Micro (1K-10K followers)
2.Mid-tier (10K-100K followers)
3.Macro (100K+ followers)

Step 3: Add 5% fake influencers with suspicious patterns:
1.High followers but low engagement
2.Unnatural like-to-comment ratios
3.Inconsistent posting patterns

Step 4: Use clustering + anomaly detection to catch the fakes!

Required Output
1.PCA plot showing real vs fake influencers
2.Business report: Which accounts need investigation?
3.Accuracy score: How well did you catch the fakes?

Success Criteria
1.Detect at least 80% of fake influencers
2.Keep false positives under 10%
3.Provide actionable insights for brands"""
