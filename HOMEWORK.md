# 🎯 Homework: Social Media Anomaly Detection

## **Challenge: Instagram Influencer Fraud Detector** 📱

### 🎯 Your Mission
Create a system to detect **fake influencers** who buy followers and engagement using unsupervised learning.

### 📊 The Data
Modify `customer_behavior_analyzer.py` to analyze these influencer metrics:
- `followers_count` (1K-1M)
- `posts_per_week` (1-20)
- `avg_likes_per_post` (10-50K)
- `avg_comments_per_post` (5-2K)
- `engagement_rate` (1-15%)

### 🔍 The Task
**Step 1**: Generate 500 influencer profiles with 3 natural segments:
- **Micro** (1K-10K followers)
- **Mid-tier** (10K-100K followers) 
- **Macro** (100K+ followers)

**Step 2**: Add 5% **fake influencers** with suspicious patterns:
- High followers but low engagement
- Unnatural like-to-comment ratios
- Inconsistent posting patterns

**Step 3**: Use clustering + anomaly detection to catch the fakes!

### 📈 Required Output
1. **PCA plot** showing real vs fake influencers
2. **Business report**: Which accounts need investigation?
3. **Accuracy score**: How well did you catch the fakes?

### 🏆 Success Criteria
- Detect at least 80% of fake influencers
- Keep false positives under 10%
- Provide actionable insights for brands

### 💡 Starter Code
```python
# Add this to generate_sample_data()
def generate_influencer_data(self, n_influencers=500):
    # Your code here: create realistic influencer segments
    # Add 5% fake accounts with suspicious metrics
    pass
```

### ⏱️ Time Limit
**60 minutes** - Focus on core implementation, not perfect tuning.

### 🎓 Learning Goals
- Apply unsupervised learning to real-world fraud detection
- Understand how anomalies manifest in social media data
- Practice translating ML results to business decisions

---
**Hint**: Fake influencers often have followers/engagement ratios that don't match natural patterns! 🕵️‍♂️ 