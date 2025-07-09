# Customer Behavior Analyzer 🎯

A streamlined Python application that demonstrates **Unsupervised Learning** and **Anomaly Detection** through customer behavior analysis. This app automatically discovers customer segments and identifies unusual purchasing patterns without any labeled data.

## 🚀 Features

- **Customer Segmentation**: Groups customers using K-means clustering
- **Anomaly Detection**: Identifies unusual behavior with multiple algorithms
- **Data Visualization**: Clear visualizations with PCA plots and histograms
- **Business Insights**: Actionable recommendations based on analysis
- **Export Results**: Saves analysis results to CSV files

## 🛠️ Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 How to Run

Simply run the main script:
```bash
python customer_behavior_analyzer.py
```

The application will:
1. Generate realistic customer data (1000 customers)
2. Explore data patterns and correlations
3. Discover customer segments using clustering
4. Detect anomalous behavior patterns
5. Generate business insights and save results

## 🔍 What the App Does

### Customer Features Analyzed:
- **Annual Spending**: Total money spent per year
- **Purchase Frequency**: Number of purchases per year
- **Average Order Value**: Average amount per purchase
- **Days Since Last Purchase**: Recency of activity
- **Customer Lifetime**: How long they've been a customer

### Machine Learning Methods:

#### 1. **K-Means Clustering**
- Automatically discovers customer segments
- Uses elbow method to find optimal clusters
- Visualizes segments using PCA

#### 2. **Anomaly Detection**
- **Isolation Forest**: Identifies outliers by isolation
- **Local Outlier Factor**: Detects density-based anomalies
- **Combined Approach**: Merges results for robust detection

## 📈 Output

### Console Output:
- Data exploration statistics
- Cluster characteristics and business interpretation
- Anomaly detection results with specific examples
- Business insights and revenue analysis

### Generated Files:
- `customer_analysis_results.csv`: Complete dataset with ML labels
- `cluster_summary.csv`: Statistical summary of segments
- `detected_anomalies.csv`: Anomalous customers for investigation

### Visualizations:
- Feature distribution histograms
- Correlation heatmap
- Elbow curve for optimal clustering
- PCA visualization of customer segments
- Anomaly detection scatter plot

## 🎯 Learning Objectives

Perfect for understanding:

### Unsupervised Learning:
- **Clustering**: Grouping customers without labels
- **Pattern Discovery**: Finding hidden data structures
- **Dimensionality Reduction**: Using PCA for visualization

### Anomaly Detection:
- **Outlier Identification**: Finding unusual patterns
- **Multiple Methods**: Comparing detection algorithms
- **Business Applications**: Fraud detection and VIP identification

## 🏢 Business Applications

### Customer Segments Discovered:
- **👑 VIP Customers**: High spending, frequent purchases
- **😊 Regular Customers**: Moderate spending, consistent behavior
- **⚠️ At-Risk Customers**: Low recent activity, churn risk
- **🌱 New/Low-Value**: Infrequent, low-spending customers

### Anomaly Detection Uses:
- **Fraud Detection**: Unusual spending patterns
- **VIP Identification**: Exceptionally high-value customers
- **Behavior Monitoring**: Detecting significant changes

## 🔧 Customization

Easy to modify:

```python
# Change data size
analyzer.generate_sample_data(n_customers=2000)

# Adjust number of clusters
analyzer.perform_clustering(n_clusters=5)

# Modify anomaly sensitivity
# Edit contamination parameter in detect_anomalies()
```

## 📚 Educational Value

This compact app teaches:
- **Unsupervised learning fundamentals**
- **Real-world ML applications**
- **Data preprocessing importance**
- **Business value extraction**
- **Visualization techniques**

## 🎓 Key Learning Outcomes

After running this app, students will understand:
- ✅ How clustering discovers hidden customer segments
- ✅ Why anomaly detection is crucial for business
- ✅ How PCA enables high-dimensional data visualization
- ✅ The importance of translating ML results to business actions
- ✅ How to evaluate and interpret unsupervised learning results

## 🎯 Homework Assignment

### **Challenge: E-commerce Fraud Detection System** 🕵️‍♂️

**Your Mission**: Create a fraud detection system for an online marketplace by modifying the existing code.

**Task**: Add a new feature called `transaction_velocity` (transactions per day) and implement a **two-stage anomaly detection**:

1. **Stage 1**: Detect customers with unusual spending patterns (existing functionality)
2. **Stage 2**: Among normal customers, find those with suspicious transaction velocity

**Requirements**:
- Generate `transaction_velocity` data (normal: 0.5-2 transactions/day, fraudulent: 10-50/day)
- Create 2% fraudulent accounts with high velocity but normal spending
- Visualize both anomaly types in different colors on the PCA plot
- Calculate the **precision** and **recall** of your fraud detection system
- Write a business report explaining which customers need immediate investigation

**Bonus Points**:
- Implement a **confidence score** for each anomaly (0-100%)
- Create a **risk matrix** plotting spending anomalies vs velocity anomalies
- Suggest specific actions for each type of detected fraud

**Expected Output**: A modified script that identifies two types of fraudulent behavior and provides actionable business insights.

**Learning Goals**: Understanding multi-dimensional anomaly detection, evaluation metrics, and real-world fraud detection challenges.

## 🚀 Next Steps

After completing the homework, explore:
1. **Different clustering algorithms** (DBSCAN, Hierarchical clustering)
2. **Real datasets** from Kaggle or your domain
3. **Time-series anomaly detection** for trend analysis
4. **Deep learning approaches** (autoencoders for anomaly detection)

## 📊 Sample Output

```
🎯 CUSTOMER BEHAVIOR ANALYZER
Unsupervised Learning Demo: Customer Segmentation & Anomaly Detection

🎲 Generating 1000 customer records...
✅ Generated 1000 customers with 3 segments + 50 anomalies

🔍 Exploring customer data...
✅ Data exploration complete

🎯 Performing customer segmentation...
Created 4 customer segments:
  Cluster 0: 200 customers (20.0%)
  Cluster 1: 600 customers (60.0%)
  ...
✅ Customer segmentation complete

🚨 Detecting anomalous customers...
Found 95 anomalous customers (9.5%)
✅ Anomaly detection complete

💼 Generating business insights...
✅ Business insights generated

🎉 Analysis Complete!
```

## 🔬 Technical Details

- **Language**: Python 3.7+
- **Key Libraries**: scikit-learn, pandas, matplotlib, seaborn
- **Algorithms**: K-means, Isolation Forest, Local Outlier Factor, PCA
- **Data**: Synthetic customer behavior data with realistic patterns
- **Output**: CSV files and visualizations

---

**Perfect for**: Students learning unsupervised learning concepts through hands-on practice with realistic business scenarios. The streamlined code makes it easy to understand and modify while maintaining educational depth. 