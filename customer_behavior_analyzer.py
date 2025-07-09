import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings('ignore')

class CustomerBehaviorAnalyzer:
    def __init__(self):
        self.data = None
        self.scaled_data = None
        self.scaler = StandardScaler()
        
    def generate_sample_data(self, n_customers=1000):
        """Generate realistic customer behavior data with natural segments"""
        print(f"🎲 Generating {n_customers} customer records...")
        
        np.random.seed(42)
        
        # Create three customer segments
        segments = {
            'vip': {
                'size': int(0.2 * n_customers),
                'annual_spending': (5000, 1000),
                'frequency_purchases': (50, 10),
                'avg_order_value': (100, 20),
                'days_since_last_purchase': (5, 2),
                'customer_lifetime_months': (36, 12)
            },
            'regular': {
                'size': int(0.6 * n_customers),
                'annual_spending': (1500, 500),
                'frequency_purchases': (20, 5),
                'avg_order_value': (50, 15),
                'days_since_last_purchase': (15, 5),
                'customer_lifetime_months': (24, 8)
            },
            'low_value': {
                'size': n_customers - int(0.2 * n_customers) - int(0.6 * n_customers),
                'annual_spending': (300, 100),
                'frequency_purchases': (5, 2),
                'avg_order_value': (25, 8),
                'days_since_last_purchase': (45, 15),
                'customer_lifetime_months': (12, 4)
            }
        }
        
        # Generate data for each segment
        all_data = {feature: [] for feature in segments['vip'].keys() if feature != 'size'}
        
        for segment_name, params in segments.items():
            size = params['size']
            for feature in all_data.keys():
                mean, std = params[feature]
                all_data[feature].extend(np.random.normal(mean, std, size))
        
        # Add anomalies (5% of data)
        anomaly_count = int(0.05 * n_customers)
        anomaly_indices = np.random.choice(n_customers, size=anomaly_count, replace=False)
        
        for idx in anomaly_indices:
            if np.random.random() > 0.5:
                # High spender anomaly
                all_data['annual_spending'][idx] = np.random.normal(15000, 2000)
                all_data['avg_order_value'][idx] = np.random.normal(500, 100)
            else:
                # High frequency, low spending anomaly
                all_data['frequency_purchases'][idx] = np.random.normal(100, 10)
                all_data['annual_spending'][idx] = np.random.normal(200, 50)
                all_data['avg_order_value'][idx] = np.random.normal(5, 2)
        
        # Create DataFrame
        self.data = pd.DataFrame(all_data).abs()
        self.data['customer_id'] = [f'CUST_{i:04d}' for i in range(n_customers)]
        
        print(f"✅ Generated {n_customers} customers with 3 segments + {anomaly_count} anomalies")
        return self.data
    
    def explore_data(self):
        """Perform basic exploratory data analysis"""
        print(f"\n🔍 Exploring customer data...")
        
        features = ['annual_spending', 'frequency_purchases', 'avg_order_value', 
                   'days_since_last_purchase', 'customer_lifetime_months']
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"\nKey statistics:")
        print(self.data[features].describe().round(2))
        
        # Enhanced visualizations with educational annotations
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('📊 Customer Behavior Patterns - Look for Multiple Peaks!', fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, feature in enumerate(features):
            row, col = i // 3, i % 3
            
            # Create histogram with better styling
            n, bins, patches = axes[row, col].hist(self.data[feature], bins=25, alpha=0.8, 
                                                  color=colors[i], edgecolor='black', linewidth=0.5)
            
            # Add mean line
            mean_val = self.data[feature].mean()
            axes[row, col].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                                  label=f'Mean: {mean_val:.0f}')
            
            # Styling
            axes[row, col].set_title(f'{feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[row, col].set_xlabel('Value', fontsize=10)
            axes[row, col].set_ylabel('Number of Customers', fontsize=10)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend(fontsize=8)
            
            # Add educational annotation
            if i == 0:  # Annual spending
                axes[row, col].text(0.7, 0.9, 'Multiple peaks =\nDifferent segments!', 
                                   transform=axes[row, col].transAxes, fontsize=9,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        fig.delaxes(axes[1, 2])
        plt.tight_layout()
        plt.show()
        
        # Enhanced correlation heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.data[features].corr()
        
        # Create mask for better visualization
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, fmt='.2f', cbar_kws={"shrink": .8},
                   annot_kws={'size': 12, 'weight': 'bold'})
        
        plt.title('🔥 Feature Relationships - Strong Correlations Help Clustering!', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        # Add educational note
        plt.figtext(0.5, 0.02, 'Red = Positive Correlation | Blue = Negative Correlation | White = No Correlation', 
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Data exploration complete - Notice the patterns that suggest natural groupings!")
    
    def perform_clustering(self, n_clusters=4):
        """Perform K-means clustering to segment customers"""
        print(f"\n🎯 Performing customer segmentation...")
        
        features = ['annual_spending', 'frequency_purchases', 'avg_order_value', 
                   'days_since_last_purchase', 'customer_lifetime_months']
        
        # Scale features
        self.scaled_data = self.scaler.fit_transform(self.data[features])
        print(f"Features scaled for clustering")
        
        # Find optimal clusters using elbow method
        inertias = []
        k_range = range(1, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Enhanced Elbow method visualization
        plt.figure(figsize=(12, 7))
        plt.plot(k_range, inertias, 'bo-', linewidth=3, markersize=8, color='#FF6B6B')
        
        # Highlight the elbow point (typically around 3-4 clusters)
        optimal_k = 3  # Educational guess for demonstration
        plt.plot(optimal_k, inertias[optimal_k-1], 'go', markersize=15, label=f'Suggested K={optimal_k}')
        
        plt.title('🎯 Elbow Method - Find the "Elbow" Point!', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Number of Clusters (K)', fontsize=14)
        plt.ylabel('Inertia (Lower = Better Clustering)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add educational annotations
        plt.annotate('The "elbow" is where\nthe curve bends most!', 
                    xy=(optimal_k, inertias[optimal_k-1]), xytext=(6, inertias[optimal_k-1] + 5000),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=11, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.scaled_data)
        self.data['cluster'] = clusters
        
        print(f"Created {n_clusters} customer segments:")
        cluster_counts = self.data['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(self.data)) * 100
            print(f"  Cluster {cluster_id}: {count} customers ({percentage:.1f}%)")
        
        # Analyze cluster characteristics
        cluster_summary = self.data.groupby('cluster')[features].mean()
        print(f"\nCluster characteristics:")
        print(cluster_summary.round(2))
        
        # Enhanced PCA visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.scaled_data)
        
        plt.figure(figsize=(14, 9))
        
        # Create scatter plot with better colors and styling
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, cluster_id in enumerate(sorted(self.data['cluster'].unique())):
            cluster_mask = clusters == cluster_id
            plt.scatter(pca_data[cluster_mask, 0], pca_data[cluster_mask, 1], 
                       c=colors[i % len(colors)], alpha=0.7, s=60, 
                       label=f'Segment {cluster_id}', edgecolors='black', linewidth=0.5)
        
        plt.xlabel(f'PC1 - Explains {pca.explained_variance_ratio_[0]:.1%} of variance', fontsize=14)
        plt.ylabel(f'PC2 - Explains {pca.explained_variance_ratio_[1]:.1%} of variance', fontsize=14)
        plt.title('🎨 Customer Segments in 2D Space\n(PCA reduces 5 features to 2 dimensions)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.legend(title='Customer Segments', title_fontsize=12, fontsize=11, 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add educational note
        total_variance = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
        plt.figtext(0.5, 0.02, f'Together, PC1 and PC2 explain {total_variance:.1%} of the data variation', 
                   ha='center', fontsize=11, style='italic')
        
        plt.tight_layout()
        plt.show()
        
        # Interpret clusters
        self._interpret_clusters(cluster_summary)
        
        print("✅ Customer segmentation complete")
        return clusters
    
    def _interpret_clusters(self, cluster_summary):
        """Interpret what each cluster represents"""
        print(f"\n💡 Cluster Interpretation:")
        
        for cluster_id in cluster_summary.index:
            spending = cluster_summary.loc[cluster_id, 'annual_spending']
            frequency = cluster_summary.loc[cluster_id, 'frequency_purchases']
            recency = cluster_summary.loc[cluster_id, 'days_since_last_purchase']
            
            if spending > 3000 and frequency > 30:
                segment = "👑 VIP Customers"
                strategy = "Exclusive offers, premium service"
            elif spending > 1000 and frequency > 15:
                segment = "😊 Regular Customers"
                strategy = "Upselling, cross-selling campaigns"
            elif recency > 30:
                segment = "⚠️ At-Risk Customers"
                strategy = "Win-back campaigns, re-engagement"
            else:
                segment = "🌱 New/Low-Value Customers"
                strategy = "Onboarding, education programs"
            
            print(f"  Cluster {cluster_id}: {segment}")
            print(f"    Avg Spending: ${spending:.0f}, Frequency: {frequency:.1f}")
            print(f"    Strategy: {strategy}")
    
    def detect_anomalies(self):
        """Detect anomalous customer behavior"""
        print(f"\n🚨 Detecting anomalous customers...")
        
        # Use two anomaly detection methods
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_anomalies = iso_forest.fit_predict(self.scaled_data)
        
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        lof_anomalies = lof.fit_predict(self.scaled_data)
        
        # Combine results
        self.data['anomaly'] = (iso_anomalies == -1) | (lof_anomalies == -1)
        
        anomaly_count = sum(self.data['anomaly'])
        print(f"Found {anomaly_count} anomalous customers ({anomaly_count/len(self.data)*100:.1f}%)")
        
        # Analyze anomalies
        anomaly_data = self.data[self.data['anomaly']]
        
        print(f"\nTop 5 anomalous customers:")
        for i, (_, row) in enumerate(anomaly_data.head(5).iterrows(), 1):
            anomaly_type = "High Spender" if row['annual_spending'] > 10000 else "Unusual Pattern"
            print(f"  {i}. {row['customer_id']}: {anomaly_type}")
            print(f"     Spending: ${row['annual_spending']:.0f}, Frequency: {row['frequency_purchases']:.1f}")
        
        # Visualize anomalies
        self._visualize_anomalies()
        
        print("✅ Anomaly detection complete")
        return anomaly_data
    
    def _visualize_anomalies(self):
        """Enhanced visualization of anomalies using PCA"""
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.scaled_data)
        
        plt.figure(figsize=(14, 9))
        
        # Plot normal points with cluster colors
        normal_mask = ~self.data['anomaly']
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, cluster_id in enumerate(sorted(self.data['cluster'].unique())):
            cluster_normal_mask = normal_mask & (self.data['cluster'] == cluster_id)
            if cluster_normal_mask.any():
                plt.scatter(pca_data[cluster_normal_mask, 0], pca_data[cluster_normal_mask, 1], 
                           c=colors[i % len(colors)], alpha=0.6, s=50, 
                           label=f'Normal - Segment {cluster_id}', edgecolors='gray', linewidth=0.3)
        
        # Plot anomalies with emphasis
        anomaly_mask = self.data['anomaly']
        plt.scatter(pca_data[anomaly_mask, 0], pca_data[anomaly_mask, 1], 
                   c='#FF4444', alpha=0.9, s=120, marker='X', 
                   label='🚨 Anomalies', edgecolors='darkred', linewidth=2)
        
        plt.xlabel(f'PC1 - Explains {pca.explained_variance_ratio_[0]:.1%} of variance', fontsize=14)
        plt.ylabel(f'PC2 - Explains {pca.explained_variance_ratio_[1]:.1%} of variance', fontsize=14)
        plt.title('🔍 Anomaly Detection Results\nRed X marks show unusual customer behavior', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.legend(title='Customer Types', title_fontsize=12, fontsize=10, 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add educational note
        anomaly_count = sum(anomaly_mask)
        total_count = len(self.data)
        plt.figtext(0.5, 0.02, f'Found {anomaly_count} anomalies out of {total_count} customers ({anomaly_count/total_count*100:.1f}%)', 
                   ha='center', fontsize=11, style='italic')
        
        plt.tight_layout()
        plt.show()
    
    def generate_insights(self):
        """Generate business insights and recommendations"""
        print(f"\n💼 Generating business insights...")
        
        # Segment analysis
        total_customers = len(self.data)
        total_revenue = self.data['annual_spending'].sum()
        
        print(f"\n📊 Business Summary:")
        print(f"  Total customers: {total_customers}")
        print(f"  Total revenue: ${total_revenue:,.0f}")
        print(f"  Average revenue per customer: ${total_revenue/total_customers:.0f}")
        
        # Cluster revenue analysis
        print(f"\n💰 Revenue by Segment:")
        for cluster_id in sorted(self.data['cluster'].unique()):
            cluster_data = self.data[self.data['cluster'] == cluster_id]
            cluster_revenue = cluster_data['annual_spending'].sum()
            cluster_size = len(cluster_data)
            
            print(f"  Cluster {cluster_id}: ${cluster_revenue:,.0f} ({cluster_revenue/total_revenue*100:.1f}%)")
            print(f"    {cluster_size} customers, ${cluster_revenue/cluster_size:.0f} avg per customer")
        
        # Anomaly insights
        anomaly_count = sum(self.data['anomaly'])
        anomaly_revenue = self.data[self.data['anomaly']]['annual_spending'].sum()
        
        print(f"\n🚨 Anomaly Impact:")
        print(f"  {anomaly_count} anomalous customers ({anomaly_count/total_customers*100:.1f}%)")
        print(f"  Generate ${anomaly_revenue:,.0f} revenue ({anomaly_revenue/total_revenue*100:.1f}%)")
        
        # Recommendations
        print(f"\n💡 Key Recommendations:")
        print(f"  1. 👑 VIP Program: Target high-value clusters with exclusive offers")
        print(f"  2. ⚠️ Retention: Re-engage at-risk customers with special campaigns")
        print(f"  3. 🚨 Investigation: Review anomalous customers for fraud/opportunities")
        print(f"  4. 📈 Growth: Develop strategies to move customers up segments")
        
        # Save results
        self._save_results()
        
        print("✅ Business insights generated")
    
    def _save_results(self):
        """Save analysis results to CSV files"""
        # Save complete dataset
        self.data.to_csv('customer_analysis_results.csv', index=False)
        
        # Save cluster summary
        features = ['annual_spending', 'frequency_purchases', 'avg_order_value', 
                   'days_since_last_purchase', 'customer_lifetime_months']
        cluster_summary = self.data.groupby('cluster')[features].agg(['mean', 'std', 'count'])
        cluster_summary.to_csv('cluster_summary.csv')
        
        # Save anomalies
        anomalies = self.data[self.data['anomaly']]
        anomalies.to_csv('detected_anomalies.csv', index=False)
        
        print(f"\n💾 Files saved:")
        print(f"  📊 customer_analysis_results.csv - Complete dataset")
        print(f"  📈 cluster_summary.csv - Segment characteristics")
        print(f"  🚨 detected_anomalies.csv - Anomalous customers")
    
    def run_analysis(self):
        """Run the complete customer behavior analysis"""
        print("🎯 CUSTOMER BEHAVIOR ANALYZER")
        print("=" * 50)
        print("Unsupervised Learning Demo: Customer Segmentation & Anomaly Detection")
        
        # Step 1: Generate data
        self.generate_sample_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Perform clustering
        self.perform_clustering()
        
        # Step 4: Detect anomalies
        self.detect_anomalies()
        
        # Step 5: Generate insights
        self.generate_insights()
        
        print(f"\n🎉 Analysis Complete!")
        print(f"Key Learnings:")
        print(f"  ✅ Discovered customer segments without labeled data")
        print(f"  ✅ Identified unusual customer behavior patterns")
        print(f"  ✅ Generated actionable business recommendations")
        print(f"  ✅ Visualized complex data in 2D using PCA")
        
        print(f"\nNext Steps:")
        print(f"  📁 Review generated CSV files for detailed results")
        print(f"  🔬 Experiment with different clustering algorithms")
        print(f"  📊 Apply these techniques to your own datasets")

def main():
    """Main function to run the analyzer"""
    analyzer = CustomerBehaviorAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 