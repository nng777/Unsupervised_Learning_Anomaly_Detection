import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


class InfluencerFraudDetector:
    #Detect fake social media influencers using unsupervised learning.

    def __init__(self):
        self.data = None
        self.scaled = None
        self.scaler = StandardScaler()

    def generate_influencer_data(self, n_influencers: int = 500, seed: int = 42) -> pd.DataFrame:
        #Generate influencer profiles including a small portion of fakes.

        rng = np.random.default_rng(seed)

        segments = {
            "micro": {
                "size": int(0.4 * n_influencers),
                "followers_count": (5_000, 2_000),
                "posts_per_week": (3, 1),
                "avg_likes_per_post": (300, 100),
                "avg_comments_per_post": (20, 8),
                "engagement_rate": (8, 1.5),
            },
            "mid_tier": {
                "size": int(0.4 * n_influencers),
                "followers_count": (50_000, 20_000),
                "posts_per_week": (5, 2),
                "avg_likes_per_post": (2_000, 500),
                "avg_comments_per_post": (50, 20),
                "engagement_rate": (7, 1.0),
            },
            "macro": {
                "size": n_influencers - int(0.8 * n_influencers),
                "followers_count": (300_000, 120_000),
                "posts_per_week": (7, 3),
                "avg_likes_per_post": (10_000, 3_000),
                "avg_comments_per_post": (200, 70),
                "engagement_rate": (5, 1.0),
            },
        }

        features = [
            "followers_count",
            "posts_per_week",
            "avg_likes_per_post",
            "avg_comments_per_post",
            "engagement_rate",
        ]

        all_data = {f: [] for f in features}

        for seg_params in segments.values():
            size = seg_params["size"]
            for feature in features:
                mean, std = seg_params[feature]
                all_data[feature].extend(rng.normal(mean, std, size))

        df = pd.DataFrame(all_data)

        # Assign a unique identifier to each influencer account
        df["influencer_id"] = [f"INF_{i:05d}" for i in range(n_influencers)]

        # Add fake influencers (5%)
        n_fake = int(0.05 * n_influencers)
        fake_idx = rng.choice(len(df), size=n_fake, replace=False)
        df["is_fake"] = False
        df.loc[fake_idx, "is_fake"] = True

        for idx in fake_idx:
            pattern = rng.integers(0, 3)
            if pattern == 0:
                # High followers, low engagement
                df.at[idx, "followers_count"] = rng.normal(200_000, 50_000)
                df.at[idx, "engagement_rate"] = rng.normal(0.5, 0.2)
                likes = df.at[idx, "followers_count"] * df.at[idx, "engagement_rate"] / 100
                df.at[idx, "avg_likes_per_post"] = likes
                df.at[idx, "avg_comments_per_post"] = likes * 0.005
            elif pattern == 1:
                # Unnatural like-to-comment ratio
                likes = rng.normal(8_000, 2_000)
                df.at[idx, "followers_count"] = rng.normal(50_000, 10_000)
                df.at[idx, "avg_likes_per_post"] = likes
                df.at[idx, "avg_comments_per_post"] = likes * rng.choice([0.001, 0.5])
                df.at[idx, "engagement_rate"] = likes / df.at[idx, "followers_count"] * 100
            else:
                # Inconsistent posting
                df.at[idx, "followers_count"] = rng.normal(80_000, 20_000)
                df.at[idx, "posts_per_week"] = rng.choice([0, rng.integers(15, 25)])
                likes = rng.normal(3_000, 800)
                df.at[idx, "avg_likes_per_post"] = likes
                df.at[idx, "avg_comments_per_post"] = likes * 0.02
                df.at[idx, "engagement_rate"] = likes / df.at[idx, "followers_count"] * 100

        self.data = df.abs()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].abs()
        self.data = df

        # Save generated influencer data for external review
        self.data.to_csv("influencer_data.csv", index=False)
        return self.data

    def _scale(self):
        features = [
            "followers_count",
            "posts_per_week",
            "avg_likes_per_post",
            "avg_comments_per_post",
            "engagement_rate",
        ]
        self.scaled = self.scaler.fit_transform(self.data[features])
        return self.scaled

    def cluster_influencers(self):
        self._scale()
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.data["cluster"] = kmeans.fit_predict(self.scaled)

    def detect_fakes(self) -> pd.DataFrame:
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(self.scaled)
        self.data["pred_fake"] = preds == -1
        return self.data[self.data["pred_fake"]]

    def pca_plot(self):
        pca = PCA(n_components=2)
        components = pca.fit_transform(self.scaled)
        plt.figure(figsize=(10, 7))
        fake_mask = self.data["pred_fake"]
        plt.scatter(
            components[~fake_mask, 0],
            components[~fake_mask, 1],
            alpha=0.6,
            label="Real",
        )
        plt.scatter(
            components[fake_mask, 0],
            components[fake_mask, 1],
            color="red",
            label="Detected Fake",
        )
        plt.title("PCA of Influencer Data")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def business_report(self):
        tp = sum(self.data["pred_fake"] & self.data["is_fake"])
        fp = sum(self.data["pred_fake"] & ~self.data["is_fake"])
        fn = sum(~self.data["pred_fake"] & self.data["is_fake"])

        detection_rate = tp / (tp + fn) if tp + fn else 0
        false_positive_rate = fp / len(self.data)
        accuracy = (tp + (len(self.data) - tp - fp - fn)) / len(self.data)

        print("\nBusiness Report")
        print("================")
        print(f"Influencers flagged for investigation: {tp + fp}")
        print(f"Detection rate of fakes: {detection_rate:.2%}")
        print(f"False positive rate: {false_positive_rate:.2%}")
        print(f"Overall accuracy: {accuracy:.2%}\n")

        flagged = self.data[self.data["pred_fake"]]
        if not flagged.empty:
            print("Accounts require review:")
            print(flagged[
                    [
                        "influencer_id",
                        "followers_count",
                        "posts_per_week",
                        "avg_likes_per_post",
                        "avg_comments_per_post",
                        "engagement_rate",
                    ]
                ]
                .round()
                .astype(int)
                .head())

            # Save the full list of flagged accounts for offline review
            export_df = flagged.copy()
            num_cols = export_df.select_dtypes(include=[np.number]).columns
            export_df[num_cols] = export_df[num_cols].round().astype(int)
            export_df.to_csv("flagged_influencers.csv", index=False)

    def run(self):
        self.generate_influencer_data()
        self.cluster_influencers()
        self.detect_fakes()
        self.pca_plot()
        self.business_report()


def main():
    detector = InfluencerFraudDetector()
    detector.run()


if __name__ == "__main__":
    main()