import utils
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import numpy as np
import pandas as pd

def main():
    df = utils.load_data()
    X = df.drop(columns=["weight", "label"])
    preprocessed_X = preprocess(X)
    sample_weight = df.weight.values
    decide_k(preprocessed_X, sample_weight)

    kmeans = KMeans(n_clusters=4)
    clusters =  kmeans.fit_predict(preprocessed_X, sample_weight=sample_weight)
    df['segment'] = clusters

    top_features = select_features(df, sample_weight)
    report(df, top_features)

def preprocess(df):
    num_cols = df.select_dtypes(exclude='object').columns
    cat_cols = df.select_dtypes(include='object').columns

    preprocessor = ColumnTransformer([('num', StandardScaler(), num_cols), ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])

    preprocessed_df = preprocessor.fit_transform(df)

    return preprocessed_df

def decide_k(preprocessed_X, sample_weight):
    inertia, sil = [], []

    K_range = range(2, 15)
    for k in tqdm(K_range):
        km = KMeans(n_clusters=k)
        labels = km.fit_predict(preprocessed_X, sample_weight=sample_weight)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(preprocessed_X, labels, n_jobs=-1, sample_size=5000))

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(K_range, inertia, 'o-')
    plt.title('Elbow Method')
    plt.xlabel('k')
    plt.ylabel('Inertia (SSE)')

    plt.subplot(1,2,2)
    plt.plot(K_range, sil, 'o-')
    plt.title('Silhouette Score')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')

    plt.show()

def select_features(df, sample_weight):
    z_num = z_score_num(df, sample_weight)
    z_cat = z_score_cat(df, sample_weight)

    top_features = {}

    for seg in df["segment"].unique():
        num_abs = z_num.loc[seg].abs().sort_values(ascending=False).head(5).index
        num_top = z_num.loc[seg, num_abs]

        cat_tops = []
        for col, zdf in z_cat.items():
            cat_abs = zdf.loc[seg].abs().sort_values(ascending=False).head(5).index
            cat_top = zdf.loc[seg, cat_abs]
            cat_top.index = [f"{col}={v}" for v in cat_top.index] 
            cat_tops.append(cat_top)

        cat_top = pd.concat(cat_tops).sort_values(key=abs, ascending=False).head(5)
        top_features[seg] = pd.concat([num_top, cat_top]).sort_values(key=abs, ascending=False)

    return top_features

def z_score_num(df, sample_weight):
    num_cols = df.select_dtypes(exclude='object').columns

    # overall_mean = df[num_cols].mean()
    # overall_std = df[num_cols].std()
    overall_mean = pd.Series(
        np.average(df[num_cols], axis=0, weights=sample_weight),
        index=num_cols
    )
    overall_std = pd.Series(
        np.sqrt(np.average((df[num_cols] - overall_mean) ** 2, 
                        axis=0, weights=sample_weight)),
        index=num_cols
    )
    # cluster_means = df.groupby("segment")[num_cols].mean()
    cluster_means = (
        df.groupby("segment")
        .apply(lambda g: pd.Series(
            np.average(g[num_cols], axis=0, weights=g["weight"]),
            index=num_cols
        ))
    )
    z_num = (cluster_means - overall_mean) / overall_std

    return z_num.drop(columns='segment')

def z_score_cat(df, sample_weight):
    cat_cols = df.select_dtypes(include='object').columns

    # overall_props = {}
    # for col in cat_cols:
    #     overall_props[col] = df[col].value_counts(normalize=True)

    # grouped_props = {}
    # for col in cat_cols:
    #     grouped_props[col] = (
    #         df.groupby("segment")[col]
    #         .value_counts(normalize=True)
    #         .unstack(fill_value=0)
    #     )

    # z_cat = {}
    # for col in cat_cols:
    #     overall = overall_props[col]
    #     grouped = grouped_props[col]
    #     z_cat[col] = (grouped - overall) / np.sqrt(overall * (1 - overall))

    z_cat = {}

    for col in cat_cols:
        categories = df[col].unique()
        # overall weighted proportions
        overall = {}
        for c in categories:
            mask = (df[col] == c)
            overall[c] = np.average(mask, weights=sample_weight)
        overall = pd.Series(overall)

        # weighted proportions per cluster
        grouped = {}
        for seg, g in df.groupby("segment"):
            seg_weights = g["weight"]
            seg_props = {c: np.average(g[col] == c, weights=seg_weights) for c in categories}
            grouped[seg] = seg_props
        grouped = pd.DataFrame(grouped).T.fillna(0)

        z_cat[col] = (grouped - overall) / np.sqrt(overall * (1 - overall))

    return z_cat

def report(df, top_features):
    def weighted_mode(x, w):
        s = pd.Series(w).groupby(x).sum()
        return s.idxmax()

    def weighted_mean(x, w):
        return np.average(x, weights=w)
    
    # segment_summary = df.groupby("segment").agg({
    #     "age": "mean",
    #     "education": lambda x: x.value_counts().index[0],
    #     "sex": lambda x: x.value_counts().index[0],
    #     "marital stat": lambda x: x.value_counts().index[0],
    #     "label": lambda x: (x == "50000+.").mean()
    # }).rename(columns={"label": "high_income_ratio"})
    
    segment_summary = df.groupby("segment").apply(lambda g: pd.Series({
        "age": weighted_mean(g["age"], g["weight"]),
        "education": weighted_mode(g["education"], g["weight"]),
        "sex": weighted_mode(g["sex"], g["weight"]),
        "marital stat": weighted_mode(g["marital stat"], g["weight"]),
        "high_income_ratio": np.average((g["label"] == "50000+.").astype(int), weights=g["weight"])
    }))

    print(segment_summary)

    for seg, feats in top_features.items():
        seg_df = df[df.segment == seg]

        print(f"\n=== Segment {seg} ===")
        
        print("Basic Summary:")
        for col in segment_summary.columns:
            val = segment_summary.loc[seg, col]
            if isinstance(val, float):
                print(f"  {col:<20}: {val:8.3f}")
            else:
                print(f"  {col:<20}: {val}")

        print("\nTop Differentiating Features:")
        for feat, zval in feats.items():
            arrow = "↑" if zval > 0 else "↓"
            zabs = abs(zval)

            if "=" not in feat:
                # seg_mean = df.loc[df.segment == seg, feat].mean()
                # overall_mean = df[feat].mean()

                seg_mean = np.average(df.loc[df.segment == seg, feat], weights=seg_df["weight"])
                overall_mean = np.average(df[feat], weights=df["weight"])
                print(f"{feat:<45} {arrow}{zabs:4.2f}σ   mean={seg_mean:8.2f} (overall={overall_mean:8.2f})")
            else:
                col, val = feat.split("=")
                # seg_ratio = (df.loc[df.segment == seg, col] == val).mean()
                # overall_ratio = (df[col] == val).mean()

                seg_ratio = np.average((seg_df[col] == val), weights=seg_df["weight"])
                overall_ratio = np.average((df[col] == val), weights=df["weight"])

                print(f"{feat:<60} {arrow}{zabs:4.2f}σ   ratio={seg_ratio:5.2%} (overall={overall_ratio:5.2%})")

if __name__ == '__main__':
    main()