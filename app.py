import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Title and Description
st.title("Spotify Song Clustering App")
st.write("""
### Visualisasi dan Analisis Klasterisasi Lagu
Upload dataset Spotify Anda, pilih fitur, dan lihat hasil klasterisasi menggunakan K-Means.
""")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    
    # Pilih fitur
    features = ['streams', 'danceability_%', 'valence_%', 'energy_%', 
                'acousticness_%', 'instrumentalness_%', 'liveness_%', 
                'speechiness_%', 'bpm']
    
    # Pastikan fitur ada di dataset
    if not set(features).issubset(data.columns):
        st.error("Dataset Anda tidak memiliki semua fitur yang diperlukan.")
    else:
        # Preprocessing
        data['streams'] = pd.to_numeric(data['streams'], errors='coerce')
        data = data.dropna(subset=features)
        
        # Standarisasi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[features])
        
        # Elbow Method untuk menentukan k
        st.subheader("Elbow Method")
        wcss = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        
        # Plot Elbow Method
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_range, wcss, marker='o')
        ax.set_title('Elbow Method for Optimal k')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)
        
        # Pilih jumlah klaster
        k_optimal = st.slider("Pilih jumlah klaster (k):", 2, 10, 4)
        
        # K-Means
        kmeans = KMeans(n_clusters=k_optimal, random_state=42)
        kmeans.fit(X_scaled)
        
        # Tambahkan label klaster ke dataset
        data['Cluster'] = kmeans.labels_
        
        # Evaluasi Silhouette Score
        sil_score = silhouette_score(X_scaled, kmeans.labels_)
        st.write(f"**Silhouette Score:** {sil_score:.2f}")
        
        # Visualisasi Klaster (PCA)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        data['PCA1'] = X_pca[:, 0]
        data['PCA2'] = X_pca[:, 1]
        
        st.subheader("Hasil Klasterisasi")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            x='PCA1', y='PCA2', hue='Cluster', data=data, palette='Set2', s=100, alpha=0.7, ax=ax
        )
        ax.set_title('Clusters of Songs')
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        st.pyplot(fig)
        
        # Tampilkan ringkasan klaster
        st.subheader("Ringkasan Klaster")
        cluster_summary = data.groupby('Cluster')[features].mean()
        st.dataframe(cluster_summary)
        
        # Unduh dataset dengan label klaster
        st.subheader("Download Dataset dengan Label Klaster")
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='clustered_songs.csv',
            mime='text/csv',
        )
