import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# App Title
st.title("🌸 Iris Dataset Exploration & Visualization")
st.write("Exploring the Iris dataset using visualizations to understand feature distributions and relationships.")

# Load Dataset
df = sns.load_dataset('iris')

# Show Dataset
st.subheader("📋 Dataset Overview")
st.write("Shape:", df.shape)
st.dataframe(df.head())

st.subheader("🔍 Filter by Species")
species = st.selectbox("Select Species", df['species'].unique())
filtered_df = df[df['species'] == species]
st.dataframe(filtered_df)

# Statistical Summary
st.subheader("📊 Statistical Summary")
st.dataframe(df.describe())

# Scatter Plot
st.subheader("🔵 Scatter Plot — Sepal Length vs Petal Length")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species', palette='Set1', ax=ax)
ax.set_title('Sepal Length vs Petal Length')
st.pyplot(fig)

# Histogram
st.subheader("📈 Feature Distributions")
fig, ax = plt.subplots(figsize=(10, 6))
df.drop(columns='species').hist(bins=20, color='steelblue', edgecolor='black', ax=ax)
plt.suptitle('Feature Distributions')
st.pyplot(fig)

# Box Plot
st.subheader("📦 Box Plots — Feature vs Species")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for i, feature in enumerate(features):
    sns.boxplot(data=df, x='species', y=feature, palette='Set2', ax=axes[i//2][i%2])
    axes[i//2][i%2].set_title(f'{feature} by Species')
plt.tight_layout()
st.pyplot(fig)

# Heatmap
st.subheader("🌡️ Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.drop(columns='species').corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Feature Correlation Heatmap')
st.pyplot(fig)

# Conclusion
st.subheader("✅ Key Insights")
st.markdown("""
- Dataset has **no missing values**
- Petal length & width are highly correlated **(r = 0.96)**
- **Setosa** species is clearly separable from others
- Petal features are more useful than sepal features
""")
