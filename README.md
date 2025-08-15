# K-Means Clustering on Mall Customers Dataset

This project demonstrates unsupervised learning using K-Means clustering on the Mall Customers dataset. The analysis is performed in a Jupyter notebook using Python libraries such as Pandas, Scikit-learn, Matplotlib, and Seaborn.

## Project Structure
- `KMeans_Mall_Customers.ipynb`: Main notebook containing all code, explanations, and visualizations.
- `Mall_Customers.csv`: Dataset file containing customer information.
- `images/`: Folder where all generated plots are saved as PNG files.

## Steps Performed in the Notebook
1. **Import Required Libraries**: Loads all necessary Python libraries for data analysis and visualization.
2. **Load the Dataset**: Reads the customer data from the CSV file and displays the first few rows.
3. **Explore and Preprocess Data**: Checks for missing values, encodes categorical variables (e.g., Gender), and selects relevant features for clustering.
4. **Optional: Dimensionality Reduction with PCA**: Reduces data to 2D for visualization using Principal Component Analysis (PCA).
5. **Determine Optimal Number of Clusters (Elbow Method)**: Uses the Elbow Method to find the best number of clusters for K-Means.
6. **Fit K-Means and Assign Cluster Labels**: Applies K-Means clustering and assigns cluster labels to each customer.
7. **Visualize Clusters**: Plots the clusters in 2D, color-coded by cluster label, and saves the plot in the `images` folder.
8. **Evaluate Clustering with Silhouette Score**: Calculates and displays the silhouette score to evaluate clustering quality.

## How to Use
1. Open the notebook `KMeans_Mall_Customers.ipynb` in Jupyter or VS Code.
2. Run each cell in order. All plots will be saved in the `images` folder for reference.
3. Review the visualizations and results to understand the clustering of mall customers.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required packages using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Notes
- The notebook sets `OMP_NUM_THREADS=1` to avoid a known KMeans memory leak warning on Windows.
- All generated plots are saved as PNG files in the `images` directory for easy access.

## References
- [Scikit-learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Mall Customers Dataset (Kaggle)](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial)
