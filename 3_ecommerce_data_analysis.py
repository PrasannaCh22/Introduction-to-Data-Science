"""
PROJECT 3: E-COMMERCE DATA ANALYSIS
Uses Pandas, NumPy, Matplotlib and Seaborn to analyze sales data.

Run:
    pip install -r requirements.txt
    python 3_ecommerce_data_analysis.py

No external dataset is required. A reproducible sample sales dataset is generated.
If ecommerce.csv is placed beside this script, it will be used instead.
Expected columns:
    order_id, order_date, product, category, quantity, unit_price, customer_id
"""
from pathlib import Path
import sys

try:
    import numpy as np
    import pandas as pd

    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import seaborn as sns

    print("All required packages imported successfully!")

except ImportError as e:
    print(f"Missing package: {e}")


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "ecommerce.csv"


def create_sample_data():
    rng = np.random.default_rng(21)
    products = {
        "Laptop": ("Electronics", 55000),
        "Headphones": ("Electronics", 1800),
        "Keyboard": ("Accessories", 1200),
        "Mouse": ("Accessories", 700),
        "Notebook": ("Stationery", 80),
        "Backpack": ("Fashion", 1400),
        "T-Shirt": ("Fashion", 650),
        "Desk Lamp": ("Home", 950),
    }
    rows = []
    start = pd.Timestamp("2025-01-01")
    for i in range(1, 151):
        product = str(rng.choice(list(products.keys())))
        category, base_price = products[product]
        quantity = int(rng.integers(1, 5))
        unit_price = round(base_price * float(rng.uniform(0.9, 1.1)), 2)
        date = start + pd.Timedelta(days=int(rng.integers(0, 180)))
        customer_id = f"C{int(rng.integers(1, 61)):03d}"
        rows.append((f"O{i:04d}", date, product, category, quantity,
                     unit_price, customer_id))
    return pd.DataFrame(rows, columns=[
        "order_id", "order_date", "product", "category",
        "quantity", "unit_price", "customer_id"
    ])


def load_data():
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded dataset: {DATA_FILE.name}")
    else:
        print("ecommerce.csv not found. Using generated demonstration data.")
        df = create_sample_data()

    required = {
        "order_id", "order_date", "product", "category",
        "quantity", "unit_price", "customer_id"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df = df.dropna(subset=[
        "order_id", "order_date", "product", "category",
        "quantity", "unit_price", "customer_id"
    ])
    df = df[(df["quantity"] > 0) & (df["unit_price"] >= 0)]
    if df.empty:
        raise ValueError("No valid rows remain after cleaning the dataset.")
    df["sales"] = df["quantity"] * df["unit_price"]
    return df


def main():
    df = load_data()
    df["month"] = df["order_date"].dt.to_period("M").astype(str)

    print("\n========== E-COMMERCE DATA ANALYSIS ==========")
    print(f"Number of orders: {df['order_id'].nunique()}")
    print(f"Number of customers: {df['customer_id'].nunique()}")
    print(f"Total units sold: {int(df['quantity'].sum())}")
    print(f"Total revenue: {df['sales'].sum():,.2f}")
    print(f"Average order value: {df.groupby('order_id')['sales'].sum().mean():,.2f}")

    category_sales = df.groupby("category")["sales"].sum().sort_values(ascending=False)
    product_sales = df.groupby("product")["sales"].sum().sort_values(ascending=False)
    monthly_sales = df.groupby("month")["sales"].sum().sort_index()
    top_customers = (
        df.groupby("customer_id")["sales"].sum()
        .sort_values(ascending=False).head(5)
    )

    print("\nRevenue by category:")
    print(category_sales.to_string())
    print("\nTop 5 products by revenue:")
    print(product_sales.head(5).to_string())
    print("\nMonthly revenue:")
    print(monthly_sales.to_string())
    print("\nTop 5 customers by revenue:")
    print(top_customers.to_string())

    # Save a cleaned copy and summary tables for project evidence.
    df.to_csv(BASE_DIR / "ecommerce_cleaned_data.csv", index=False)
    category_sales.rename("revenue").to_csv(BASE_DIR / "category_revenue.csv")
    monthly_sales.rename("revenue").to_csv(BASE_DIR / "monthly_revenue.csv")

    plt.figure(figsize=(8, 5))
    plt.bar(category_sales.index.astype(str), category_sales.values)
    plt.title("Revenue by Product Category")
    plt.xlabel("Category")
    plt.ylabel("Revenue")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ecommerce_category_revenue.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    monthly_sales.plot(marker="o")
    plt.title("Monthly E-commerce Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ecommerce_monthly_revenue.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.barh(product_sales.head(5).index.astype(str), product_sales.head(5).values)
    plt.title("Top 5 Products by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Product")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ecommerce_top_products.png", dpi=150)
    plt.close()

    print("\nFiles generated:")
    print("- ecommerce_cleaned_data.csv")
    print("- category_revenue.csv")
    print("- monthly_revenue.csv")
    print("- ecommerce_category_revenue.png")
    print("- ecommerce_monthly_revenue.png")
    print("- ecommerce_top_products.png")
    print("\nE-commerce analysis completed successfully.")


if __name__ == "__main__":
    main()
