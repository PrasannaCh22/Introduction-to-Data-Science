"""
PROJECT 2: STUDENT PERFORMANCE + POLYNOMIAL CUSTOMER PREDICTION
Part A: Predict student exam scores using Linear Regression.
Part B: Demonstrate Polynomial Regression on customer annual income vs spending score.

Run:
    pip install -r requirements.txt
    python 2_student_and_polynomial_regression.py

The program creates reproducible demonstration datasets automatically.
No external CSV files are required.
"""
from pathlib import Path
import sys

try:
    import numpy as np
    import pandas as pd

    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    print("All required packages imported successfully!")

except ImportError as exc:
    print("A required package is missing.")
    print("Install dependencies with: pip install numpy pandas matplotlib scikit-learn")
    raise SystemExit(1) from exc


BASE_DIR = Path(__file__).resolve().parent


def student_performance():
    """Train and evaluate a linear regression model for student marks."""
    rng = np.random.default_rng(42)
    study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 11])
    attendance = np.array([55, 60, 62, 65, 70, 72, 75, 80, 82, 90,
                           58, 63, 67, 69, 73, 77, 79, 84, 88, 92])
    noise = rng.normal(0, 3, len(study_hours))
    marks = np.clip(25 + 5.8 * study_hours + 0.22 * attendance + noise, 0, 100)

    df = pd.DataFrame({
        "study_hours_per_day": study_hours,
        "attendance_percent": attendance,
        "exam_score": marks
    })

    X = df[["study_hours_per_day", "attendance_percent"]]
    y = df["exam_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    print("\n========== PART A: STUDENT PERFORMANCE ==========")
    print("Algorithm: Multiple Linear Regression")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, predicted):.2f} marks")
    print(f"Mean Squared Error: {mean_squared_error(y_test, predicted):.2f}")
    print(f"R-squared score: {r2_score(y_test, predicted):.3f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print("Coefficients:")
    for feature, coefficient in zip(X.columns, model.coef_):
        print(f"  {feature}: {coefficient:.2f}")

    hours = float(input("\nEnter a student's daily study hours (0-16): "))
    attendance_value = float(input("Enter attendance percentage (0-100): "))
    if not 0 <= hours <= 16 or not 0 <= attendance_value <= 100:
        raise ValueError("Study hours must be 0-16 and attendance must be 0-100.")
    score = model.predict(pd.DataFrame({
        "study_hours_per_day": [hours],
        "attendance_percent": [attendance_value]
    }))[0]
    print(f"Predicted exam score: {np.clip(score, 0, 100):.2f} / 100")

    plt.figure(figsize=(7, 5))
    plt.scatter(df["study_hours_per_day"], df["exam_score"], label="Student records")
    order = np.argsort(df["study_hours_per_day"].to_numpy())
    # A one-feature visual trend line, fitted only for visualization.
    visual_model = LinearRegression().fit(
        df[["study_hours_per_day"]], df["exam_score"]
    )
    plt.plot(
        df["study_hours_per_day"].to_numpy()[order],
        visual_model.predict(df[["study_hours_per_day"]].iloc[order]),
        label="Linear trend"
    )
    plt.xlabel("Study hours per day")
    plt.ylabel("Exam score")
    plt.title("Student Study Hours vs Exam Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "student_performance_plot.png", dpi=150)
    plt.close()
    print("Saved graph: student_performance_plot.png")


def polynomial_customer_prediction():
    """Fit a polynomial model to a curved customer spending pattern."""
    rng = np.random.default_rng(7)
    income = np.linspace(15, 120, 45)  # annual income in thousand currency units
    spending = 18 + 0.95 * income - 0.008 * income**2 + rng.normal(0, 3, len(income))
    df = pd.DataFrame({
        "annual_income_thousands": income,
        "spending_score": np.clip(spending, 0, 100)
    })

    X = df[["annual_income_thousands"]]
    y = df["spending_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    print("\n========== PART B: POLYNOMIAL CUSTOMER DATA ==========")
    print("Task: Estimate customer spending score from annual income.")
    print("Algorithm: Polynomial Regression (degree 2)")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, predicted):.2f}")
    print(f"R-squared score: {r2_score(y_test, predicted):.3f}")

    income_value = float(input(
        "\nEnter customer annual income in thousands (15-120): "
    ))
    if not 15 <= income_value <= 120:
        raise ValueError("Income must be between 15 and 120 thousand.")
    spend_score = model.predict(pd.DataFrame({
        "annual_income_thousands": [income_value]
    }))[0]
    print(f"Predicted customer spending score: {np.clip(spend_score, 0, 100):.2f} / 100")

    grid = np.linspace(income.min(), income.max(), 250)
    curve = model.predict(pd.DataFrame({"annual_income_thousands": grid}))
    plt.figure(figsize=(7, 5))
    plt.scatter(income, spending, label="Customer data")
    plt.plot(grid, curve, label="Polynomial regression (degree 2)")
    plt.xlabel("Annual income (thousands)")
    plt.ylabel("Spending score")
    plt.title("Customer Income vs Spending Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "customer_polynomial_regression_plot.png", dpi=150)
    plt.close()
    print("Saved graph: customer_polynomial_regression_plot.png")


def main():
    print("STUDENT PERFORMANCE AND POLYNOMIAL REGRESSION PROJECT")
    student_performance()
    polynomial_customer_prediction()
    print("\nBoth regression demonstrations completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"\nInput error: {exc}")
        raise SystemExit(1) from exc
