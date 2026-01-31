import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import io
import base64
from fraud_detection import train_model

def create_prediction_report(model, df, input_data, save_path=None):
    print(f"Generating report for input: {input_data}...")
    
    # Pre-calculated features logic (Using passed df)
    categorical_features = ['location', 'device_type', 'merchant_category']
    numerical_features = ['amount', 'transaction_hour', 'past_transaction_count', 'avg_spending']
    
    # 2. Predict on Input
    input_df = pd.DataFrame([input_data])
    prob = model.predict_proba(input_df)[0][1]
    prob_pct = prob * 100
    is_high_risk = prob > 0.5
    color = 'red' if is_high_risk else 'green'
    
    # 3. Create Plots
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(f"Fraud Prediction Report\nProbability: {prob_pct:.2f}%", fontsize=20, color=color, weight='bold')
    
    # A. Gauge Chart (Simulated with Pie)
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title("Risk Gauge", fontsize=14)
    wedges, texts = ax1.pie([prob_pct, 100-prob_pct], startangle=90, colors=[color, '#e0e0e0'], counterclock=False, wedgeprops={'width':0.4})
    ax1.text(0, 0, f"{prob_pct:.1f}%", ha='center', va='center', fontsize=20, weight='bold')

    # B. Feature Contribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title("Input vs Averages", fontsize=14)
    
    avg_fraud = df[df['is_fraud']==1][numerical_features].mean()
    avg_normal = df[df['is_fraud']==0][numerical_features].mean()
    
    metrics = ['amount', 'transaction_hour', 'past_transaction_count']
    x = np.arange(len(metrics))
    width = 0.25
    
    val_input = [input_data[m] for m in metrics]
    val_fraud = [avg_fraud[m] for m in metrics]
    val_normal = [avg_normal[m] for m in metrics]
    
    ax2.bar(x - width, val_input, width, label='This Input', color='blue')
    ax2.bar(x, val_fraud, width, label='Avg Fraud', color='red', alpha=0.5)
    ax2.bar(x + width, val_normal, width, label='Avg Normal', color='green', alpha=0.5)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.set_yscale('log')
    
    # C. Categorical Risk
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title("Category Risk Factors", fontsize=14)
    
    cats = ['location', 'device_type', 'merchant_category']
    risk_vals = []
    cat_names = []
    
    for c in cats:
        val = input_data[c]
        if val in df[c].values:
            rate = df[df[c] == val]['is_fraud'].mean() * 100
        else:
            rate = 0
        risk_vals.append(rate)
        cat_names.append(f"{c}\n({val})")
        
    ax3.bar(cat_names, risk_vals, color=['orange', 'purple', 'brown'])
    ax3.set_ylabel("Historical Fraud Rate (%)")
    
    # D. Amount Distribution Context
    ax4 = plt.subplot(2, 3, 4)
    sns.kdeplot(data=df[df['is_fraud']==0]['amount'], fill=True, color='green', label='Normal', ax=ax4)
    sns.kdeplot(data=df[df['is_fraud']==1]['amount'], fill=True, color='red', label='Fraud', ax=ax4)
    ax4.axvline(input_data['amount'], color='blue', linestyle='--', linewidth=2, label='Current')
    ax4.set_xscale('log')
    ax4.set_title("Amount Context")
    ax4.legend()

    # E. Hour Distribution Context
    ax5 = plt.subplot(2, 3, 5)
    sns.histplot(data=df, x='transaction_hour', hue='is_fraud', multiple='fill', bins=24, palette={0: 'green', 1: 'red'}, ax=ax5, legend=False)
    ax5.axvline(input_data['transaction_hour'], color='blue', linestyle='--', linewidth=3)
    ax5.set_title("Hour Risk (Red=Fraud Share)")
    ax5.set_ylabel("Proportion")

    # F. Text Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = (
        f"Input Details:\n"
        f"Amount: ${input_data['amount']}\n"
        f"Loc: {input_data['location']}\n"
        f"Merch: {input_data['merchant_category']}\n"
        f"Hour: {input_data['transaction_hour']}\n\n"
        f"Verdict: {'HIGH RISK' if is_high_risk else 'Low Risk'}\n"
        f"Action: {'Flag for Review' if is_high_risk else 'Approve'}"
    )
    ax6.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=14, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        
    plt.close()
    return image_base64

if __name__ == "__main__":
    # Load model once
    model, df = train_model()

    # --- 1. HIGH RISK INPUT (Suggested) ---
    high_risk_input = {
        'amount': 500.0,
        'location': 'NY',
        'device_type': 'Mobile',
        'merchant_category': 'Electronics',
        'transaction_hour': 3,
        'past_transaction_count': 2,
        'avg_spending': 50.0
    }
    
    # --- 2. LOW RISK INPUT (Suggested) ---
    low_risk_input = {
        'amount': 45.0,
        'location': 'CA',
        'device_type': 'Mobile',
        'merchant_category': 'Groceries',
        'transaction_hour': 14,
        'past_transaction_count': 10,
        'avg_spending': 50.0
    }
    
    create_prediction_report(model, df, high_risk_input, "high_risk_report.png")
    create_prediction_report(model, df, low_risk_input, "low_risk_report.png")
    
    print("\nDONE. Created 'high_risk_report.png' and 'low_risk_report.png'.")
