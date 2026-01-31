import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# 1. Dataset Generation
def generate_data(n_rows=5000):
    np.random.seed(42)
    
    locations = ['NY', 'CA', 'TX', 'FL', 'IL']
    device_types = ['Mobile', 'Desktop', 'Tablet']
    merchant_categories = ['Electronics', 'Groceries', 'Clothing', 'Restaurant', 'Travel']
    
    data = {
        'amount': np.random.exponential(scale=100, size=n_rows),
        'location': np.random.choice(locations, size=n_rows),
        'device_type': np.random.choice(device_types, size=n_rows),
        'merchant_category': np.random.choice(merchant_categories, size=n_rows),
        'transaction_hour': np.random.randint(0, 24, size=n_rows),
        'past_transaction_count': np.random.poisson(lam=10, size=n_rows),
        'avg_spending': np.random.normal(loc=50, scale=20, size=n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Generate imbalanced target (2% - 5% fraud)
    fraud_prob = 0.02 + \
                 (df['amount'] > 300) * 0.1 + \
                 (df['transaction_hour'] < 5) * 0.05 + \
                 (df['location'] == 'NY') * 0.03
                 
    fraud_prob = fraud_prob / fraud_prob.max() * 0.2
    random_fraud = np.random.binomial(1, 0.03, size=n_rows)
    df['is_fraud'] = np.where(fraud_prob > np.random.random(n_rows), 1, random_fraud)
    
    return df

def train_model():
    print("Generating Dataset...")
    df = generate_data(5000)
    
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    # 2. Train-Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Model Pipeline
    categorical_features = ['location', 'device_type', 'merchant_category']
    numerical_features = ['amount', 'transaction_hour', 'past_transaction_count', 'avg_spending']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"Training XGBoost with scale_pos_weight: {scale_pos_weight:.2f}")
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            random_state=42
        ))
    ])
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluation
    print("\nModel Evaluation on Test Set:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model, df # Return model and dataframe (for context in reports)

if __name__ == "__main__":
    model, df = train_model()
    
    # Save dataset to CSV for user inspection
    df.to_csv('transactions.csv', index=False)
    print("Dataset saved to 'transactions.csv'")
    
    # --- NEW: VISUALIZATION ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Re-extract features for visualization logic if needed, 
    # but for this script we just want to run the prediction loop.
    # Note: feature_importances plotting removed from here to keep main clean, 
    # but can be added back if requested. For now focusing on User Input.

    
    # 4. User Input Portion
    print("\n" + "="*30)
    print("ENTER TRANSACTION DETAILS FOR PREDICTION")
    print("Format: amount, location, device_type, merchant_category, transaction_hour, past_transaction_count, avg_spending")
    print("Example: 150.5, NY, Mobile, Electronics, 14, 5, 45.0")
    print("="*30)
    
    try:
        user_input_str = input("Input Row: ")
        # user_input_str = "150.5, NY, Mobile, Electronics, 14, 5, 45.0" # Debug line
        
        parts = [x.strip() for x in user_input_str.split(',')]
        if len(parts) != 7:
            print("Error: Expected 7 values.")
        else:
            inputs = {
                'amount': [float(parts[0])],
                'location': [parts[1]],
                'device_type': [parts[2]],
                'merchant_category': [parts[3]],
                'transaction_hour': [int(parts[4])],
                'past_transaction_count': [int(parts[5])],
                'avg_spending': [float(parts[6])]
            }
            
            input_df = pd.DataFrame(inputs)
            
            # Predict Probability
            prob = model.predict_proba(input_df)[0][1]
            prob_pct = prob * 100
            
            # Output
            print(f"\nFraud Probability: {prob_pct:.2f}%")
            
            # --- DYNAMIC VISUALIZATION ---
            plt.figure(figsize=(10, 6))
            plt.suptitle(f"Transaction Analysis\nFraud Probability: {prob_pct:.2f}%", fontsize=16, color='red' if prob > 0.5 else 'green')
            
            # 1. Probability Gauge
            plt.subplot(2, 2, 1)
            plt.barh(['Fraud Probability'], [prob_pct], color='red' if prob > 0.5 else 'green')
            plt.xlim(0, 100)
            plt.xlabel("Probability (%)")
            plt.title("Risk Score")
            
            # 2. Amount Context
            plt.subplot(2, 2, 2)
            sns.histplot(data=df, x='amount', hue='is_fraud', element='step', common_norm=False, palette='Set2')
            plt.axvline(input_df['amount'].values[0], color='red', linestyle='--', label='Your Amount')
            plt.yscale('log')
            plt.title("Amount Distribution Context")
            plt.legend()
            
            # 3. Hour Context
            plt.subplot(2, 2, 3)
            sns.histplot(data=df, x='transaction_hour', hue='is_fraud', multiple='fill', palette='Set2', bins=24)
            plt.axvline(input_df['transaction_hour'].values[0], color='red', linestyle='--', label='Your Hour')
            plt.title("Hour Context (Fraud Ratio)")
            plt.ylabel("Proportion")
            
            # 4. Past Transaction Count Context
            plt.subplot(2, 2, 4)
            sns.kdeplot(data=df[df['is_fraud']==0]['past_transaction_count'], label='Normal', shade=True)
            sns.kdeplot(data=df[df['is_fraud']==1]['past_transaction_count'], label='Fraud', shade=True)
            plt.axvline(input_df['past_transaction_count'].values[0], color='red', linestyle='--', label='Your Count')
            plt.title("Past Tx Count Context")
            plt.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig('prediction_report.png')
            print("Visualization saved to 'prediction_report.png'")            

            
    except Exception as e:
        print(f"An error occurred: {e}")
