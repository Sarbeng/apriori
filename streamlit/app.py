import pandas as pd
import numpy as np
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Function to automatically determine min_support
def auto_min_support(transactions):
    avg_transaction_length = np.mean([len(t) for t in transactions])
    unique_items = len(set(item for transaction in transactions for item in transaction))
    
    base_support = 1 / (unique_items * 0.1)
    adjusted_support = base_support * (avg_transaction_length / 5)
    min_support = max(0.001, min(0.2, adjusted_support))
    
    if len(transactions) > 10000:
        min_support = max(0.0005, min_support)
    
    return round(min_support, 4)

# Function to preprocess data with selected columns
def preprocess_data(df, transaction_col, item_col):
    transactions = df.groupby(transaction_col)[item_col].apply(list).values
    return transactions

# Function to generate association rules
def generate_rules(frequent_itemsets, confidence_level=0.5):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence_level)
    return rules

# Function to plot item counts
def plot_item_counts(transactions):
    all_items = [item for transaction in transactions for item in transaction]
    item_counts = pd.Series(all_items).value_counts().head(20)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=item_counts.values, y=item_counts.index, palette='viridis')
    plt.title('Top 20 Most Frequent Items')
    plt.xlabel('Count')
    plt.ylabel('Item')
    st.pyplot(plt)

# Streamlit app
def main():
    st.title("🛒 Market Basket Analysis")
    st.write("Upload your transaction data and select columns for analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a file (CSV or Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except:
            try:
                df = pd.read_excel(uploaded_file)
            except:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return
        
        st.success("✅ Data loaded successfully!")
        st.write("Preview of your data:")
        st.dataframe(df.head())
        
        # Column selection
        st.subheader("🔘 Column Selection")
        cols = df.columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            transaction_col = st.selectbox("Select Transaction ID Column", cols, index=0)
        with col2:
            item_col = st.selectbox("Select Item Column", cols, index=1 if len(cols) > 1 else 0)
        
        # Preprocess data
        transactions = preprocess_data(df, transaction_col, item_col)
        
        if transactions is not None:
            unique_items = len(set(item for transaction in transactions for item in transaction))
            st.success(f"🔢 Processed {len(transactions)} transactions with {unique_items} unique items")
            
            # Plot item counts
            st.subheader("📊 Item Frequency")
            plot_item_counts(transactions)
            
            # Parameters section
            st.subheader("⚙️ Analysis Parameters")
            
            # Auto-calculate min_support
            min_support = auto_min_support(transactions)
            
            # Numeric input for min_support
            min_support = st.number_input(
                "Minimum Support (0-1)",
                min_value=0.0001,
                max_value=0.5,
                value=min_support,
                step=0.001,
                format="%.4f",
                help="Lower values will generate more itemsets but take longer to process"
            )
            
            # Numeric input for min_confidence
            min_confidence = st.number_input(
                "Minimum Confidence (0-1)",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
                format="%.2f",
                help="Higher values will generate stronger but fewer rules"
            )
            
            # Run analysis button
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner('Analyzing...'):
                    # Convert transactions to one-hot encoded format
                    te = TransactionEncoder()
                    te_ary = te.fit(transactions).transform(transactions)
                    encoded_df = pd.DataFrame(te_ary, columns=te.columns_)
                    
                    # Generate frequent itemsets
                    frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True)
                    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
                    
                    if len(frequent_itemsets) == 0:
                        st.warning("No frequent itemsets found. Try lowering the minimum support.")
                    else:
                        st.success(f"Found {len(frequent_itemsets)} frequent itemsets")
                        
                        # Generate association rules
                        rules = generate_rules(frequent_itemsets, min_confidence)
                        
                        if len(rules) == 0:
                            st.warning("No association rules found. Try lowering the minimum confidence.")
                        else:
                            st.success(f"Generated {len(rules)} association rules")
                            
                            # Show top rules
                            st.subheader("🏆 Top Association Rules")
                            tab1, tab2 = st.tabs(["By Confidence", "By Lift"])
                            
                            with tab1:
                                st.dataframe(rules.sort_values('confidence', ascending=False).head(10))
                            
                            with tab2:
                                st.dataframe(rules.sort_values('lift', ascending=False).head(10))
                            
                            # Business recommendations
                            st.subheader("💡 Business Insights")
                            
                            # Strategic Product Placement
                            with st.expander("📍 Strategic Product Placement"):
                                high_lift = rules.sort_values('lift', ascending=False).head(5)
                                for _, row in high_lift.iterrows():
                                    ants = ", ".join(list(row['antecedents']))
                                    cons = ", ".join(list(row['consequents']))
                                    st.write(f"Place **{ants}** near **{cons}** (lift: {row['lift']:.2f})")
                            
                            # Promotional Bundles
                            with st.expander("🎁 Promotional Bundle Ideas"):
                                high_conf = rules.sort_values('confidence', ascending=False).head(5)
                                for _, row in high_conf.iterrows():
                                    ants = ", ".join(list(row['antecedents']))
                                    cons = ", ".join(list(row['consequents']))
                                    st.write(f"Bundle **{ants}** with **{cons}** (confidence: {row['confidence']:.2f})")

if __name__ == "__main__":
    main()