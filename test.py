# app.py
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Adaptive Apriori Business Recommendation Tool")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    item_col = st.text_input("Item column (leave blank if basket format)")
    transaction_col = st.text_input("Transaction ID column (leave blank if basket format)")

    if st.button("Run Apriori"):
        if item_col and transaction_col:
            basket = df.groupby([transaction_col, item_col]).size().unstack(fill_value=0)
            basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        else:
            basket = df.copy()

        num_transactions = basket.shape[0]
        num_items = basket.shape[1]
        sparsity = 1.0 - (basket.sum().sum() / (num_transactions * num_items))

        if sparsity > 0.9:
            min_support = 0.001
        elif sparsity > 0.8:
            min_support = 0.005
        elif sparsity > 0.6:
            min_support = 0.01
        else:
            min_support = 0.05

        if num_transactions < 100:
            min_support = max(min_support, 0.05)
        elif num_transactions > 100000:
            min_support = min(min_support, 0.005)

        min_confidence = 0.3 if sparsity > 0.9 else (0.4 if sparsity > 0.8 else 0.5)

        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        st.subheader("Detected Parameters")
        st.write(f"Min Support: {min_support}")
        st.write(f"Min Confidence: {min_confidence}")

        st.subheader("Frequent Itemsets")
        st.dataframe(frequent_itemsets)

        st.subheader("Association Rules")
        st.dataframe(rules)

        st.subheader("Business Recommendations")
        for _, row in rules.iterrows():
            ant = ', '.join(list(row['antecedents']))
            con = ', '.join(list(row['consequents']))
            st.write(f"**Rule:** If a customer buys [{ant}], they are likely to also buy [{con}].")
            st.write(f"- Strategic Product Placement: Place [{ant}] near [{con}] in-store or online.")
            st.write(f"- Promotional Bundles: Offer [{ant}] with [{con}] at a discount.")
            st.write(f"- Dynamic Inventory Management: Keep [{con}] stocked when [{ant}] is in demand.")
            st.write(f"- Targeted Marketing: Send offers for [{con}] to buyers of [{ant}].")
            st.write("---")
