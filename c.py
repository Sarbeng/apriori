import streamlit as st
import pandas as pd
from itertools import combinations
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------
# Custom Apriori Implementation
# ---------------------------
class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def fit(self, transactions):
        item_support = defaultdict(int)
        self.transactions = list(map(set, transactions))
        num_transactions = len(self.transactions)

        # 1-itemsets
        for transaction in self.transactions:
            for item in transaction:
                item_support[frozenset([item])] += 1

        current_itemsets = {item for item, count in item_support.items() if count / num_transactions >= self.min_support}
        frequent_itemsets = dict(item_support)
        k = 2

        while current_itemsets:
            candidate_itemsets = set(
                [i.union(j) for i in current_itemsets for j in current_itemsets if len(i.union(j)) == k]
            )
            item_support_k = defaultdict(int)
            for transaction in self.transactions:
                for candidate in candidate_itemsets:
                    if candidate.issubset(transaction):
                        item_support_k[candidate] += 1

            current_itemsets = {item for item, count in item_support_k.items() if count / num_transactions >= self.min_support}
            frequent_itemsets.update(item_support_k)
            k += 1

        self.frequent_itemsets = {
            item: support / num_transactions for item, support in frequent_itemsets.items() if support / num_transactions >= self.min_support
        }

        return self.frequent_itemsets

    def generate_rules(self):
        rules = []
        for itemset in self.frequent_itemsets:
            if len(itemset) < 2:
                continue
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    support_itemset = self.frequent_itemsets[itemset]
                    support_antecedent = self.frequent_itemsets.get(antecedent, 0)
                    confidence = support_itemset / support_antecedent if support_antecedent else 0
                    if confidence >= self.min_confidence:
                        rules.append({
                            'antecedent': set(antecedent),
                            'consequent': set(consequent),
                            'support': round(support_itemset, 4),
                            'confidence': round(confidence, 4)
                        })
        return rules

# ---------------------------
# Adaptive Support/Confidence Finder
# ---------------------------
def find_optimal_params(transactions, start_support=0.2, start_conf=0.6, min_rules=1):
    support = start_support
    confidence = start_conf
    step = 0.05
    max_iterations = 30
    iterations = 0

    while iterations < max_iterations:
        apriori = Apriori(min_support=support, min_confidence=confidence)
        apriori.fit(transactions)
        rules = apriori.generate_rules()
        if len(rules) >= min_rules:
            return support, confidence
        else:
            support -= step
            if support < 0.001:
                support = 0.001
                confidence -= 0.05
            if confidence < 0.1:
                confidence = 0.1
        iterations += 1

    return support, confidence

# ---------------------------
# Business Insights
# ---------------------------
def generate_business_insights(rules):
    insights = []
    for rule in rules:
        items = ", ".join(rule['antecedent'])
        recs = ", ".join(rule['consequent'])
        insights.append(f"If a customer buys [{items}], recommend [{recs}] to increase sales.")

    suggestions = [
        "Strategic Product Placement: Place frequently bought-together items close to each other.",
        "Promotional Bundles: Offer discounts on product combinations that appear in high-confidence rules.",
        "Dynamic Inventory Management: Stock up on products that are often purchased together.",
        "Targeted Marketing: Send personalized offers based on customers’ past purchases."
    ]
    return insights + suggestions

# ---------------------------
# Graph Visualization
# ---------------------------
def plot_rules_graph(rules):
    G = nx.DiGraph()
    for rule in rules:
        for antecedent in rule['antecedent']:
            for consequent in rule['consequent']:
                G.add_edge(antecedent, consequent, weight=rule['confidence'])

    pos = nx.spring_layout(G, k=0.5, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2000, font_size=10, font_weight="bold", arrows=True)
    labels = nx.get_edge_attributes(G, 'weight')
    labels = {k: f"{v:.2f}" for k, v in labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    st.pyplot(plt)

# ---------------------------
# Streamlit App
# ---------------------------
st.title("Market Basket Analysis & Business Recommendations C")

uploaded_file = st.file_uploader("Upload your transaction dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())

    # Select columns
    col1 = st.selectbox("Select Transaction ID column", df.columns)
    col2 = st.selectbox("Select Item column", df.columns)

    if col1 != col2:
        transactions = df.groupby(col1)[col2].apply(list).tolist()

        # Auto-tune support/confidence
        opt_support, opt_conf = find_optimal_params(transactions)

        # Let user adjust
        min_support = st.number_input("Minimum Support", min_value=0.001, max_value=1.0, value=opt_support, step=0.001, format="%.3f")
        min_confidence = st.number_input("Minimum Confidence", min_value=0.1, max_value=1.0, value=opt_conf, step=0.05, format="%.2f")

        # Sliders for quick adjustment
        min_support = st.slider("Adjust Minimum Support", 0.001, 1.0, min_support, 0.001)
        min_confidence = st.slider("Adjust Minimum Confidence", 0.1, 1.0, min_confidence, 0.05)

        # Run Apriori
        apriori = Apriori(min_support=min_support, min_confidence=min_confidence)
        frequent_itemsets = apriori.fit(transactions)
        rules = apriori.generate_rules()

        # Display results
        st.subheader("Frequent Itemsets")
        st.dataframe(pd.DataFrame([
            {"Itemset": list(k), "Support": v} for k, v in frequent_itemsets.items()
        ]))

        st.subheader("Association Rules")
        st.dataframe(pd.DataFrame(rules))

        st.subheader("Business Insights & Recommendations")
        for insight in generate_business_insights(rules):
            st.write("- ", insight)

        st.subheader("Association Rules Graph")
        if rules:
            plot_rules_graph(rules)
        else:
            st.warning("No rules to visualize. Try lowering min_support or min_confidence.")
    else:
        st.error("Please select different columns for Transaction ID and Item.")
