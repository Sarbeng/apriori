import pandas as pd
from itertools import combinations
from collections import defaultdict
import time

class AprioriOptimized:
    def __init__(self, min_support=0.5, min_confidence=0.6, max_itemset_size=3, max_candidates=50000):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_itemset_size = max_itemset_size
        self.max_candidates = max_candidates
        self.transactions = []
        self.item_support = {}
        self.rules = []

    def load_transactions_from_list(self, raw_data):
        self.transactions = [set(t) for t in raw_data if t]

    def fit(self):
        start_time = time.time()
        total_transactions = len(self.transactions)

        # Step 1: 1-itemsets
        item_counts = defaultdict(int)
        for transaction in self.transactions:
            for item in transaction:
                item_counts[frozenset([item])] += 1

        # Step 2: Prune infrequent 1-itemsets
        current_frequent = {
            itemset: count / total_transactions
            for itemset, count in item_counts.items()
            if (count / total_transactions) >= self.min_support
        }
        self.item_support.update(current_frequent)

        k = 2
        while current_frequent and k <= self.max_itemset_size:
            prev_frequent_itemsets = list(current_frequent.keys())
            candidates = self.generate_candidates(prev_frequent_itemsets, k)

            if len(candidates) > self.max_candidates:
                print(f"⚠️ Candidate limit reached at size {k}, stopping early.")
                break

            candidate_counts = defaultdict(int)
            for transaction in self.transactions:
                for candidate in candidates:
                    if candidate <= transaction:
                        candidate_counts[candidate] += 1

            current_frequent = {
                itemset: count / total_transactions
                for itemset, count in candidate_counts.items()
                if (count / total_transactions) >= self.min_support
            }
            self.item_support.update(current_frequent)
            k += 1

        self.generate_rules()
        print(f"✅ Finished in {time.time() - start_time:.2f} seconds.")

    def generate_candidates(self, prev_frequent_itemsets, k):
        candidates = set()
        prev_set = set(prev_frequent_itemsets)
        sorted_prev = [sorted(list(itemset)) for itemset in prev_frequent_itemsets]

        for i in range(len(sorted_prev)):
            for j in range(i + 1, len(sorted_prev)):
                l1, l2 = sorted_prev[i], sorted_prev[j]
                if l1[:k - 2] == l2[:k - 2]:
                    candidate = frozenset(set(l1) | set(l2))
                    if all(frozenset(subset) in prev_set for subset in combinations(candidate, k - 1)):
                        candidates.add(candidate)
        return candidates

    def generate_rules(self):
        for itemset in self.item_support:
            if len(itemset) < 2:
                continue
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    if antecedent in self.item_support and consequent in self.item_support:
                        support = self.item_support[itemset]
                        confidence = support / self.item_support[antecedent]
                        lift = confidence / self.item_support[consequent]
                        if confidence >= self.min_confidence:
                            self.rules.append({
                                'antecedent': set(antecedent),
                                'consequent': set(consequent),
                                'support': round(support, 4),
                                'confidence': round(confidence, 4),
                                'lift': round(lift, 4)
                            })

    def get_frequent_itemsets(self):
        return {tuple(item): support for item, support in self.item_support.items()}

    def get_rules(self):
        return self.rules


# -------------------------
# Run on maroun.csv
# -------------------------
df = pd.read_csv("maroun.csv")
item_col = "product_name"
transaction_col = "order_no"

# Preprocess
df[item_col] = df[item_col].astype(str).str.lower().str.strip()
df = df.drop_duplicates(subset=[transaction_col, item_col])
transactions = df.groupby(transaction_col)[item_col].apply(list).tolist()

# Auto-tune parameters
num_transactions = len(transactions)
num_items = len(set(i for t in transactions for i in t))
sparsity = 1.0 - (sum(len(t) for t in transactions) / (num_transactions * num_items))

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
    min_support = max(min_support, 0.01)  # avoid huge candidate space

min_confidence = 0.3 if sparsity > 0.9 else (0.4 if sparsity > 0.8 else 0.5)

# Run optimized Apriori
apriori_opt = AprioriOptimized(min_support=min_support, min_confidence=min_confidence, max_itemset_size=3)
apriori_opt.load_transactions_from_list(transactions)
apriori_opt.fit()

print("Frequent itemsets:", list(apriori_opt.get_frequent_itemsets().items())[:10])
print("Rules:", apriori_opt.get_rules()[:10])
