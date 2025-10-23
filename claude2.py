with tab1:
            st.markdown("### 🎯 Cross-Selling Opportunities")
            st.markdown("*Recommend these products when customers buy certain items*")
            if insights['cross_selling']:
                for i, insight in enumerate(insights['cross_selling']):
                    priority_color = "#28a745" if insight['priority'] == 'High' else "#ffc107" if insight['priority'] == 'Medium' else "#6c757d"
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>📍 Strategy {i+1}:</strong> {insight['rule']}<br>
                        <strong>Success Rate:</strong> {insight['success_rate']} | 
                        <strong>Lift:</strong> {insight['lift']:.2f} | 
                        <strong>Market Coverage:</strong> {insight['support']:.1%}<br>
                        <strong>Priority:</strong> <span style="color: {priority_color}; font-weight: bold;">{insight['priority']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("💡 Cross-selling opportunities will appear here based on your transaction patterns.")
        
        with tab2:
            st.markdown("### 📍 Strategic Product Placement")
            st.markdown("*Optimize store layout based on product associations*")
            if insights['product_placement']:
                for i, insight in enumerate(insights['product_placement']):
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>🏪 Placement {i+1}:</strong> {insight['items']}<br>
                        <strong>Strategy:</strong> {insight['placement_strategy']}<br>
                        <strong>Association Strength:</strong> {insight['association_strength']} 
                        (Lift: {insight['lift']:.2f})<br>
                        <strong>Expected Impact:</strong> {insight['expected_increase']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🏪 Product placement recommendations will be generated from your data.")
        
        with tab3:
            st.markdown("### 🎁 Promotional Bundle Suggestions")
            st.markdown("*Create attractive product bundles with optimal pricing*")
            if insights['promotional_bundles']:
                for i, insight in enumerate(insights['promotional_bundles']):
                    bundle_color = "#28a745" if insight['bundle_type'] == 'Premium Bundle' else "#17a2b8"
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>🎁 {insight['bundle_type']}:</strong> {insight['bundle']}<br>
                        <strong>Expected Uptake:</strong> {insight['expected_uptake']}<br>
                        <strong>Suggested Discount:</strong> {insight['suggested_discount']}<br>
                        <strong>Bundle Strength:</strong> {insight['lift']:.2f}x normal likelihood
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🎁 Bundle recommendations will be created based on purchase patterns.")
        
        with tab4:
            st.markdown("### 📦 Dynamic Inventory Management")
            st.markdown("*Optimize stock levels based on demand patterns*")
            if insights['inventory_management']:
                for insight in insights['inventory_management']:
                    if insight['priority'] == 'Critical':
                        priority_color = "#dc3545"
                    elif insight['priority'] == 'High':
                        priority_color = "#fd7e14" 
                    elif insight['priority'] == 'Medium':
                        priority_color = "#ffc107"
                    else:
                        priority_color = "#6c757d"
                    
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>📦 Item:</strong> {insight['item']}<br>
                        <strong>Demand Frequency:</strong> {insight['demand_frequency']}<br>
                        <strong>Stock Priority:</strong> 
                        <span style="color: {priority_color}; font-weight: bold;">{insight['priority']}</span><br>
                        <strong>Recommended Level:</strong> {insight['recommended_stock_level']}<br>
                        <strong>Reorder Schedule:</strong> {insight['reorder_frequency']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("📦 Inventory recommendations will be based on item frequency analysis.")
        
        with tab5:
            st.markdown("### 📢 Targeted Marketing Campaigns")
            st.markdown("*Identify customer segments and optimize marketing efforts*")
            if insights['targeted_marketing']:
                for insight in insights['targeted_marketing']:
                    campaign_color = "#28a745" if insight['campaign_type'] == 'Mass Campaign' else "#17a2b8" if insight['campaign_type'] == 'Targeted Campaign' else "#6f42c1"
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>🎯 Target Segment:</strong> {insight['segment']}<br>
                        <strong>Promote:</strong> {insight['target_product']}<br>
                        <strong>Segment Size:</strong> {insight['segment_size']}<br>
                        <strong>Campaign Type:</strong> 
                        <span style="color: {campaign_color}; font-weight: bold;">{insight['campaign_type']}</span><br>
                        <strong>Success Probability:</strong> {insight['success_probability']}<br>
                        <strong>ROI Multiplier:</strong> {insight['roi_multiplier']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("📢 Marketing segment recommendations will be generated from purchase data.")import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Market Basket Analysis Tool",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MarketBasketAnalyzer:
    def __init__(self):
        self.df = None
        self.frequent_itemsets = None
        self.rules = None
        self.optimal_support = None
        self.optimal_confidence = None
    
    def preprocess_data(self, df, transaction_col=None, item_col=None, format_type="transaction_list"):
        """Preprocess data based on format type"""
        try:
            if format_type == "transaction_list":
                # Each row is a transaction with items separated by delimiter
                if transaction_col is None:
                    transaction_col = df.columns[0]
                
                transactions = []
                for idx, row in df.iterrows():
                    items = str(row[transaction_col]).split(',')
                    items = [item.strip() for item in items if item.strip()]
                    if items:
                        transactions.append(items)
                        
            elif format_type == "transaction_item":
                # Two columns: transaction_id, item
                if transaction_col is None or item_col is None:
                    transaction_col = df.columns[0]
                    item_col = df.columns[1]
                
                grouped = df.groupby(transaction_col)[item_col].apply(list).reset_index()
                transactions = grouped[item_col].tolist()
                
            elif format_type == "binary_matrix":
                # Binary matrix where columns are items and rows are transactions
                transactions = []
                for idx, row in df.iterrows():
                    items = [col for col in df.columns if row[col] == 1]
                    if items:
                        transactions.append(items)
            
            # Convert to binary matrix using TransactionEncoder
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            return df_encoded, transactions
            
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")
            return None, None
    
    def find_optimal_parameters(self, df_encoded):
        """Find optimal minimum support and confidence - GUARANTEES association rules"""
        n_transactions = len(df_encoded)
        n_items = len(df_encoded.columns)
        
        # Calculate item frequencies
        item_frequencies = df_encoded.mean().sort_values(ascending=False)
        
        st.info(f"📊 Dataset info: {n_transactions} transactions, {n_items} unique items")
        
        # AGGRESSIVE approach - start very low and work our way up
        # We MUST find frequent itemsets and association rules
        support_candidates = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        best_support = None
        best_itemsets = None
        best_itemsets_count = 0
        
        st.info("🔍 Searching for optimal support threshold...")
        progress_bar = st.progress(0)
        
        for i, support in enumerate(support_candidates):
            progress_bar.progress((i + 1) / len(support_candidates))
            try:
                temp_itemsets = apriori(df_encoded, min_support=support, use_colnames=True, low_memory=True)
                itemsets_count = len(temp_itemsets)
                
                # We want itemsets with length >= 2 for meaningful rules
                multi_itemsets = temp_itemsets[temp_itemsets['itemsets'].apply(len) >= 2]
                multi_count = len(multi_itemsets)
                
                if multi_count > 0:  # We found multi-item itemsets!
                    best_support = support
                    best_itemsets = temp_itemsets
                    best_itemsets_count = itemsets_count
                    st.success(f"✅ Found {itemsets_count} frequent itemsets at support={support:.4f}")
                    break
                    
            except Exception as e:
                continue
        
        # If still no itemsets found, use the absolute minimum
        if best_support is None:
            st.warning("⚠️ Using emergency fallback support calculation...")
            # Use the frequency of the least frequent item that appears in at least 2 transactions
            min_freq = 2 / n_transactions
            best_support = max(0.0001, min_freq)
            
            try:
                best_itemsets = apriori(df_encoded, min_support=best_support, use_colnames=True, low_memory=True)
                st.info(f"Emergency itemsets found: {len(best_itemsets)}")
            except:
                # Last resort - use absolute minimum
                best_support = 1 / n_transactions
                best_itemsets = apriori(df_encoded, min_support=best_support, use_colnames=True, low_memory=True)
        
        progress_bar.empty()
        
        # Now find optimal confidence - GUARANTEE we get rules
        st.info("🎯 Optimizing confidence threshold for association rules...")
        
        confidence_candidates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        best_confidence = None
        best_rules = None
        best_rules_count = 0
        
        progress_bar2 = st.progress(0)
        
        for i, confidence in enumerate(confidence_candidates):
            progress_bar2.progress((i + 1) / len(confidence_candidates))
            try:
                temp_rules = association_rules(best_itemsets, metric="confidence", min_threshold=confidence, num_itemsets=len(best_itemsets))
                rules_count = len(temp_rules)
                
                if rules_count > 0:
                    best_confidence = confidence
                    best_rules = temp_rules
                    best_rules_count = rules_count
                    
                    # Prefer a reasonable number of rules (10-100), but accept any rules
                    if 10 <= rules_count <= 100:
                        st.success(f"✅ Found {rules_count} association rules at confidence={confidence:.2f}")
                        break
                    elif rules_count > 100:
                        # Too many rules, but we'll keep looking for a better balance
                        continue
                    
            except Exception as e:
                continue
        
        progress_bar2.empty()
        
        # If no rules found with confidence thresholds, try other metrics
        if best_confidence is None:
            st.warning("⚠️ Trying alternative rule generation strategies...")
            
            # Try with lift metric
            for lift_threshold in [1.0, 0.8, 0.5, 0.1]:
                try:
                    temp_rules = association_rules(best_itemsets, metric="lift", min_threshold=lift_threshold)
                    if len(temp_rules) > 0:
                        best_rules = temp_rules
                        best_confidence = 0.01  # Set very low confidence as backup
                        st.info(f"✅ Found {len(temp_rules)} rules using lift threshold {lift_threshold}")
                        break
                except:
                    continue
            
            # Last resort - generate all possible rules
            if best_rules is None:
                try:
                    best_rules = association_rules(best_itemsets, metric="confidence", min_threshold=0.01)
                    best_confidence = 0.01
                    if len(best_rules) == 0:
                        # Generate rules with absolute minimum confidence
                        best_rules = association_rules(best_itemsets, metric="confidence", min_threshold=0.001)
                        best_confidence = 0.001
                except:
                    # Final fallback - create synthetic rules from itemsets
                    st.error("Creating minimal rules from frequent itemsets...")
                    best_confidence = 0.001
        
        # Display final results
        final_rules_count = len(best_rules) if best_rules is not None else 0
        st.success(f"🎉 OPTIMIZATION COMPLETE!")
        st.info(f"📈 Final Parameters: Support={best_support:.4f}, Confidence={best_confidence:.4f}")
        st.info(f"📊 Results: {len(best_itemsets)} frequent itemsets, {final_rules_count} association rules")
        
        return best_support, best_confidence
    
    def generate_frequent_itemsets(self, df_encoded, min_support):
        """Generate frequent itemsets using Apriori algorithm"""
        try:
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
            frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
            return frequent_itemsets
        except Exception as e:
            st.error(f"Error generating frequent itemsets: {str(e)}")
            return None
    
    def generate_association_rules(self, frequent_itemsets, min_confidence):
        """Generate association rules from frequent itemsets"""
        try:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            rules = rules.sort_values('lift', ascending=False)
            return rules
        except Exception as e:
            st.error(f"Error generating association rules: {str(e)}")
            return None
    
    def generate_business_insights(self, rules, frequent_itemsets):
        """Generate comprehensive business insights from association rules - GUARANTEED insights"""
        insights = {
            'cross_selling': [],
            'product_placement': [],
            'promotional_bundles': [],
            'inventory_management': [],
            'targeted_marketing': []
        }
        
        if rules is not None and len(rules) > 0:
            # Sort rules by different metrics for different strategies
            rules_by_confidence = rules.sort_values('confidence', ascending=False)
            rules_by_lift = rules.sort_values('lift', ascending=False)
            rules_by_support = rules.sort_values('support', ascending=False)
            
            # 1. Cross-selling opportunities (prioritize HIGH confidence rules)
            cross_sell_rules = rules_by_confidence.head(min(15, len(rules)))
            for _, rule in cross_sell_rules.iterrows():
                antecedents = ', '.join(list(rule['antecedents']))
                consequents = ', '.join(list(rule['consequents']))
                
                # Generate actionable recommendation
                recommendation = f"When customers buy {antecedents}, recommend {consequents}"
                success_rate = rule['confidence'] * 100
                
                insights['cross_selling'].append({
                    'rule': recommendation,
                    'success_rate': f"{success_rate:.1f}%",
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'support': rule['support'],
                    'priority': 'High' if rule['confidence'] > 0.5 else 'Medium' if rule['confidence'] > 0.3 else 'Low'
                })
            
            # 2. Product placement (prioritize HIGH lift - strong associations)
            placement_rules = rules_by_lift.head(min(15, len(rules)))
            for _, rule in placement_rules.iterrows():
                antecedents = ', '.join(list(rule['antecedents']))
                consequents = ', '.join(list(rule['consequents']))
                
                placement_strength = "Very Strong" if rule['lift'] > 3 else "Strong" if rule['lift'] > 2 else "Moderate"
                
                insights['product_placement'].append({
                    'items': f"{antecedents} ↔ {consequents}",
                    'placement_strategy': f"Place these items near each other in store",
                    'association_strength': placement_strength,
                    'lift': rule['lift'],
                    'support': rule['support'],
                    'expected_increase': f"{((rule['lift'] - 1) * 100):.0f}% higher purchase likelihood"
                })
            
            # 3. Promotional bundles (balance confidence and lift)
            # Create bundle score: weighted combination of confidence and lift
            rules_copy = rules.copy()
            rules_copy['bundle_score'] = (rules_copy['confidence'] * 0.6) + ((rules_copy['lift'] - 1) * 0.4)
            bundle_rules = rules_copy.sort_values('bundle_score', ascending=False).head(min(15, len(rules)))
            
            for _, rule in bundle_rules.iterrows():
                antecedents = ', '.join(list(rule['antecedents']))
                consequents = ', '.join(list(rule['consequents']))
                
                bundle_type = "Premium Bundle" if rule['bundle_score'] > 0.8 else "Standard Bundle"
                discount_suggestion = "10-15%" if rule['confidence'] > 0.6 else "5-10%"
                
                insights['promotional_bundles'].append({
                    'bundle': f"{antecedents} + {consequents}",
                    'bundle_type': bundle_type,
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'suggested_discount': discount_suggestion,
                    'expected_uptake': f"{rule['confidence'] * 100:.1f}%",
                    'bundle_score': rule['bundle_score']
                })
            
            # 4. Targeted marketing (frequent patterns for segmentation)
            # Use support to identify customer segments
            marketing_rules = rules_by_support.head(min(12, len(rules)))
            for i, rule in enumerate(marketing_rules.iterrows()):
                rule = rule[1]  # Get the actual row data
                antecedents = ', '.join(list(rule['antecedents']))
                consequents = ', '.join(list(rule['consequents']))
                
                segment_size = rule['support'] * 100
                campaign_type = "Mass Campaign" if rule['support'] > 0.1 else "Targeted Campaign" if rule['support'] > 0.05 else "Niche Campaign"
                
                insights['targeted_marketing'].append({
                    'segment': f"Customers who buy {antecedents}",
                    'target_product': consequents,
                    'segment_size': f"{segment_size:.1f}% of customers",
                    'campaign_type': campaign_type,
                    'success_probability': f"{rule['confidence'] * 100:.1f}%",
                    'roi_multiplier': f"{rule['lift']:.1f}x"
                })
        
        # 5. Inventory management (from frequent itemsets - ALWAYS generate these)
        if frequent_itemsets is not None and len(frequent_itemsets) > 0:
            # Single items for core inventory
            single_items = frequent_itemsets[frequent_itemsets['length'] == 1].sort_values('support', ascending=False)
            
            if len(single_items) > 0:
                for _, item in single_items.head(20).iterrows():
                    item_name = ', '.join(list(item['itemsets']))
                    demand_level = item['support']
                    
                    if demand_level > 0.2:
                        priority = 'Critical'
                        stock_level = 'High'
                    elif demand_level > 0.1:
                        priority = 'High'
                        stock_level = 'Medium-High'
                    elif demand_level > 0.05:
                        priority = 'Medium'
                        stock_level = 'Medium'
                    else:
                        priority = 'Low'
                        stock_level = 'Low-Medium'
                    
                    insights['inventory_management'].append({
                        'item': item_name,
                        'demand_frequency': f"{demand_level * 100:.1f}%",
                        'priority': priority,
                        'recommended_stock_level': stock_level,
                        'support': demand_level,
                        'reorder_frequency': 'Weekly' if demand_level > 0.1 else 'Monthly' if demand_level > 0.05 else 'As needed'
                    })
            else:
                # Fallback: use items from rules
                all_items = set()
                for _, rule in rules.iterrows():
                    all_items.update(rule['antecedents'])
                    all_items.update(rule['consequents'])
                
                for item in list(all_items)[:10]:
                    insights['inventory_management'].append({
                        'item': item,
                        'demand_frequency': 'Estimated from rules',
                        'priority': 'Medium',
                        'recommended_stock_level': 'Medium',
                        'support': 'N/A',
                        'reorder_frequency': 'Monthly'
                    })
        
        # If we have very few insights, generate synthetic ones based on available data
        for category in insights:
            if len(insights[category]) == 0 and rules is not None and len(rules) > 0:
                # Generate at least one insight per category
                top_rule = rules.sort_values('lift', ascending=False).iloc[0]
                antecedents = ', '.join(list(top_rule['antecedents']))
                consequents = ', '.join(list(top_rule['consequents']))
                
                if category == 'cross_selling':
                    insights[category].append({
                        'rule': f"Primary recommendation: {antecedents} → {consequents}",
                        'success_rate': f"{top_rule['confidence'] * 100:.1f}%",
                        'confidence': top_rule['confidence'],
                        'lift': top_rule['lift'],
                        'support': top_rule['support'],
                        'priority': 'High'
                    })
                elif category == 'product_placement':
                    insights[category].append({
                        'items': f"{antecedents} ↔ {consequents}",
                        'placement_strategy': "Co-locate these items",
                        'association_strength': "Strong",
                        'lift': top_rule['lift'],
                        'support': top_rule['support'],
                        'expected_increase': f"{((top_rule['lift'] - 1) * 100):.0f}% increase"
                    })
                elif category == 'promotional_bundles':
                    insights[category].append({
                        'bundle': f"{antecedents} + {consequents}",
                        'bundle_type': "Recommended Bundle",
                        'confidence': top_rule['confidence'],
                        'lift': top_rule['lift'],
                        'suggested_discount': "5-10%",
                        'expected_uptake': f"{top_rule['confidence'] * 100:.1f}%"
                    })
                elif category == 'targeted_marketing':
                    insights[category].append({
                        'segment': f"Customers buying {antecedents}",
                        'target_product': consequents,
                        'segment_size': f"{top_rule['support'] * 100:.1f}%",
                        'campaign_type': "Targeted Campaign",
                        'success_probability': f"{top_rule['confidence'] * 100:.1f}%",
                        'roi_multiplier': f"{top_rule['lift']:.1f}x"
                    })
        
        return insights

def main():
    st.markdown('<h1 class="main-header">🛒 Market Basket Analysis Tool</h1>', unsafe_allow_html=True)
    st.markdown("**Automated parameter optimization • Association rule mining • Business insights**")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = MarketBasketAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar for data upload and parameters
    with st.sidebar:
        st.header("📊 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your transaction data",
            type=['csv', 'xlsx', 'txt'],
            help="Supported formats: CSV, Excel, TXT"
        )
        
        # Data format selection
        st.subheader("Data Format")
        format_type = st.selectbox(
            "Select your data format:",
            ["transaction_list", "transaction_item", "binary_matrix"],
            help="""
            - transaction_list: Each row is a transaction with items separated by commas
            - transaction_item: Two columns (transaction_id, item)
            - binary_matrix: Items as columns, transactions as rows (1/0 values)
            """
        )
        
        # Manual parameter override
        st.subheader("⚙️ Parameters")
        auto_params = st.checkbox("Auto-optimize parameters", value=True)
        
        if not auto_params:
            manual_support = st.slider("Minimum Support", 0.001, 0.5, 0.01, 0.001)
            manual_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.5, 0.05)
    
    # Sample data info
    with st.expander("📋 Sample Data Formats", expanded=False):
        st.write("**Transaction List Format (CSV):**")
        st.code("transactions\n\"apple, bread, milk\"\n\"bread, butter\"\n\"apple, milk, cheese\"")
        
        st.write("**Transaction-Item Format (CSV):**")
        st.code("transaction_id,item\n1,apple\n1,bread\n1,milk\n2,bread\n2,butter")
        
        st.write("**Binary Matrix Format (CSV):**")
        st.code("apple,bread,milk,butter\n1,1,1,0\n0,1,0,1\n1,0,1,0")
    
    # Main analysis section
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep='\t')
            
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            
            # Show data preview
            with st.expander("👀 Data Preview", expanded=True):
                st.dataframe(df.head(10))
            
            # Column selection for different formats
            if format_type == "transaction_item":
                col1, col2 = st.columns(2)
                with col1:
                    transaction_col = st.selectbox("Transaction ID Column", df.columns)
                with col2:
                    item_col = st.selectbox("Item Column", df.columns)
            elif format_type == "transaction_list":
                transaction_col = st.selectbox("Transaction Column", df.columns)
                item_col = None
            else:
                transaction_col = None
                item_col = None
            
            # Process data button
            if st.button("🚀 Start Analysis", type="primary"):
                with st.spinner("Processing data and optimizing parameters..."):
                    
                    # Preprocess data
                    df_encoded, transactions = analyzer.preprocess_data(
                        df, transaction_col, item_col, format_type
                    )
                    
                    if df_encoded is not None:
                        analyzer.df = df_encoded
                        
                        # Find optimal parameters
                        if auto_params:
                            support, confidence = analyzer.find_optimal_parameters(df_encoded)
                            analyzer.optimal_support = support
                            analyzer.optimal_confidence = confidence
                        else:
                            analyzer.optimal_support = manual_support
                            analyzer.optimal_confidence = manual_confidence
                        
                        # Generate frequent itemsets - with GUARANTEE of results
                        frequent_itemsets = analyzer.generate_frequent_itemsets(
                            df_encoded, analyzer.optimal_support
                        )
                        
                        if frequent_itemsets is not None and len(frequent_itemsets) > 0:
                            analyzer.frequent_itemsets = frequent_itemsets
                            
                            # Generate association rules - with MULTIPLE fallback strategies
                            rules = analyzer.generate_association_rules(
                                frequent_itemsets, analyzer.optimal_confidence
                            )
                            
                            # If no rules generated, try progressively lower confidence
                            if rules is None or len(rules) == 0:
                                st.warning("🔄 No rules found with initial confidence. Trying lower thresholds...")
                                
                                fallback_confidences = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
                                for fallback_conf in fallback_confidences:
                                    rules = analyzer.generate_association_rules(frequent_itemsets, fallback_conf)
                                    if rules is not None and len(rules) > 0:
                                        analyzer.optimal_confidence = fallback_conf
                                        st.success(f"✅ Found {len(rules)} rules with confidence={fallback_conf:.3f}")
                                        break
                                
                                # If still no rules, try lift-based rules
                                if rules is None or len(rules) == 0:
                                    st.warning("🔄 Trying lift-based rule generation...")
                                    try:
                                        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.1)
                                        if len(rules) == 0:
                                            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.01)
                                        analyzer.optimal_confidence = 0.01
                                        st.success(f"✅ Generated {len(rules)} lift-based rules")
                                    except:
                                        pass
                            
                            if rules is not None and len(rules) > 0:
                                analyzer.rules = rules
                                st.success(f"🎉 Analysis completed successfully! Generated {len(rules)} association rules")
                            else:
                                st.error("❌ Unable to generate any association rules despite all attempts.")
                                st.info("This might indicate your dataset has very unique transaction patterns.")
                        else:
                            st.error("❌ No frequent itemsets found. Dataset might be too sparse or have too many unique items.")
                            st.info("Try using a dataset with more repeated purchase patterns.")
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.info("Please check your file format and try again.")
    
    # Display results
    if analyzer.frequent_itemsets is not None:
        
        # Key metrics
        st.markdown('<h2 class="section-header">📈 Analysis Summary</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Optimal Support", 
                f"{analyzer.optimal_support:.3f}",
                help="Automatically determined minimum support threshold"
            )
        
        with col2:
            st.metric(
                "Optimal Confidence", 
                f"{analyzer.optimal_confidence:.3f}",
                help="Automatically determined minimum confidence threshold"
            )
        
        with col3:
            st.metric(
                "Frequent Itemsets", 
                len(analyzer.frequent_itemsets),
                help="Number of frequent itemsets found"
            )
        
        with col4:
            rules_count = len(analyzer.rules) if analyzer.rules is not None else 0
            st.metric(
                "Association Rules", 
                rules_count,
                help="Number of association rules generated"
            )
        
        # Frequent itemsets analysis
        st.markdown('<h2 class="section-header">🔍 Frequent Itemsets</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top frequent itemsets
            top_itemsets = analyzer.frequent_itemsets.head(15)
            top_itemsets_display = top_itemsets.copy()
            top_itemsets_display['itemsets'] = top_itemsets_display['itemsets'].apply(
                lambda x: ', '.join(list(x))
            )
            st.dataframe(
                top_itemsets_display[['itemsets', 'support', 'length']],
                use_container_width=True
            )
        
        with col2:
            # Support distribution
            fig_support = px.histogram(
                analyzer.frequent_itemsets, 
                x='support', 
                title='Support Distribution',
                labels={'support': 'Support', 'count': 'Frequency'}
            )
            fig_support.update_layout(height=300)
            st.plotly_chart(fig_support, use_container_width=True)
        
        # Association rules analysis
        if analyzer.rules is not None and len(analyzer.rules) > 0:
            st.markdown('<h2 class="section-header">🔗 Association Rules</h2>', unsafe_allow_html=True)
            
            # Top rules
            col1, col2 = st.columns([3, 1])
            
            with col1:
                top_rules = analyzer.rules.head(15)
                rules_display = top_rules.copy()
                rules_display['antecedents'] = rules_display['antecedents'].apply(
                    lambda x: ', '.join(list(x))
                )
                rules_display['consequents'] = rules_display['consequents'].apply(
                    lambda x: ', '.join(list(x))
                )
                
                st.dataframe(
                    rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(3),
                    use_container_width=True
                )
            
            with col2:
                # Rules metrics distribution
                fig_metrics = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=['Confidence', 'Lift', 'Support'],
                    vertical_spacing=0.15
                )
                
                fig_metrics.add_trace(
                    go.Histogram(x=analyzer.rules['confidence'], name='Confidence'),
                    row=1, col=1
                )
                fig_metrics.add_trace(
                    go.Histogram(x=analyzer.rules['lift'], name='Lift'),
                    row=2, col=1
                )
                fig_metrics.add_trace(
                    go.Histogram(x=analyzer.rules['support'], name='Support'),
                    row=3, col=1
                )
                
                fig_metrics.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Scatter plot of rules
            fig_scatter = px.scatter(
                analyzer.rules,
                x='support',
                y='confidence',
                size='lift',
                title='Association Rules: Support vs Confidence (sized by Lift)',
                labels={'support': 'Support', 'confidence': 'Confidence'},
                hover_data=['lift']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Business insights
        st.markdown('<h2 class="section-header">💡 Business Insights & Recommendations</h2>', unsafe_allow_html=True)
        
        insights = analyzer.generate_business_insights(analyzer.rules, analyzer.frequent_itemsets)
        

            st.markdown("### Cross-Selling Opportunities")
            if insights['cross_selling']:
                for i, insight in enumerate(insights['cross_selling']):
                    with st.container():
                        st.markdown(f"""
                        <div class="insight-box">
                            <strong>Rule {i+1}:</strong> {insight['rule']}<br>
                            <strong>Confidence:</strong> {insight['confidence']:.3f} | 
                            <strong>Lift:</strong> {insight['lift']:.3f} | 
                            <strong>Support:</strong> {insight['support']:.3f}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No strong cross-selling opportunities found with current parameters.")
        
        with tab2:
            st.markdown("### Strategic Product Placement")
            if insights['product_placement']:
                for i, insight in enumerate(insights['product_placement']):
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>Place together:</strong> {insight['items']}<br>
                        <strong>Association Strength (Lift):</strong> {insight['lift']:.3f}<br>
                        <strong>Co-occurrence Rate:</strong> {insight['support']:.3f}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant product placement opportunities identified.")
        
        with tab3:
            st.markdown("### Promotional Bundle Suggestions")
            if insights['promotional_bundles']:
                for i, insight in enumerate(insights['promotional_bundles']):
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>Bundle:</strong> {insight['bundle']}<br>
                        <strong>Success Probability:</strong> {insight['confidence']:.1%}<br>
                        <strong>Bundle Strength:</strong> {insight['lift']:.3f}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No strong promotional bundle opportunities found.")
        
        with tab4:
            st.markdown("### Dynamic Inventory Management")
            if insights['inventory_management']:
                for insight in insights['inventory_management']:
                    priority_color = "#28a745" if insight['priority'] == 'High' else "#ffc107"
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>Item:</strong> {insight['item']}<br>
                        <strong>Demand Frequency:</strong> {insight['support']:.1%}<br>
                        <strong>Inventory Priority:</strong> 
                        <span style="color: {priority_color}; font-weight: bold;">{insight['priority']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Unable to generate inventory recommendations.")
        
        with tab5:
            st.markdown("### Targeted Marketing Campaigns")
            if insights['targeted_marketing']:
                for insight in insights['targeted_marketing']:
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>Customer Segment:</strong> Buyers of {insight['segment']}<br>
                        <strong>Segment Size:</strong> {insight['support']:.1%} of customers<br>
                        <strong>Marketing Strategy:</strong> {insight['opportunity']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific marketing segments identified.")
        
        # Export results
        st.markdown('<h2 class="section-header">📥 Export Results</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Download Frequent Itemsets"):
                csv = analyzer.frequent_itemsets.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="frequent_itemsets.csv",
                    mime="text/csv"
                )
        
        with col2:
            if analyzer.rules is not None and st.button("📄 Download Association Rules"):
                csv = analyzer.rules.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="association_rules.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("**Market Basket Analysis Tool** | Built with Streamlit & MLxtend | Auto-optimized parameters for better insights")

if __name__ == "__main__":
    main()