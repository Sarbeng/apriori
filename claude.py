import streamlit as st
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
        """Find optimal minimum support and confidence"""
        n_transactions = len(df_encoded)
        n_items = len(df_encoded.columns)
        
        # Calculate item frequencies
        item_frequencies = df_encoded.mean().sort_values(ascending=False)
        
        # Strategy for finding optimal support
        # Start with different support levels and see which gives reasonable results
        support_candidates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
        
        best_support = 0.01
        best_itemsets_count = 0
        
        for support in support_candidates:
            try:
                temp_itemsets = apriori(df_encoded, min_support=support, use_colnames=True)
                itemsets_count = len(temp_itemsets)
                
                # We want a reasonable number of itemsets (between 10-500)
                if 10 <= itemsets_count <= 500:
                    if itemsets_count > best_itemsets_count:
                        best_support = support
                        best_itemsets_count = itemsets_count
                elif itemsets_count > 500:
                    break  # Support too low
                    
            except:
                continue
        
        # If no good support found, use adaptive approach
        if best_itemsets_count == 0:
            # Use median frequency of top 20% items as support
            top_items = item_frequencies.head(int(n_items * 0.2))
            best_support = max(0.001, top_items.median())
        
        # Find optimal confidence
        # Try to generate rules with different confidence levels
        confidence_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        try:
            temp_itemsets = apriori(df_encoded, min_support=best_support, use_colnames=True)
            
            best_confidence = 0.5
            best_rules_count = 0
            
            for confidence in confidence_candidates:
                try:
                    temp_rules = association_rules(temp_itemsets, metric="confidence", min_threshold=confidence)
                    rules_count = len(temp_rules)
                    
                    # We want a reasonable number of rules (between 5-200)
                    if 5 <= rules_count <= 200:
                        if rules_count > best_rules_count:
                            best_confidence = confidence
                            best_rules_count = rules_count
                    elif rules_count > 200:
                        break  # Confidence too low
                        
                except:
                    continue
                    
        except:
            best_confidence = 0.5
        
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
        """Generate business insights from association rules"""
        insights = {
            'cross_selling': [],
            'product_placement': [],
            'promotional_bundles': [],
            'inventory_management': [],
            'targeted_marketing': []
        }
        
        if rules is not None and len(rules) > 0:
            # Cross-selling opportunities (high confidence rules)
            high_conf_rules = rules[rules['confidence'] > 0.7].head(10)
            for _, rule in high_conf_rules.iterrows():
                antecedents = ', '.join(list(rule['antecedents']))
                consequents = ', '.join(list(rule['consequents']))
                insights['cross_selling'].append({
                    'rule': f"If customer buys {antecedents} → suggest {consequents}",
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'support': rule['support']
                })
            
            # Product placement (items that appear together frequently)
            high_lift_rules = rules[rules['lift'] > 1.5].head(10)
            for _, rule in high_lift_rules.iterrows():
                antecedents = ', '.join(list(rule['antecedents']))
                consequents = ', '.join(list(rule['consequents']))
                insights['product_placement'].append({
                    'items': f"{antecedents} + {consequents}",
                    'lift': rule['lift'],
                    'support': rule['support']
                })
            
            # Promotional bundles (strong associations)
            strong_rules = rules[(rules['confidence'] > 0.5) & (rules['lift'] > 1.2)].head(10)
            for _, rule in strong_rules.iterrows():
                antecedents = ', '.join(list(rule['antecedents']))
                consequents = ', '.join(list(rule['consequents']))
                insights['promotional_bundles'].append({
                    'bundle': f"{antecedents} + {consequents}",
                    'confidence': rule['confidence'],
                    'lift': rule['lift']
                })
        
        if frequent_itemsets is not None and len(frequent_itemsets) > 0:
            # Inventory management (high support single items)
            single_items = frequent_itemsets[frequent_itemsets['length'] == 1].head(10)
            for _, item in single_items.iterrows():
                item_name = ', '.join(list(item['itemsets']))
                insights['inventory_management'].append({
                    'item': item_name,
                    'support': item['support'],
                    'priority': 'High' if item['support'] > 0.1 else 'Medium'
                })
            
            # Targeted marketing (frequent combinations)
            frequent_combinations = frequent_itemsets[frequent_itemsets['length'] >= 2].head(10)
            for _, combo in frequent_combinations.iterrows():
                combo_name = ', '.join(list(combo['itemsets']))
                insights['targeted_marketing'].append({
                    'segment': combo_name,
                    'support': combo['support'],
                    'opportunity': 'Bundle marketing campaign'
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
                        
                        # Generate frequent itemsets
                        frequent_itemsets = analyzer.generate_frequent_itemsets(
                            df_encoded, analyzer.optimal_support
                        )
                        
                        if frequent_itemsets is not None:
                            analyzer.frequent_itemsets = frequent_itemsets
                            
                            # Generate association rules
                            rules = analyzer.generate_association_rules(
                                frequent_itemsets, analyzer.optimal_confidence
                            )
                            
                            if rules is not None:
                                analyzer.rules = rules
                                st.success("✅ Analysis completed successfully!")
                            else:
                                st.warning("⚠️ No association rules found. Try lowering the confidence threshold.")
                        else:
                            st.error("❌ No frequent itemsets found. Try lowering the support threshold.")
        
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
        
        # Create tabs for different business strategies
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Cross-Selling", 
            "📍 Product Placement", 
            "🎁 Promotional Bundles", 
            "📦 Inventory Management", 
            "📢 Targeted Marketing"
        ])
        
        with tab1:
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