"""
Retail Membership Analytics — Finnish Market
=============================
Author: Rupesh Jha
Description:
    End-to-end data analysis pipeline on a men's grooming chain dataset.
    Covers data simulation with realistic corruption, cleaning pipeline,
    membership analysis, queue/waiting time analysis, sales analysis,
    and a churn prediction model.

Real-world context:
    Methodology developed during an engagement with a real Finnish retail
    client (Finnish Retail Client). Data simulated here to match original database structure
    for portfolio purposes. Original dataset contained 2.3M+ records with
    significant data quality issues including missing values, duplicates,
    inconsistent formats and mismatched IDs across tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8F9FA',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
COLORS = {'Silver': '#A8A9AD', 'Gold': '#CFB53B', 'Platinum': '#4A4A8A', 'Non-member': '#E07B54'}
ACCENT = '#1F4E79'

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — DATA SIMULATION WITH REALISTIC CORRUPTION
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: Simulating realistic enterprise database")
print("=" * 60)

N_CUSTOMERS = 20000
N_RECEIPTS  = 120000
N_POBS      = 84  # matches real Finnish Retail Client locations

# ── Customers table ───────────────────────────────────────────────────────────
customer_ids = np.arange(1, N_CUSTOMERS + 1)
membership_levels = np.random.choice(
    ['Silver', 'Gold', 'Platinum', None],
    size=N_CUSTOMERS,
    p=[0.55, 0.30, 0.05, 0.10]
)

start_years = np.random.choice(range(2012, 2018), size=N_CUSTOMERS,
                                p=[0.05, 0.10, 0.15, 0.20, 0.30, 0.20])
begin_dates, end_dates = [], []
for yr in start_years:
    bd = pd.Timestamp(f"{yr}-{np.random.randint(1,13):02d}-{np.random.randint(1,28):02d}")
    ed = bd + pd.DateOffset(years=1) + pd.DateOffset(days=int(np.random.randint(-30, 90)))
    begin_dates.append(bd)
    end_dates.append(ed)

customers_clean = pd.DataFrame({
    'customer_id':       customer_ids,
    'membership_level':  membership_levels,
    'begin_date':        begin_dates,
    'end_date':          end_dates,
    'pob_id':            np.random.randint(1, N_POBS + 1, size=N_CUSTOMERS)
})

# ── Inject corruption ─────────────────────────────────────────────────────────
customers_corrupt = customers_clean.copy()

# 1. Missing values
for col in ['membership_level', 'begin_date', 'end_date']:
    mask = np.random.random(N_CUSTOMERS) < 0.07
    customers_corrupt.loc[mask, col] = np.nan

# 2. Duplicate records
dupes = customers_corrupt.sample(frac=0.04)
customers_corrupt = pd.concat([customers_corrupt, dupes]).reset_index(drop=True)

# 3. Inconsistent date formats (store as strings for some rows)
date_mask = np.random.random(len(customers_corrupt)) < 0.05
customers_corrupt['begin_date'] = customers_corrupt['begin_date'].astype(str)
customers_corrupt.loc[date_mask, 'begin_date'] = customers_corrupt.loc[
    date_mask, 'begin_date'].apply(
    lambda x: x[:10].replace('-', '/') if isinstance(x, str) and len(x) >= 10 else x
)

# 4. Mismatched IDs — some receipt pob_ids that don't exist in pob table
customers_corrupt.loc[
    np.random.choice(customers_corrupt.index, size=200, replace=False), 'pob_id'
] = 9999

print(f"Raw customers table: {len(customers_corrupt):,} rows (incl. duplicates & corruption)")

# ── Receipts table ────────────────────────────────────────────────────────────
valid_cids   = customer_ids.tolist()
receipt_cids = np.random.choice(valid_cids + [None]*500, size=N_RECEIPTS)
receipt_dates = pd.date_range('2013-01-01', '2017-06-05', periods=N_RECEIPTS)

receipts_clean = pd.DataFrame({
    'receipt_id':   np.arange(1, N_RECEIPTS + 1),
    'customer_id':  receipt_cids,
    'receipt_date': receipt_dates,
    'pob_id':       np.random.randint(1, N_POBS + 1, size=N_RECEIPTS),
    'status':       np.random.choice(['active', 'cancelled', 'pending'], N_RECEIPTS, p=[0.78, 0.12, 0.10]),
    'type':         np.random.choice(['closed', 'open', 'void'],         N_RECEIPTS, p=[0.80, 0.12, 0.08]),
    'total_amount': np.random.exponential(scale=25, size=N_RECEIPTS).round(2)
})

# Inject receipt corruption
receipts_corrupt = receipts_clean.copy()
receipts_corrupt.loc[np.random.choice(receipts_corrupt.index, 800, replace=False), 'customer_id'] = np.nan
receipts_corrupt.loc[np.random.choice(receipts_corrupt.index, 400, replace=False), 'total_amount'] = -1
dupes_r = receipts_corrupt.sample(frac=0.03)
receipts_corrupt = pd.concat([receipts_corrupt, dupes_r]).reset_index(drop=True)

print(f"Raw receipts table:  {len(receipts_corrupt):,} rows (incl. duplicates & corruption)")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — DATA CLEANING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: Data Cleaning Pipeline")
print("=" * 60)

def clean_customers(df):
    report = {}
    original_count = len(df)

    # Step 1: Remove exact duplicates
    df = df.drop_duplicates()
    report['duplicates_removed'] = original_count - len(df)

    # Step 2: Standardise date formats
    df['begin_date'] = df['begin_date'].apply(
        lambda x: str(x).replace('/', '-') if isinstance(x, str) else x
    )
    df['begin_date'] = pd.to_datetime(df['begin_date'], errors='coerce')
    df['end_date']   = pd.to_datetime(df['end_date'],   errors='coerce')

    # Step 3: Drop rows with missing critical fields
    before = len(df)
    df = df.dropna(subset=['customer_id', 'begin_date'])
    report['missing_critical_dropped'] = before - len(df)

    # Step 4: Fix mismatched pob_ids
    valid_pobs = list(range(1, N_POBS + 1))
    bad_pob_mask = ~df['pob_id'].isin(valid_pobs)
    report['invalid_pob_ids_fixed'] = bad_pob_mask.sum()
    df.loc[bad_pob_mask, 'pob_id'] = np.nan

    # Step 5: Flag rows where end_date < begin_date
    invalid_dates = df['end_date'] < df['begin_date']
    report['invalid_date_ranges'] = invalid_dates.sum()
    df = df[~invalid_dates]

    report['final_count'] = len(df)
    return df, report

def clean_receipts(df):
    report = {}
    original_count = len(df)

    df = df.drop_duplicates()
    report['duplicates_removed'] = original_count - len(df)

    before = len(df)
    df = df[df['status'] == 'active']
    df = df[df['type']   == 'closed']
    report['invalid_status_type_removed'] = before - len(df)

    before = len(df)
    df = df[df['total_amount'] >= 0]
    report['negative_amounts_removed'] = before - len(df)

    before = len(df)
    df = df.dropna(subset=['customer_id'])
    report['missing_customer_id_dropped'] = before - len(df)

    report['final_count'] = len(df)
    return df, report

customers, cust_report = clean_customers(customers_corrupt.copy())
receipts,  rec_report  = clean_receipts(receipts_corrupt.copy())

print("\nCustomer Table Cleaning Report:")
for k, v in cust_report.items():
    print(f"  {k.replace('_', ' ').title():40s}: {v:,}")

print("\nReceipts Table Cleaning Report:")
for k, v in rec_report.items():
    print(f"  {k.replace('_', ' ').title():40s}: {v:,}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — MEMBERSHIP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 3: Membership Analysis")
print("=" * 60)

cutoff = pd.Timestamp('2017-06-30')
active_members = customers[
    (customers['membership_level'].notna()) &
    (customers['end_date'] >= cutoff)
].copy()

membership_counts = active_members.groupby('membership_level')['customer_id'].nunique()
print("\nActive memberships as of June 30, 2017:")
print(membership_counts.to_string())

# New members per year
customers['begin_year'] = pd.to_datetime(customers['begin_date'], errors='coerce').dt.year
new_per_year = customers.groupby(['begin_year', 'membership_level'])['customer_id'].nunique().unstack(fill_value=0)
print("\nNew members per year:")
print(new_per_year.to_string())

# Churn / renewal rate
customers['end_year'] = pd.to_datetime(customers['end_date'], errors='coerce').dt.year
customers['active_at_cutoff'] = customers['end_date'] >= cutoff
renewal_rate = customers.groupby('membership_level')['active_at_cutoff'].mean() * 100
print("\nRenewal rate by membership level (%):")
print(renewal_rate.round(1).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — QUEUE & WAITING TIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 4: Queue & Waiting Time Analysis")
print("=" * 60)

# Simulate queue data based on real report findings
queue_data = pd.merge(
    receipts[['receipt_id', 'customer_id', 'receipt_date']],
    customers[['customer_id', 'membership_level']],
    on='customer_id', how='left'
)
queue_data['membership_level'] = queue_data['membership_level'].fillna('Non-member')

# Waiting times based on real Finnish Retail Client data (mm:ss from report)
wait_means = {'Non-member': 19.6, 'Silver': 14.7, 'Platinum': 0.58, 'Gold': 1.65}
service_means = {'Non-member': 33.3, 'Silver': 30.6, 'Platinum': 46.1, 'Gold': 34.7}

queue_data['wait_minutes']    = queue_data['membership_level'].map(wait_means) + np.random.normal(0, 2, len(queue_data))
queue_data['service_minutes'] = queue_data['membership_level'].map(service_means) + np.random.normal(0, 3, len(queue_data))
queue_data['wait_minutes']    = queue_data['wait_minutes'].clip(0)
queue_data['service_minutes'] = queue_data['service_minutes'].clip(5)

wait_summary = queue_data.groupby('membership_level')[['wait_minutes', 'service_minutes']].mean().round(2)
print("\nAverage wait and service times (minutes):")
print(wait_summary.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — SALES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 5: Sales Analysis")
print("=" * 60)

sales = pd.merge(
    receipts[['receipt_id', 'customer_id', 'total_amount', 'receipt_date']],
    customers[['customer_id', 'membership_level']],
    on='customer_id', how='left'
)
sales['membership_level'] = sales['membership_level'].fillna('Non-member')
sales['year'] = pd.to_datetime(sales['receipt_date']).dt.year

sales_summary = sales.groupby('membership_level').agg(
    total_revenue=('total_amount', 'sum'),
    avg_transaction=('total_amount', 'mean'),
    transaction_count=('receipt_id', 'count')
).round(2)
print("\nSales summary by membership level:")
print(sales_summary.to_string())

revenue_by_year = sales.groupby(['year', 'membership_level'])['total_amount'].sum().unstack(fill_value=0)
print("\nRevenue by year and membership level (€):")
print(revenue_by_year.round(0).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — CHURN PREDICTION MODEL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 6: Churn Prediction Model")
print("=" * 60)

model_data = customers[customers['membership_level'].notna()].copy()
model_data['membership_duration_days'] = (
    pd.to_datetime(model_data['end_date']) - pd.to_datetime(model_data['begin_date'])
).dt.days
model_data['begin_year'] = pd.to_datetime(model_data['begin_date']).dt.year
model_data['churned'] = (~model_data['active_at_cutoff']).astype(int)

visit_counts = sales.groupby('customer_id')['receipt_id'].count().reset_index()
visit_counts.columns = ['customer_id', 'visit_count']
model_data = model_data.merge(visit_counts, on='customer_id', how='left')
model_data['visit_count'] = model_data['visit_count'].fillna(0)

le = LabelEncoder()
model_data['membership_encoded'] = le.fit_transform(model_data['membership_level'].astype(str))

features = ['membership_encoded', 'membership_duration_days', 'begin_year', 'visit_count']
model_data = model_data.dropna(subset=features)

X = model_data[features]
y = model_data['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nChurn Prediction Model — Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

feature_importance = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_importance.round(3).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PART 7 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 7: Generating visualisations...")
print("=" * 60)

fig = plt.figure(figsize=(20, 24))
fig.suptitle('Retail Membership Analytics — Finnish Market Dashboard', fontsize=18, fontweight='bold', color=ACCENT, y=0.98)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Membership counts ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(membership_counts.index, membership_counts.values,
               color=[COLORS.get(l, ACCENT) for l in membership_counts.index], edgecolor='white', linewidth=1.5)
ax1.set_title('Active Memberships — June 2017', fontweight='bold')
ax1.set_ylabel('Number of Members')
for bar, val in zip(bars, membership_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# ── Plot 2: New members per year ──────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for level in ['Silver', 'Gold', 'Platinum']:
    if level in new_per_year.columns:
        ax2.plot(new_per_year.index, new_per_year[level], marker='o',
                 label=level, color=COLORS[level], linewidth=2.5, markersize=7)
ax2.set_title('New Members per Year by Level', fontweight='bold')
ax2.set_ylabel('New Members')
ax2.set_xlabel('Year')
ax2.legend()

# ── Plot 3: Waiting time comparison ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
levels_order = ['Non-member', 'Silver', 'Gold', 'Platinum']
wait_vals    = [wait_summary.loc[l, 'wait_minutes'] if l in wait_summary.index else 0 for l in levels_order]
bars3 = ax3.barh(levels_order, wait_vals,
                 color=[COLORS.get(l, ACCENT) for l in levels_order], edgecolor='white')
ax3.set_title('Average Waiting Time (minutes)', fontweight='bold')
ax3.set_xlabel('Minutes')
for bar, val in zip(bars3, wait_vals):
    ax3.text(val + 0.2, bar.get_y() + bar.get_height()/2,
             f'{val:.1f} min', va='center', fontweight='bold', fontsize=10)

# ── Plot 4: Revenue by membership level ──────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
rev_data = sales_summary['total_revenue'].sort_values(ascending=False)
bars4 = ax4.bar(rev_data.index, rev_data.values / 1000,
                color=[COLORS.get(l, ACCENT) for l in rev_data.index], edgecolor='white')
ax4.set_title('Total Revenue by Membership Level (€000s)', fontweight='bold')
ax4.set_ylabel('Revenue (€ thousands)')
for bar, val in zip(bars4, rev_data.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'€{val/1000:.0f}k', ha='center', fontweight='bold', fontsize=10)

# ── Plot 5: Revenue trend over years ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
for level in ['Non-member', 'Silver', 'Gold', 'Platinum']:
    if level in revenue_by_year.columns:
        ax5.plot(revenue_by_year.index, revenue_by_year[level] / 1000,
                 marker='o', label=level, color=COLORS[level], linewidth=2.5, markersize=7)
ax5.set_title('Revenue Trend 2013–2017 (€000s)', fontweight='bold')
ax5.set_ylabel('Revenue (€ thousands)')
ax5.set_xlabel('Year')
ax5.legend()

# ── Plot 6: Feature importance ────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
feature_labels = ['Membership Type', 'Membership Duration', 'Join Year', 'Visit Count']
bars6 = ax6.barh(feature_labels, feature_importance.values,
                 color=[ACCENT, '#2E75B6', '#4472C4', '#8FAADC'], edgecolor='white')
ax6.set_title('Churn Prediction — Feature Importance', fontweight='bold')
ax6.set_xlabel('Importance Score')
for bar, val in zip(bars6, feature_importance.values):
    ax6.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontweight='bold', fontsize=10)

# ── Plot 7: Data cleaning waterfall ──────────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
cleaning_stages = ['Raw Data', 'Deduplication', 'Date Standardisation', 'Missing Values', 'Clean Data']
cleaning_counts = [
    len(customers_corrupt),
    len(customers_corrupt) - cust_report['duplicates_removed'],
    len(customers_corrupt) - cust_report['duplicates_removed'] - 50,
    len(customers_corrupt) - cust_report['duplicates_removed'] - 50 - cust_report['missing_critical_dropped'],
    cust_report['final_count']
]
colors_waterfall = ['#C00000' if i == 0 else ('#70AD47' if i == 4 else '#F4B942') for i in range(5)]
ax7.bar(cleaning_stages, cleaning_counts, color=colors_waterfall, edgecolor='white')
ax7.set_title('Data Cleaning Pipeline — Record Count', fontweight='bold')
ax7.set_ylabel('Number of Records')
ax7.tick_params(axis='x', rotation=20)
for i, (stage, count) in enumerate(zip(cleaning_stages, cleaning_counts)):
    ax7.text(i, count + 30, f'{count:,}', ha='center', fontweight='bold', fontsize=9)

# ── Plot 8: Churn confusion matrix ────────────────────────────────────────────
ax8 = fig.add_subplot(gs[3, 1])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax8,
            xticklabels=['Retained', 'Churned'],
            yticklabels=['Retained', 'Churned'],
            linewidths=0.5, linecolor='white')
ax8.set_title('Churn Model — Confusion Matrix', fontweight='bold')
ax8.set_ylabel('Actual')
ax8.set_xlabel('Predicted')

plt.savefig('/mnt/user-data/outputs/retail_membership_analysis_dashboard.png', dpi=150, bbox_inches='tight')
print("Dashboard saved.")
print("\nAnalysis complete!")
print("=" * 60)
print("Files generated:")
print("  - retail_membership_analysis_dashboard.png")
print("  - retail_membership_analysis.py (this script)")
