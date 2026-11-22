from typing import Dict, Any, List
import pandas as pd

MARKETING_BLUEPRINTS = {
    "VIP High-Value Customers": {
        "behavior_summary": "Top spenders with high purchase frequency and recent transactions. Drives majority of platform revenue margin.",
        "marketing_goal": "Maximal retention, lifetime loyalty, and organic advocacy.",
        "recommended_campaign": "VIP Concierge Club: Early product launch access, zero-threshold free shipping, and dedicated relationship management.",
        "channel_mix": ["Direct VIP Email", "SMS Notifications", "Personalized Account Manager Concierge"],
        "expected_roi_impact": "High retention leverage; 5% increase in VIP retention yields ~25% gross margin boost.",
    },
    "Loyal Customers": {
        "behavior_summary": "Consistent buyers with strong recency and recurring order patterns. Steady revenue contributors.",
        "marketing_goal": "Upsell to VIP tier through cross-category basket expansion and frequency incentives.",
        "recommended_campaign": "Loyalty Tier Accelerator: Double reward points on complementary product categories and referral credits.",
        "channel_mix": ["Automated Email Drips", "In-App Banners", "Targeted Push Notifications"],
        "expected_roi_impact": "Significant incremental revenue through cross-sell and basket value expansion.",
    },
    "Regular Customers": {
        "behavior_summary": "Moderate recency and baseline spending. Represents broad customer base with growth potential.",
        "marketing_goal": "Increase purchase velocity and convert into habitual repeat buyers.",
        "recommended_campaign": "Seasonal Replenishment Reminders: Discount bundles, limited-time category promos, and first-order reviews.",
        "channel_mix": ["Email Newsletters", "Social Retargeting Ads", "Seasonal Catalogues"],
        "expected_roi_impact": "Medium-term baseline lift and customer lifetime expansion.",
    },
    "At-Risk Customers": {
        "behavior_summary": "Lapsed buyers with high days since last purchase. Previously generated revenue but now showing churn indicators.",
        "marketing_goal": "Immediate win-back and customer reactivation before permanent loss.",
        "recommended_campaign": "We Miss You Reactivation: Time-sensitive 15% discount coupon and survey asking for feedback.",
        "channel_mix": ["Dynamic Win-Back Email Series", "SMS Re-engagement Promo", "Paid Ad Retargeting"],
        "expected_roi_impact": "Recaptures 10-18% of lapsing buyers at low customer acquisition cost compared to net new buyers.",
    },
}

def generate_marketing_intelligence(profiles_df: pd.DataFrame, rfm_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Synthesize data-driven marketing playbooks and commercial strategies for each cluster persona."""
    total_rev = float(profiles_df["Total_Revenue"].sum())
    strategies: List[Dict[str, Any]] = []

    for _, row in profiles_df.iterrows():
        persona = str(row["Persona"])
        rev = float(row["Total_Revenue"])
        rev_pct = round(rev / total_rev * 100, 1) if total_rev > 0 else 0.0

        blueprint = MARKETING_BLUEPRINTS.get(persona, {
            "behavior_summary": f"Behavioral segment characterized by Average Recency of {row['Average_Recency']:.0f} days and Average Monetary spend of ${row['Average_Monetary']:,.2f}.",
            "marketing_goal": "Improve customer engagement and optimize transaction cadence.",
            "recommended_campaign": "Targeted category promotional emails and personalized recommendations.",
            "channel_mix": ["Email", "In-App Notifications"],
            "expected_roi_impact": "Consistent incremental margin contribution.",
        })

        strategies.append({
            "persona": persona,
            "customer_count": int(row["Total_Customers"]),
            "revenue_contribution": round(rev, 2),
            "revenue_pct": rev_pct,
            "avg_rfm": {
                "recency": float(row["Average_Recency"]),
                "frequency": float(row["Average_Frequency"]),
                "monetary": float(row["Average_Monetary"]),
                "median_monetary": float(row.get("Median_Monetary", row["Average_Monetary"])),
                "clv_estimate": float(row.get("CLV_Estimate", row["Average_Monetary"] * 1.2)),
            },
            "behavior_summary": blueprint["behavior_summary"],
            "marketing_goal": blueprint["marketing_goal"],
            "recommended_campaign": blueprint["recommended_campaign"],
            "channel_mix": blueprint["channel_mix"],
            "expected_roi_impact": blueprint["expected_roi_impact"],
        })

    return strategies
