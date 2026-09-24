import numpy as np
import pandas as pd
from schemas import UserData


def generate_synthetic_data(
    n_users: int = 20000, random_seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(random_seed)

    # Demographics
    user_ids = [i for i in range(1, n_users + 1)]

    age_groups = np.random.choice(
        ["18-24", "25-34", "35-44", "45-54", "55+"],
        size=n_users,
        p=[0.15, 0.35, 0.30, 0.15, 0.05],
    )

    countries = np.random.choice(
        ["Greece", "Ireland", "Cyprus", "Finland", "Brazil"],
        size=n_users,
        p=[0.20, 0.10, 0.10, 0.05, 0.55],
    )

    # Betting Behavior
    days_since_registration = np.random.exponential(scale=120, size=n_users).astype(int)
    days_since_registration = np.clip(days_since_registration, 1, 365)

    user_activity_segment = np.random.choice(
        ["low", "medium", "high"], size=n_users, p=[0.4, 0.4, 0.2]
    )

    num_bets_last_30d = np.zeros(n_users)
    num_bets_last_30d[user_activity_segment == "low"] = np.random.poisson(
        5, size=np.sum(user_activity_segment == "low")
    )
    num_bets_last_30d[user_activity_segment == "medium"] = np.random.poisson(
        20, size=np.sum(user_activity_segment == "medium")
    )
    num_bets_last_30d[user_activity_segment == "high"] = np.random.poisson(
        50, size=np.sum(user_activity_segment == "high")
    )

    num_bets_last_7d = np.random.binomial(num_bets_last_30d.astype(int), 0.3)
    num_bets_last_7d = np.maximum(
        1, num_bets_last_7d
    )  # At least 1 bet in last 7 days (active users)

    # Favorite sport (optional categorical feature)
    favorite_sports = np.random.choice(
        ["Football", "Basketball", "Tennis", "Horse_Racing", "Other"],
        size=n_users,
        p=[0.5, 0.15, 0.15, 0.1, 0.1],
    )

    # Engagement Metrics
    # For active users only: days_since_last_bet should be within last 7 days
    days_since_last_bet = np.zeros(n_users)
    days_since_last_bet[user_activity_segment == "low"] = np.random.exponential(
        3, size=np.sum(user_activity_segment == "low")
    )
    days_since_last_bet[user_activity_segment == "medium"] = np.random.exponential(
        2, size=np.sum(user_activity_segment == "medium")
    )
    days_since_last_bet[user_activity_segment == "high"] = np.random.exponential(
        1, size=np.sum(user_activity_segment == "high")
    )
    days_since_last_bet = np.clip(days_since_last_bet, 0, 6).astype(int)

    # Financial Behavior
    num_deposits_30d = np.maximum(1, np.random.poisson(3, size=n_users))

    num_withdrawals_30d = np.random.poisson(1, size=n_users)

    # Churn Label - Predict if user will stop betting in NEXT 7 days
    # Base churn probability
    churn_probability = np.zeros(n_users) + 0.15

    # Risk factors for FUTURE churn:
    churn_probability += (
        days_since_last_bet >= 4
    ) * 0.25  # Haven't bet in 4-6 days (getting inactive)
    churn_probability += (num_bets_last_7d <= 2) * 0.2  # Very low recent activity
    churn_probability += (num_bets_last_30d < 10) * 0.15  # Low overall activity
    churn_probability += (
        days_since_registration < 30
    ) * 0.2  # New users have higher churn
    churn_probability += (
        num_deposits_30d == 1
    ) * 0.1  # Only one deposit (not committed)
    churn_probability += (
        num_withdrawals_30d > num_deposits_30d
    ) * 0.15  # Withdrawing more than depositing

    # Protective factors (less likely to churn):
    churn_probability -= (num_bets_last_7d > 15) * 0.25  # High recent activity
    churn_probability -= (
        num_deposits_30d > 3
    ) * 0.15  # Multiple deposits show commitment
    churn_probability -= (
        days_since_registration > 180
    ) * 0.15  # Established users more loyal
    churn_probability -= (days_since_last_bet == 0) * 0.1  # Bet today (very engaged)

    churn_probability = np.clip(churn_probability, 0.05, 0.85)
    churned = (np.random.random(n_users) < churn_probability).astype(int)

    features = pd.DataFrame(
        {
            "user_id": user_ids,
            "days_since_registration": days_since_registration,
            "age_group": age_groups,
            "country": countries,
            "num_bets_last_30d": num_bets_last_30d.astype(int),
            "num_bets_last_7d": num_bets_last_7d.astype(int),
            "days_since_last_bet": days_since_last_bet.astype(int),
            "num_deposits_30d": num_deposits_30d,
            "num_withdrawals_30d": num_withdrawals_30d,
            "favorite_sport": favorite_sports,
        }
    )

    target = pd.Series(churned, name="churned")

    # Ensure data types match pydantic schema
    UserData.model_validate(features.iloc[0].to_dict(), strict=True, extra="forbid")

    return features, target


if __name__ == "__main__":
    features, target = generate_synthetic_data()
    features.to_csv("synthetic_user_data.csv", index=False)
    target.to_csv("synthetic_user_target.csv", index=False)
