import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from metric import precision_at_recall

DATA_DIR = Path("data")
SUBMISSION_PATH = Path("submission.csv")
META_DATES = ["cookie_created_at", "window_start_ts", "window_end_ts"]

VALIDATION_FOLDS = [
    ("2026-04-10", "2026-04-12"),
    ("2026-04-12", "2026-04-14"),
    ("2026-04-14", "2026-04-16"),
    ("2026-04-16", "2026-04-18"),
    ("2026-04-18", "2026-04-20"),
]
SEEDS = [42, 43, 44]
N_JOBS = 4

FAST_PAUSE_S = 10
LONG_GAP_S = 1800
NIGHT_END_HOUR = 6
DEEP_SEARCH_PAGE = 3
TOP_FEATURES_TO_LOG = 15

PLATFORM_NAMES = {
    "web": "web",
    "desktop": "web",
    "android": "android",
    "ios": "ios",
    "iphone": "ios",
}
EVENT_NAMES = [
    "item_view",
    "search_results_view",
    "photo_swipe",
    "favorite_add",
    "seller_page_view",
    "contact_phone_show",
    "contact_chat_open",
    "contact_message_sent",
    "login",
]
CATEGORICAL_FEATURES = ["platform", "browser", "system"]
BASELINE_FEATURES = ["events", "unique_items"]

MODEL_PARAMS = {
    "objective": "binary",
    "n_estimators": 350,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 30,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "n_jobs": N_JOBS,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}

logger = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=META_DATES)
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=META_DATES)
    events = pd.read_csv(DATA_DIR / "events.csv.gz", parse_dates=["event_ts"])
    return train, test, events


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    logger.info("duplicated events: %d", events.duplicated().sum())
    events = events.drop_duplicates().copy()
    events["platform"] = events["platform"].str.lower().map(PLATFORM_NAMES)
    assert events["platform"].notna().all()
    return events.sort_values(["cookie_id", "event_ts"], kind="stable").reset_index(drop=True)


def keep_window_events(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    windows = meta[["cookie_id", "window_start_ts", "window_end_ts"]]
    events = events.merge(windows, on="cookie_id")
    inside = (events["event_ts"] >= events["window_start_ts"]) & (
        events["event_ts"] < events["window_end_ts"]
    )
    events = events[inside].reset_index(drop=True)
    assert set(events["event_name"]) <= set(EVENT_NAMES)
    return events


def browser_name(user_agent: str) -> str:
    if "HeadlessChrome" in user_agent:
        return "headless"
    if user_agent.startswith("Avito/"):
        return "avito_app"
    if "YaBrowser" in user_agent:
        return "yandex"
    if "Firefox" in user_agent:
        return "firefox"
    if "Chrome" in user_agent:
        return "chrome"
    if "Safari" in user_agent:
        return "safari"
    return "other"


def system_name(user_agent: str) -> str:
    if "Android" in user_agent:
        return "android"
    if "iPhone" in user_agent or "iOS" in user_agent:
        return "ios"
    if "Windows" in user_agent:
        return "windows"
    if "Macintosh" in user_agent:
        return "mac"
    if "Linux" in user_agent:
        return "linux"
    return "other"


def platform_matches_system(platform: str, system: str) -> bool:
    if platform == "web":
        return system in ("windows", "mac", "linux")
    return platform == system


def only_where_known(flag: pd.Series, source: pd.Series) -> pd.Series:
    return flag.astype(float).where(source.notna())


def add_event_columns(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    by_cookie = events.groupby("cookie_id")
    pause = by_cookie["event_ts"].diff().dt.total_seconds()
    step_x = by_cookie["pointer_x"].diff()
    step_y = by_cookie["pointer_y"].diff()
    user_agent = events["user_agent"].fillna("")

    events["pause_s"] = pause
    events["log_pause"] = np.log1p(pause)
    events["is_fast_pause"] = only_where_known(pause <= FAST_PAUSE_S, pause)
    events["is_long_gap"] = (pause > LONG_GAP_S).astype(int)
    events["hour"] = events["event_ts"].dt.hour
    events["is_night"] = (events["hour"] < NIGHT_END_HOUR).astype(int)
    events["has_pointer"] = events["pointer_x"].notna().astype(int)
    events["pointer_step"] = np.hypot(step_x, step_y)
    events["is_deep_page"] = only_where_known(
        events["search_page"] > DEEP_SEARCH_PAGE, events["search_page"]
    )
    events["is_pro_seller"] = only_where_known(
        events["seller_type"] == "pro", events["seller_type"]
    )
    events["browser"] = user_agent.map(browser_name)
    events["system"] = user_agent.map(system_name)
    events["is_headless"] = (events["browser"] == "headless").astype(int)
    events["is_mismatch"] = [
        int(not platform_matches_system(platform, system))
        for platform, system in zip(events["platform"], events["system"], strict=True)
    ]
    return events


def entropy(values: pd.Series) -> float:
    shares = values.value_counts(normalize=True)
    return float(-(shares * np.log(shares)).sum())


def aggregate_events(events: pd.DataFrame) -> pd.DataFrame:
    return events.groupby("cookie_id").agg(
        events=("event_ts", "size"),
        first_event=("event_ts", "min"),
        last_event=("event_ts", "max"),
        pause_median=("pause_s", "median"),
        pause_min=("pause_s", "min"),
        pause_count=("pause_s", "count"),
        unique_pauses=("pause_s", "nunique"),
        log_pause_mean=("log_pause", "mean"),
        log_pause_std=("log_pause", "std"),
        fast_pause_share=("is_fast_pause", "mean"),
        long_gaps=("is_long_gap", "sum"),
        hour_mean=("hour", "mean"),
        unique_hours=("hour", "nunique"),
        night_share=("is_night", "mean"),
        unique_items=("item_id", "nunique"),
        item_events=("item_id", "count"),
        unique_categories=("item_category", "nunique"),
        category_entropy=("item_category", entropy),
        unique_locations=("item_location", "nunique"),
        unique_queries=("search_query", "nunique"),
        query_events=("search_query", "count"),
        search_page_mean=("search_page", "mean"),
        search_page_max=("search_page", "max"),
        deep_page_share=("is_deep_page", "mean"),
        pro_seller_share=("is_pro_seller", "mean"),
        pointer_share=("has_pointer", "mean"),
        pointer_x_mean=("pointer_x", "mean"),
        pointer_x_max=("pointer_x", "max"),
        pointer_x_std=("pointer_x", "std"),
        pointer_y_std=("pointer_y", "std"),
        pointer_step_median=("pointer_step", "median"),
        headless=("is_headless", "max"),
        mismatch_share=("is_mismatch", "mean"),
        unique_user_agents=("user_agent", "nunique"),
        platform=("platform", "first"),
        browser=("browser", "first"),
        system=("system", "first"),
    )


def add_ratio_features(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    span_min = (features["last_event"] - features["first_event"]).dt.total_seconds() / 60
    features["span_min"] = span_min
    features["events_per_min"] = features["events"] / (span_min + 1)
    features["sessions"] = features["long_gaps"] + 1
    features["pause_unique_share"] = features["unique_pauses"] / features["pause_count"]
    features["items_per_event"] = features["unique_items"] / features["events"]
    features["item_revisits"] = features["item_events"] / (features["unique_items"] + 1)
    features["query_repeats"] = features["query_events"] / (features["unique_queries"] + 1)
    helper_columns = ["first_event", "last_event", "long_gaps", "pause_count", "unique_pauses"]
    return features.drop(columns=helper_columns)


def event_type_shares(events: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(events["cookie_id"], events["event_name"])
    counts = counts.reindex(columns=EVENT_NAMES, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    return shares.add_prefix("share_")


def build_features(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    events = add_event_columns(keep_window_events(events, meta))
    features = add_ratio_features(aggregate_events(events))
    features = features.join(event_type_shares(events))
    assert set(meta["cookie_id"]) <= set(features.index)
    return features.loc[meta["cookie_id"]]


def set_common_categories(train_x: pd.DataFrame, test_x: pd.DataFrame) -> None:
    for column in CATEGORICAL_FEATURES:
        categories = sorted(set(train_x[column]) | set(test_x[column]))
        train_x[column] = pd.Categorical(train_x[column], categories=categories)
        test_x[column] = pd.Categorical(test_x[column], categories=categories)


def predict_lightgbm(
    train_x: pd.DataFrame, train_y: np.ndarray, test_x: pd.DataFrame
) -> np.ndarray:
    scores = []
    for seed in SEEDS:
        model = lgb.LGBMClassifier(**MODEL_PARAMS, random_state=seed)
        model.fit(train_x, train_y)
        scores.append(model.predict_proba(test_x)[:, 1])
    return np.mean(scores, axis=0)


def predict_baseline(
    train_x: pd.DataFrame, train_y: np.ndarray, test_x: pd.DataFrame
) -> np.ndarray:
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, n_jobs=N_JOBS, random_state=SEEDS[0]
    )
    model.fit(train_x[BASELINE_FEATURES], train_y)
    return model.predict_proba(test_x[BASELINE_FEATURES])[:, 1]


def validate(features: pd.DataFrame, target: np.ndarray, days: pd.Series) -> pd.DataFrame:
    rows = []
    for start, end in VALIDATION_FOLDS:
        is_train = (days < start).to_numpy()
        is_valid = ((days >= start) & (days < end)).to_numpy()
        train_x, train_y = features[is_train], target[is_train]
        valid_x, valid_y = features[is_valid], target[is_valid]
        baseline = predict_baseline(train_x, train_y, valid_x)
        lightgbm = predict_lightgbm(train_x, train_y, valid_x)
        rows.append(
            {
                "fold": f"{start}..{end}",
                "cookies": len(valid_y),
                "bots": int(valid_y.sum()),
                "constant": valid_y.mean(),
                "baseline": precision_at_recall(valid_y, baseline),
                "lightgbm": precision_at_recall(valid_y, lightgbm),
            }
        )
    return pd.DataFrame(rows)


def feature_importance(features: pd.DataFrame, target: np.ndarray) -> pd.Series:
    model = lgb.LGBMClassifier(**MODEL_PARAMS, random_state=SEEDS[0])
    model.fit(features, target)
    gain = pd.Series(model.booster_.feature_importance("gain"), index=features.columns)
    return (gain / gain.sum()).sort_values(ascending=False)


def check_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    assert len(submission) == len(test)
    assert submission["cookie_id"].is_unique
    assert set(submission["cookie_id"]) == set(test["cookie_id"])
    assert submission["score"].between(0, 1).all()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train, test, events = load_data()
    events = clean_events(events)
    train_x = build_features(events, train)
    test_x = build_features(events, test)
    set_common_categories(train_x, test_x)
    target = train["target"].to_numpy()

    results = validate(train_x, target, train["window_start_ts"])
    logger.info("precision at recall 0.7:\n%s", results.round(3).to_string(index=False))
    logger.info("mean baseline %.3f", results["baseline"].mean())
    logger.info("mean lightgbm %.3f", results["lightgbm"].mean())
    importance = feature_importance(train_x, target).head(TOP_FEATURES_TO_LOG)
    logger.info("feature importance:\n%s", importance.round(3).to_string())

    scores = predict_lightgbm(train_x, target, test_x)
    submission = pd.DataFrame({"cookie_id": test["cookie_id"], "score": scores})
    check_submission(submission, test)
    submission.to_csv(SUBMISSION_PATH, index=False)
    logger.info("saved %s", SUBMISSION_PATH)


if __name__ == "__main__":
    main()
