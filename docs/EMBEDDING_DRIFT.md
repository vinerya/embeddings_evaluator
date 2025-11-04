# Embedding Drift: Detection and Monitoring for RAG Systems

## Table of Contents

1. [Introduction](#introduction)
2. [What is Embedding Drift?](#what-is-embedding-drift)
3. [Why Drift Matters for RAG](#why-drift-matters-for-rag)
4. [Types of Embedding Drift](#types-of-embedding-drift)
5. [Detection Methods](#detection-methods)
6. [Using Pairwise Similarity Analysis](#using-pairwise-similarity-analysis)
7. [Practical Implementation](#practical-implementation)
8. [Case Studies](#case-studies)
9. [Best Practices](#best-practices)
10. [Code Examples](#code-examples)

---

## Introduction

Embedding drift is a silent killer of RAG (Retrieval-Augmented Generation) system performance. Unlike traditional ML drift where model predictions degrade, embedding drift manifests as subtle changes in the semantic space that can dramatically impact retrieval quality without obvious warning signs.

This article explores how to detect, monitor, and respond to embedding drift using distribution analysis and pairwise similarity metrics.

---

## What is Embedding Drift?

**Embedding drift** occurs when the distribution of embeddings in your vector database changes over time, causing:

- Degraded retrieval precision
- Inconsistent search results
- Poor RAG response quality
- Increased false positives/negatives

### The Core Problem

Embeddings map text to high-dimensional vectors. When this mapping changes—whether due to model updates, data shifts, or pipeline changes—the semantic relationships between documents shift, breaking the assumptions your retrieval system was built on.

**[PLOT 1: Embedding Space Drift Visualization]**

```
Description: 2D projection of embedding space showing drift
- X-axis: First principal component
- Y-axis: Second principal component
- Plot Type: Scatter plot with two overlaid distributions
- Colors: Blue points (baseline embeddings), Red points (drifted embeddings)
- Annotations: Arrows showing direction of drift, cluster centroids marked
- What to look for: Separation between blue and red clusters, shift in density
```

---

## Why Drift Matters for RAG

### Impact on Retrieval Quality

When embeddings drift, your RAG system suffers:

1. **Precision Loss**: Relevant documents aren't retrieved
2. **Recall Degradation**: Retrieved documents aren't relevant
3. **Ranking Issues**: Top-k results contain noise
4. **Consistency Problems**: Same query returns different results over time

### Real-World Consequences

- **Customer Support**: Wrong knowledge base articles retrieved
- **Legal/Compliance**: Missing critical documents in discovery
- **E-commerce**: Poor product recommendations
- **Healthcare**: Incorrect medical information surfaced

**[PLOT 2: Retrieval Quality Over Time]**

```
Description: Line graph showing degradation of retrieval metrics
- X-axis: Time (weeks/months)
- Y-axis: Metric value (0-1 scale)
- Lines:
  * Precision@10 (blue line, declining)
  * Recall@10 (green line, declining)
  * MRR (orange line, declining)
- Annotations: Vertical line marking "Model Update" event
- What to look for: Sharp drops after changes, gradual degradation over time
```

---

## Types of Embedding Drift

### 1. Model Drift

**Cause**: Embedding model changes

- Model version updates
- Fine-tuning on new data
- Architecture changes
- Different model entirely

**Characteristics**:

- Sudden, dramatic shift
- Affects all embeddings uniformly
- Often reversible (rollback possible)

### 2. Data Drift

**Cause**: Input data distribution changes

- New document types added
- Topic distribution shifts
- Language/style changes
- Domain expansion

**Characteristics**:

- Gradual accumulation
- Affects subsets of embeddings
- Harder to reverse

### 3. Concept Drift

**Cause**: Meaning of concepts changes

- Terminology evolution
- New jargon/acronyms
- Semantic shifts in domain
- Cultural/contextual changes

**Characteristics**:

- Very gradual
- Subtle but impactful
- Requires domain expertise to detect

**[PLOT 3: Three Types of Drift Comparison]**

```
Description: Three subplots showing different drift patterns
- Layout: 3 rows × 1 column
- Each subplot:
  * X-axis: Time
  * Y-axis: Mean pairwise similarity
  * Baseline: Horizontal dashed line

Subplot 1 (Model Drift):
  * Sharp step change at model update point
  * Stable before and after

Subplot 2 (Data Drift):
  * Gradual linear increase over time
  * Smooth curve

Subplot 3 (Concept Drift):
  * Very gradual, almost imperceptible increase
  * Noisy but trending upward
```

---

## Detection Methods

### Statistical Approaches

#### 1. Distribution Comparison

Compare the distribution of pairwise similarities between baseline and current embeddings.

**Key Metrics**:

- **Mean Similarity**: Average cosine similarity between all pairs
- **Standard Deviation**: Spread of similarity distribution
- **Percentiles**: 10th, 25th, 75th, 90th percentiles
- **Skewness**: Distribution asymmetry
- **Kurtosis**: Distribution tail behavior

#### 2. Kolmogorov-Smirnov Test

Statistical test for distribution equality:

```python
from scipy.stats import ks_2samp

statistic, p_value = ks_2samp(baseline_sims, current_sims)
if p_value < 0.05:
    print("Significant drift detected!")
```

#### 3. Jensen-Shannon Divergence

Measure of similarity between probability distributions:

```python
from scipy.spatial.distance import jensenshannon

divergence = jensenshannon(baseline_hist, current_hist)
# divergence > 0.1 indicates significant drift
```

**[PLOT 4: Statistical Test Results Dashboard]**

```
Description: Multi-panel dashboard showing statistical tests
- Layout: 2×2 grid

Panel 1 (Top-Left): Distribution Overlay
  * Histogram with KDE curves
  * Blue: Baseline, Red: Current
  * Shaded overlap region
  * KS statistic and p-value annotated

Panel 2 (Top-Right): Q-Q Plot
  * Quantile-quantile comparison
  * Diagonal reference line
  * Points should follow line if distributions match

Panel 3 (Bottom-Left): Metric Changes
  * Bar chart comparing metrics
  * Grouped bars: Baseline vs Current
  * Metrics: Mean, Std, Median, P90

Panel 4 (Bottom-Right): Divergence Score
  * Gauge/meter visualization
  * Green zone (< 0.05): No drift
  * Yellow zone (0.05-0.1): Warning
  * Red zone (> 0.1): Critical drift
```

---

## Using Pairwise Similarity Analysis

### Why Pairwise Similarity?

Pairwise cosine similarity distribution captures the **fundamental structure** of your embedding space:

- **Mean**: Overall document separation
- **Std Dev**: Discriminative power
- **Shape**: Clustering behavior

### The Method

1. **Baseline Establishment**

   ```python
   baseline_sims = pairwise_similarities_auto(baseline_embeddings)
   baseline_stats = calculate_statistics(baseline_sims)
   ```

2. **Periodic Monitoring**

   ```python
   current_sims = pairwise_similarities_auto(current_embeddings)
   current_stats = calculate_statistics(current_sims)
   ```

3. **Drift Detection**
   ```python
   drift_score = calculate_drift(baseline_stats, current_stats)
   if drift_score > threshold:
       trigger_alert()
   ```

### Key Advantages

✅ **Fast**: Sampling makes it feasible for large datasets
✅ **Sensitive**: Detects subtle distribution changes
✅ **Interpretable**: Visual and statistical clarity
✅ **Actionable**: Clear metrics guide decisions

**[PLOT 5: Pairwise Similarity Distribution Evolution]**

```
Description: Animated sequence showing drift over time
- Plot Type: Overlaid histograms with KDE curves
- X-axis: Cosine similarity (0 to 1)
- Y-axis: Probability density
- Time periods shown:
  * Week 1 (Baseline): Dark blue, solid line
  * Week 2: Blue, dashed line
  * Week 3: Orange, dashed line
  * Week 4 (Current): Red, solid line
- Annotations:
  * Mean markers for each period
  * Arrows showing peak shift
  * Shaded regions showing distribution spread
- What to look for:
  * Peak moving right = documents becoming more similar (bad)
  * Distribution narrowing = loss of discriminative power (bad)
  * Tail changes = outlier behavior shifting
```

---

## Practical Implementation

### Step 1: Establish Baseline

```python
from embeddings_evaluator import load_qdrant_embeddings, pairwise_similarities_auto
import numpy as np
import json
from datetime import datetime

# Load production embeddings
embeddings = load_qdrant_embeddings("production_collection")

# Calculate pairwise similarities (fast with sampling)
similarities = pairwise_similarities_auto(embeddings)

# Calculate comprehensive statistics
baseline_stats = {
    'timestamp': datetime.now().isoformat(),
    'n_vectors': len(embeddings),
    'mean': float(np.mean(similarities)),
    'std': float(np.std(similarities)),
    'median': float(np.median(similarities)),
    'min': float(np.min(similarities)),
    'max': float(np.max(similarities)),
    'percentiles': {
        '10': float(np.percentile(similarities, 10)),
        '25': float(np.percentile(similarities, 25)),
        '75': float(np.percentile(similarities, 75)),
        '90': float(np.percentile(similarities, 90)),
        '95': float(np.percentile(similarities, 95)),
        '99': float(np.percentile(similarities, 99))
    }
}

# Save baseline
with open('baseline_stats.json', 'w') as f:
    json.dump(baseline_stats, f, indent=2)
```

### Step 2: Monitoring Script

```python
import schedule
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_baseline():
    with open('baseline_stats.json', 'r') as f:
        return json.load(f)

def calculate_drift_score(baseline, current):
    """Calculate composite drift score."""
    mean_drift = abs(current['mean'] - baseline['mean']) / baseline['mean']
    std_drift = abs(current['std'] - baseline['std']) / baseline['std']

    # Weighted combination
    drift_score = (mean_drift * 0.6) + (std_drift * 0.4)
    return drift_score

def check_drift():
    logger.info("Starting drift check...")

    # Load current embeddings
    embeddings = load_qdrant_embeddings("production_collection")
    similarities = pairwise_similarities_auto(embeddings)

    current_stats = {
        'timestamp': datetime.now().isoformat(),
        'mean': float(np.mean(similarities)),
        'std': float(np.std(similarities)),
        'median': float(np.median(similarities))
    }

    # Load baseline
    baseline = load_baseline()

    # Calculate drift
    drift_score = calculate_drift_score(baseline, current_stats)

    logger.info(f"Drift score: {drift_score:.4f}")

    # Alert thresholds
    if drift_score > 0.10:  # 10% drift
        send_critical_alert(drift_score, baseline, current_stats)
    elif drift_score > 0.05:  # 5% drift
        send_warning_alert(drift_score, baseline, current_stats)

    # Log metrics
    log_to_monitoring_system(current_stats, drift_score)

    # Save snapshot
    save_snapshot(current_stats)

# Schedule checks
schedule.every().day.at("02:00").do(check_drift)  # Daily at 2 AM
schedule.every().monday.at("09:00").do(generate_weekly_report)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Step 3: Alert System

```python
def send_critical_alert(drift_score, baseline, current):
    """Send critical alert for significant drift."""
    message = f"""
    🚨 CRITICAL: Embedding Drift Detected

    Drift Score: {drift_score:.2%}

    Baseline (established {baseline['timestamp']}):
    - Mean similarity: {baseline['mean']:.4f}
    - Std deviation: {baseline['std']:.4f}

    Current:
    - Mean similarity: {current['mean']:.4f} ({get_change_pct(baseline['mean'], current['mean'])})
    - Std deviation: {current['std']:.4f} ({get_change_pct(baseline['std'], current['std'])})

    Impact: Retrieval quality likely degraded
    Action Required: Investigate immediately
    """

    # Send to Slack/PagerDuty/Email
    send_to_slack(message, severity='critical')
    create_pagerduty_incident(message)
    send_email(message, recipients=['ml-team@company.com'])

def send_warning_alert(drift_score, baseline, current):
    """Send warning for moderate drift."""
    message = f"""
    ⚠️ WARNING: Moderate Embedding Drift

    Drift Score: {drift_score:.2%}

    Changes detected:
    - Mean: {baseline['mean']:.4f} → {current['mean']:.4f}
    - Std: {baseline['std']:.4f} → {current['std']:.4f}

    Action: Monitor closely, investigate if persists
    """

    send_to_slack(message, severity='warning')
```

**[PLOT 6: Monitoring Dashboard]**

```
Description: Real-time monitoring dashboard
- Layout: 4-panel dashboard

Panel 1 (Top-Left): Drift Score Timeline
  * Line graph with threshold bands
  * X-axis: Time (last 30 days)
  * Y-axis: Drift score (0-0.2)
  * Green band (0-0.05): Normal
  * Yellow band (0.05-0.10): Warning
  * Red band (>0.10): Critical
  * Points colored by severity

Panel 2 (Top-Right): Current vs Baseline Distribution
  * Overlaid histograms
  * Blue: Baseline, Red: Current
  * Vertical lines at means
  * Shaded difference regions

Panel 3 (Bottom-Left): Metric Trends
  * Multi-line graph
  * Lines for: Mean, Std, Median
  * X-axis: Time
  * Y-axis: Metric value
  * Baseline reference lines (dashed)

Panel 4 (Bottom-Right): Alert History
  * Timeline of alerts
  * Icons: 🚨 (critical), ⚠️ (warning), ✅ (normal)
  * Hover shows details
  * Color-coded by severity
```

---

## Case Studies

### Case Study 1: Model Update Gone Wrong

**Scenario**: Team updates embedding model from v1 to v2

**Before Update**:

- Mean similarity: 0.28
- Std deviation: 0.14
- Peak location: 0.22
- RAG precision@10: 0.87

**After Update**:

- Mean similarity: 0.45 (+61%)
- Std deviation: 0.09 (-36%)
- Peak location: 0.41 (+86%)
- RAG precision@10: 0.62 (-29%)

**Analysis**:

- Documents became much more similar (mean increased)
- Lost discriminative power (std decreased)
- Distribution peak shifted right (clustering)
- Retrieval quality degraded significantly

**Action Taken**: Rolled back to v1, investigated v2 training data

**[PLOT 7: Model Update Impact]**

```
Description: Before/after comparison of model update
- Layout: 2×2 grid

Top Row: Distribution Comparison
  * Left: v1 distribution (blue)
  * Right: v2 distribution (red)
  * Same scale for comparison
  * Annotations showing mean, std, peak

Bottom Row: Retrieval Metrics
  * Left: Bar chart of metrics (Precision, Recall, MRR)
  * Right: Confusion matrix style heatmap
    - Rows: v1 top-10 results
    - Columns: v2 top-10 results
    - Shows overlap/divergence
```

### Case Study 2: Gradual Data Drift

**Scenario**: E-commerce site adds new product categories over 6 months

**Timeline**:

- Month 0 (Baseline): Mean 0.31, Std 0.13
- Month 2: Mean 0.33, Std 0.12
- Month 4: Mean 0.36, Std 0.11
- Month 6: Mean 0.39, Std 0.10

**Analysis**:

- Gradual increase in mean similarity
- Steady decrease in std deviation
- New categories not well-separated from existing ones
- Search quality degrading slowly

**Action Taken**:

- Re-embedded entire catalog with updated model
- Implemented category-specific fine-tuning
- Established monthly drift monitoring

**[PLOT 8: Gradual Drift Over Time]**

```
Description: 6-month drift progression
- Plot Type: Dual-axis line graph
- X-axis: Time (months 0-6)
- Left Y-axis: Mean similarity (0.30-0.40)
- Right Y-axis: Std deviation (0.09-0.14)
- Lines:
  * Blue line (left axis): Mean similarity, trending up
  * Orange line (right axis): Std deviation, trending down
- Annotations:
  * Vertical markers for "New Category Added" events
  * Shaded regions showing normal/warning/critical zones
  * Trend lines showing linear regression
- What to look for:
  * Correlation between category additions and drift
  * Accelerating vs linear drift
  * When thresholds were crossed
```

### Case Study 3: Successful Drift Prevention

**Scenario**: Healthcare knowledge base with quarterly content updates

**Strategy**:

1. **Pre-update Testing**

   - Embed sample of new content
   - Compare distribution with existing
   - Predict drift impact

2. **Staged Rollout**

   - Add 10% of new content
   - Monitor for 1 week
   - Gradually increase if stable

3. **Continuous Monitoring**
   - Daily drift checks
   - Weekly distribution analysis
   - Monthly baseline updates

**Results**:

- Drift score maintained < 0.03 over 12 months
- No retrieval quality degradation
- Proactive adjustments prevented issues

**[PLOT 9: Drift Prevention Success]**

```
Description: Controlled drift management
- Plot Type: Control chart
- X-axis: Time (12 months)
- Y-axis: Drift score (0-0.10)
- Elements:
  * Line graph of drift score
  * Upper control limit (UCL) at 0.05
  * Lower control limit (LCL) at 0
  * Center line at 0.025
  * Points colored: Green (in control), Yellow (near limit)
  * Vertical markers for "Content Update" events
- Annotations:
  * "Pre-update testing" callouts
  * "Staged rollout" periods (shaded)
  * "Baseline refresh" markers
- What to look for:
  * Drift staying within control limits
  * Quick return to baseline after updates
  * No sustained upward trends
```

---

## Best Practices

### 1. Establish Robust Baselines

**Do**:

- ✅ Use production data, not test data
- ✅ Ensure baseline represents steady state
- ✅ Document baseline conditions
- ✅ Include multiple baseline snapshots
- ✅ Update baseline periodically (quarterly)

**Don't**:

- ❌ Use data during system changes
- ❌ Baseline during known issues
- ❌ Use too small sample sizes
- ❌ Forget to version baselines

### 2. Set Appropriate Thresholds

**Threshold Guidelines**:

| Metric                 | Warning | Critical | Notes                    |
| ---------------------- | ------- | -------- | ------------------------ |
| Mean Similarity Change | 5%      | 10%      | Relative to baseline     |
| Std Deviation Change   | 10%     | 20%      | Relative to baseline     |
| Drift Score            | 0.05    | 0.10     | Composite metric         |
| KS Test p-value        | 0.05    | 0.01     | Statistical significance |

**Adjust based on**:

- System criticality
- Update frequency
- Historical patterns
- Business impact

### 3. Monitoring Cadence

**Recommended Schedule**:

| System Type                  | Check Frequency | Baseline Update |
| ---------------------------- | --------------- | --------------- |
| Critical (Healthcare, Legal) | Daily           | Monthly         |
| High-traffic (E-commerce)    | Daily           | Quarterly       |
| Standard (Internal tools)    | Weekly          | Quarterly       |
| Low-risk (Experimental)      | Monthly         | Semi-annually   |

### 4. Response Procedures

**When Drift Detected**:

1. **Immediate** (< 1 hour):

   - Verify alert is not false positive
   - Check for recent system changes
   - Assess current impact on users

2. **Short-term** (< 24 hours):

   - Investigate root cause
   - Quantify impact on retrieval quality
   - Decide: rollback, fix, or monitor

3. **Long-term** (< 1 week):
   - Implement fix if needed
   - Update monitoring thresholds
   - Document incident and learnings
   - Update runbooks

### 5. Prevention Strategies

**Proactive Measures**:

1. **Pre-deployment Testing**

   ```python
   def test_embedding_update(new_embeddings, baseline_stats):
       """Test new embeddings before deployment."""
       new_sims = pairwise_similarities_auto(new_embeddings)
       new_stats = calculate_statistics(new_sims)

       predicted_drift = calculate_drift_score(baseline_stats, new_stats)

       if predicted_drift > 0.05:
           return False, f"Predicted drift: {predicted_drift:.2%}"
       return True, "Safe to deploy"
   ```

2. **Staged Rollouts**

   - Deploy to 10% of traffic
   - Monitor for 24-48 hours
   - Gradually increase if stable

3. **A/B Testing**

   - Run old and new embeddings in parallel
   - Compare retrieval quality metrics
   - Switch only if new is better

4. **Regular Audits**
   - Monthly distribution analysis
   - Quarterly baseline refresh
   - Annual system review

**[PLOT 10: Prevention Workflow]**

```
Description: Flowchart of drift prevention process
- Flowchart with decision nodes
- Start: "Embedding Update Planned"
- Steps:
  1. Pre-deployment testing
     - Diamond: "Drift < threshold?"
     - Yes → Continue
     - No → "Investigate & Fix"
  2. Staged rollout (10%)
     - Diamond: "Stable for 48h?"
     - Yes → Continue
     - No → "Rollback & Investigate"
  3. Gradual increase (25%, 50%, 100%)
     - Monitor at each stage
  4. Post-deployment monitoring
     - Daily checks for 1 week
  5. Baseline update
- Color coding:
  * Green: Safe to proceed
  * Yellow: Caution/monitoring
  * Red: Stop/rollback
```

---

## Code Examples

### Complete Drift Detection System

```python
"""
drift_detector.py - Complete embedding drift detection system
"""

import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass, asdict
from embeddings_evaluator import load_qdrant_embeddings, pairwise_similarities_auto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingStats:
    """Statistics for embedding distribution."""
    timestamp: str
    n_vectors: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    percentiles: Dict[str, float]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class DriftDetector:
    """Embedding drift detection and monitoring."""

    def __init__(
        self,
        collection_name: str,
        baseline_path: str = "baseline_stats.json",
        history_path: str = "drift_history.json",
        warning_threshold: float = 0.05,
        critical_threshold: float = 0.10
    ):
        self.collection_name = collection_name
        self.baseline_path = Path(baseline_path)
        self.history_path = Path(history_path)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def calculate_stats(self, embeddings: np.ndarray) -> EmbeddingStats:
        """Calculate comprehensive statistics for embeddings."""
        logger.info(f"Calculating statistics for {len(embeddings)} vectors...")

        similarities = pairwise_similarities_auto(embeddings, verbose=True)

        stats = EmbeddingStats(
            timestamp=datetime.now().isoformat(),
            n_vectors=len(embeddings),
            mean=float(np.mean(similarities)),
            std=float(np.std(similarities)),
            median=float(np.median(similarities)),
            min=float(np.min(similarities)),
            max=float(np.max(similarities)),
            percentiles={
                '10': float(np.percentile(similarities, 10)),
                '25': float(np.percentile(similarities, 25)),
                '75': float(np.percentile(similarities, 75)),
                '90': float(np.percentile(similarities, 90)),
                '95': float(np.percentile(similarities, 95)),
                '99': float(np.percentile(similarities, 99))
            }
        )

        logger.info(f"Stats calculated: mean={stats.mean:.4f}, std={stats.std:.4f}")
        return stats

    def establish_baseline(self, embeddings: np.ndarray = None):
        """Establish baseline statistics."""
        if embeddings is None:
            logger.info(f"Loading embeddings from {self.collection_name}...")
            embeddings = load_qdrant_embeddings(self.collection_name)

        stats = self.calculate_stats(embeddings)

        # Save baseline
        with open(self.baseline_path, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)

        logger.info(f"Baseline established and saved to {self.baseline_path}")
        return stats

    def load_baseline(self) -> EmbeddingStats:
        """Load baseline statistics."""
        if not self.baseline_path.exists():
            raise FileNotFoundError(
                f"Baseline not found at {self.baseline_path}. "
                "Run establish_baseline() first."
            )

        with open(self.baseline_path, 'r') as f:
            data = json.load(f)

        return EmbeddingStats.from_dict(data)

    def calculate_drift_score(
        self,
        baseline: EmbeddingStats,
        current: EmbeddingStats
    ) -> float:
        """Calculate composite drift score."""
        # Relative changes
        mean_change = abs(current.mean - baseline.mean) / baseline.mean
        std_change = abs(current.std - baseline.std) / baseline.std
        median_change = abs(current.median - baseline.median) / baseline.median

        # Weighted combination
        drift_score = (
            mean_change * 0.5 +
            std_change * 0.3 +
            median_change * 0.2
        )

        return drift_score

    def detect_drift(self, embeddings: np.ndarray = None) -> Dict:
        """Detect drift in current embeddings."""
        # Load current embeddings if not provided
        if embeddings is None:
            logger.info(f"Loading embeddings from {self.collection_name}...")
            embeddings = load_qdrant_embeddings(self.collection_name)

        # Calculate current stats
        current_stats = self.calculate_stats(embeddings)

        # Load baseline
        baseline_stats = self.load_baseline()

        # Calculate drift
        drift_score = self.calculate_drift_score(baseline_stats, current_stats)

        # Determine severity
        if drift_score >= self.critical_threshold:
            severity = "CRITICAL"
        elif drift_score >= self.warning_threshold:
            severity = "WARNING"
        else:
            severity = "NORMAL"

        # Compile results
        result = {
            'timestamp': current_stats.timestamp,
            'drift_score': drift_score,
            'severity': severity,
            'baseline': baseline_stats.to_dict(),
            'current': current_stats.to_dict(),
            'changes': {
                'mean': {
                    'absolute': current_stats.mean - baseline_stats.mean,
                    'relative': (current_stats.mean - baseline_stats.mean) / baseline_stats.mean
                },
                'std': {
                    'absolute': current_stats.std - baseline_stats.std,
                    'relative': (current_stats.std - baseline_stats.std) / baseline_stats.std
                },
                'median': {
                    'absolute': current_stats.median - baseline_stats.median,
                    'relative': (current_stats.median - baseline_stats.median) / baseline_stats.median
                }
            }
        }

        # Save to history
        self._save_to_history(result)

        # Log results
        logger.info(f"Drift detection complete: {severity} (score: {drift_score:.4f})")

        return result

    def _save_to_history(self, result: Dict):
        """Save drift detection result to history."""
        # Load existing history
        if self.history_path.exists():
            with open(self.history_path, 'r') as f:
                history = json.load(f)
        else:
            history = []

        # Append new result
        history.append(result)

        # Keep last 90 days
        cutoff = (datetime.now() - timedelta(days=90)).isoformat()
        history = [h for h in history if h['timestamp'] > cutoff]

        # Save
        with open(self.history_path, 'w') as f:
            json.dump(history, f, indent=2)

    def get_history(self, days: int = 30) -> List[Dict]:
        """Get drift history for last N days."""
        if not self.history_path.exists():
            return []

        with open(self.history_path, 'r') as f:
            history = json.load(f)

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        return [h for h in history if h['timestamp'] > cutoff]

    def generate_report(self) -> str:
        """Generate drift detection report."""
        history = self.get_history(days=30)

        if not history:
            return "No drift history available."

        latest = history[-1]

        report = f"""
# Embedding Drift Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Collection: {self.collection_name}

## Current Status
- **Severity**: {latest['severity']}
- **Drift Score**: {latest['drift_score']:.4f}
- **Last Check**: {latest['timestamp']}

## Baseline vs Current
| Metric | Baseline | Current | Change |
|--------|----------|---------|--------|
| Mean | {latest['baseline']['mean']:.4f} | {latest['current']['mean']:.4f} | {latest['changes']['mean']['relative']:.2%} |
| Std Dev | {latest['baseline']['std']:.4f} | {latest['current']['std']:.4f} | {latest['changes']['std']['relative']:.2%} |
| Median | {latest['baseline']['median']:.4f} | {latest['current']['median']:.4f} | {latest['changes']['median']['relative']:.2%} |

## 30-Day History
Total checks: {len(history)}
- Normal: {sum(1 for h in history if h['severity'] == 'NORMAL')}
- Warning: {sum(1 for h in history if h['severity'] == 'WARNING')}
- Critical: {sum(1 for h in history if h['severity'] == 'CRITICAL')}

## Recommendations
"""

        if latest['severity'] == 'CRITICAL':
            report += """
⚠️ **IMMEDIATE ACTION REQUIRED**
1. Investigate recent changes (model updates, data changes)
2. Compare retrieval quality metrics
3. Consider rollback if quality degraded
4. Review drift history for patterns
"""
        elif latest['severity'] == 'WARNING':
            report += """
⚠️ **MONITOR CLOSELY**
1. Increase monitoring frequency
2. Review recent changes
3. Prepare rollback plan
4. Track if drift continues
"""
        else:
            report += """
✅ **SYSTEM HEALTHY**
1. Continue regular monitoring
2. Maintain current baseline
3. Document any planned changes
"""

        return report


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = DriftDetector(
        collection_name="production_embeddings",
        warning_threshold=0.05,
        critical_threshold=0.10
    )

    # Establish baseline (first time only)
    # detector.establish_baseline()

    # Detect drift
    result = detector.detect_drift()

    # Print results
    print(f"\nDrift Detection Results:")
    print(f"Severity: {result['severity']}")
    print(f"Drift Score: {result['drift_score']:.4f}")
    print(f"\nChanges:")
    print(f"  Mean: {result['changes']['mean']['relative']:.2%}")
    print(f"  Std: {result['changes']['std']['relative']:.2%}")

    # Generate report
    report = detector.generate_report()
    print(report)

    # Save report
    with open('drift_report.md', 'w') as f:
        f.write(report)
```

---

## Summary

This comprehensive guide covers:

1. **Understanding Drift**: What it is, why it matters, and types of drift
2. **Detection Methods**: Statistical approaches and pairwise similarity analysis
3. **Implementation**: Complete code examples for monitoring and alerting
4. **Case Studies**: Real-world examples of drift scenarios
5. **Best Practices**: Guidelines for baselines, thresholds, and response procedures

### Key Takeaways

✅ **Embedding drift is detectable** using distribution analysis
✅ **Pairwise similarity** provides fast, accurate drift detection
✅ **Proactive monitoring** prevents retrieval quality degradation
✅ **Automated systems** enable continuous drift detection
✅ **Clear thresholds** guide response decisions

### Next Steps

1. Establish baseline for your production embeddings
2. Set up automated monitoring
3. Configure alerting thresholds
4. Document response procedures
5. Review and update baselines quarterly

For more information and tools, visit the [embeddings_evaluator](https://github.com/vinerya/embeddings_evaluator) repository.

---

_Last updated: November 2025_
