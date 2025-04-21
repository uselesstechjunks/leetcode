#############################################################################
Problems: Bias
#############################################################################
*****************************************************************************
Freshness and Fatigue Bias
*****************************************************************************
#. You notice that your model consistently overpredicts CTR for a set of items that are all under 6 hours old and were launched during a holiday sale. How do you determine if this is a freshness spike or user-level fatigue not being captured correctly?
#. A user has seen a product ad 20 times over 5 days. Your CTR model is under-predicting its click probability, yet the ad gets clicked again after a gap. What bias might be causing this, and how would you fix it without retraining the model?
#. You deploy a Thompson Sampling strategy to promote new ads. Over a week, you observe that many of them dominate top slots and burn out quickly—CTR drops but rank persists. What went wrong in the TS configuration or data?
#. Your calibration plot shows that for impressions 1–3, predicted CTR aligns well with actual CTR, but for impressions 8+, the model becomes overconfident. Is this more likely due to freshness or fatigue bias, and what’s the mitigation strategy?

*****************************************************************************
Feedback Loop
*****************************************************************************
