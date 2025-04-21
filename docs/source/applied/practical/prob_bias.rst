#############################################################################
Problems: Bias
#############################################################################
*****************************************************************************
Features
*****************************************************************************
- Fatigue: time_since_last_impressed, impression_count
- Freshness: item_age, item_impression_count, item_click_count
- Starvation:

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
#. Your model predicts high CTR for a set of job ads. They are shown more, clicked more, and then retrained on that data. Over the next few weeks, model diversity drops, and new job ads struggle to gain traction. What is the feedback loop at play here, and which parts of your pipeline are reinforcing it?
#. In your marketplace recommender, tail sellers consistently receive low predicted CVR and thus low exposure. There is no hard cap, but they rarely show up. What metrics would you track to detect a feedback loop affecting tail sellers, and what signals would confirm it?
#. Your team adds a new "engagement score" feature derived from click count and dwell time. Within one week, CTR increases short-term, but system engagement quality drops. How can this new feature contribute to a feedback loop, and how would you check whether it’s hurting long-term utility?
#. You’re training a ranking model on click labels. Over time, CTR@10 looks strong, but CTR@50 keeps deteriorating. You suspect a feedback loop is starving the tail. How do you modify your training or sampling pipeline to mitigate this exposure bias without adding exploration traffic?
#. Your system boosts CTR by ranking long-form videos higher, but user dwell time per session is going down. Stakeholders ask for a fix that preserves CTR while improving watch quality. What changes would you make to break the watch-ratio feedback loop?
#. Your CTR model is trained on observed clicks but deployed for value-based bidding (CTR × CVR × value). Over time, the system over-optimizes for CTR but fails to deliver high-value conversions. What do you change in model training, serving, or calibration to align with end value?
