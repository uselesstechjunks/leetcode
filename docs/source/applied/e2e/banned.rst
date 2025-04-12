###########################################################################
Integrity Systems
###########################################################################
***************************************************************************
Banned Item Sale
***************************************************************************
#. Problem: Prevent banned item sale on Marketplace
#. Assumptions
	#. Scale
		- 1M sellers, 50M listings
		- 1M/day new listings
		- Banned rate: 0.02-0.05% -> 200-500/day banned items
	#. Labels
		- Review budget - 2k/day (clean labels: banned + not banned)
		- Feedbacks - report listing (illegal/policy violation), report seller (fraud/scam, unresponsive) - 1000/day
	#. System behaviour
		- Creation time trigger (RT/NRT-batched) - policy violation checks (banned words, banned objects)
		- Feedback based trigger -> manual review
	#. Business metrics
		- Exposure - reduce (number of users exposed to banned items)
		- Review cost - reduce (number of listings correctly banned by ML - increase)
	#. Others
		- Banned item list - fixed, country specific
		- Banned listings fall under single policy violation
#. Problem type
	- Multi-class classification
	- Metric: per class precision, recall, f1 -> macro precision, recall, f1
#. Data
	- Listings 
		- content: text (title, description, metadata, tags), images, video
		- context: upload location, upload time
	- Seller 
		- user profile (demographics, age, gender, account age)
		- community stats - reputation (#pos feedbacks/total feedbacks), response rate, report rate
		- activity stats - upload time based (time of day, day of week)
		- conversation history with buyers - text messages (last 5 text messages)
#. Features
	- categorical -> one hot -> learned embedding -> concat
	- numerical -> normalised (account age), log-transform (stats based)
	- pretrained embeddings -> mBERT/distillBERT, ViT, ViViT
#. Learning strategy
	- 2 weeks for data collection
	- 30k clean labelled examples -> upsample rare classes if required
	- Strategies
		- Direct supervision: from 30k clean labels
			- Pros: simple
			- Cons: overfitting risk, regularise by dropout, early stopping
		- Data augmentation: upsample 3X = 100k
			- text: back translation, synonym replacement, masked token pred
			- image: rand augment (crop + resize, flip, blur, noise)
			- video: frame drop, jitter
			- Pros: more robust
			- Cons: need careful augmentation techniques
		- Semi supervision: 10M unlabelled examples + 30k clean labels with consistency/entropy regularisation
			- Mean teacher, ICT, UDA with randaugment, self-training
			- Pros: better generalisation, learns useful embeddings
			- Cons: more resource, complicated training process, requires tuning for semi supervised loss weight
			- If we go for this then a further distillation would be useful
#. Model + Training
	- Arch
		- early fusion between modalities for listings for interaction learning
	- Choice:
		- concat -> MLP [2-3x] -> classification head (simple)
		- project + concat + MLP -> classification head
		- project + cross-attention + concat + project + classification head
	- Training
		- cross entropy loss, dropout, backprop, frozen pretrained encoder
#. Eval
	- Offline - golden eval set
	- Online - live traffic A/B testing
		- random traffic alloc - might not be invokved at all
		- user alloc - users change their profile and try again
		- geo alloc - better reliable
		- baseline: basic keyword based filtering
#. Deployment
	- Distributed deployment, horizontal scaling, NRT system with batch
	- Continuous training? Learn from mistakes (items that are banned but missed by the system)
#. Monitoring
	- ML metrics, drift metrics
#. Improvements
	- Use domain pretrained encoders for different modalities (e.g., encoders for product search)
	- Use proxy labels from LLMs
	- Explore hard negative mining strategies

***************************************************************************
Banned Product Ads
***************************************************************************
#. Problem: Banned product ads sale on facebook news feed
#. Assumptions:
	- Scale 
		- 10M advertisers, 100M/day ad creatives (text/image/video)
		- 1B/day ad impression
		- Banned rate: 0.01-0.05%, 10-50k/day
	- Labels
		- Expert labels - 10k/day label budget
		- User flags - 100k/day flagged by users
		- Policy matching
	- System behaviour
		- Submission time queue/block (if high confidence)
		- real-time trigger based filter
	- Business metrics
		- Exposure to banned items
		- Rejection cost
		- Review cost
