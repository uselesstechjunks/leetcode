########################################################################################
NetApp
########################################################################################
****************************************************************************************
Round 1: Machine Learning / Data Science Coding
****************************************************************************************
Python-Based Questions
========================================================================================
Data Preprocessing:
	.. info::
		Problem: Given a dataset with missing values, normalize all numerical columns after imputing the missing values with their column means.

Model Evaluation:
	.. info::
		Problem: Write a function to compute precision, recall, and F1-score given true and predicted labels.
	
Feature Engineering:
	.. info::
		Problem: Given a dataset of file uploads with columns file_id, file_size, and upload_date, create a new feature representing the file size as a percentage of the average file size for its upload date.
	
Time Series Analysis:
	.. info::
		Problem: Write a function to detect outliers in a time series data based on a rolling window standard deviation.
	
Clustering:
	.. info::
		Problem: Implement k-means clustering from scratch in Python and cluster a given dataset into 3 groups.
	
Decision Trees:
	.. info::
		Problem: Implement a simple decision tree classifier to predict whether a file is likely to be accessed frequently based on features like file size, user ID, and file type.
	
API Data Fetching:
	.. info::
		Problem: Fetch data from a public API (e.g., GitHub repositories), clean it, and find the top 5 repositories with the most stars.
	
SQL-Based Questions
========================================================================================
Basic Query:
	.. info::
		Problem: Find the average file size from a table files with columns file_id, file_name, and file_size.

Join and Aggregation:
	.. info::
		Problem: Given two tables, users (with user_id, name) and files (with file_id, user_id, file_size), find the total file size uploaded by each user.

Window Functions:
	.. info::
		Problem: Write a query to calculate the rank of each user based on their total file size uploaded in descending order.Data Cleaning:

Data Cleaning:
	.. info::
		Problem: Find and delete duplicate rows in a table files based on the columns file_name and upload_date.

Complex Joins:
	.. info::
		Problem: Given three tables—users, files, and tags—find all files tagged as "important" by users who have uploaded more than 100 files.

Dynamic Queries:
	.. info::
		Problem: Create a query to find the average file size for each file_type, and return only those averages above a threshold (e.g., 100 MB).

****************************************************************************************
Round 2: Machine Learning System Design
****************************************************************************************
Design a Scalable Recommendation System for File Storage Optimization:
	.. info::
		Approach:
		Discuss data sources: user behavior logs, file metadata.
		Feature engineering: file access frequency, user preferences.
		Model: Collaborative filtering or content-based filtering.
		System architecture: Data ingestion pipeline, model training (batch), real-time inference using a microservices-based architecture.

Monitoring and Maintaining a ML Model for Anomaly Detection in Cloud Storage:
	.. info::
		Discuss:
		Metrics: Precision, recall, drift detection.
		Automation: Retraining pipelines, model versioning.
		Infrastructure: Use of Docker/Kubernetes for deployment, cloud services for scalability.

Scalable File Deduplication System:
	.. info::
		Problem: Design a system that detects duplicate files in a distributed storage system.
		Considerations: Hashing, sharding strategies, and handling partial duplicates.

Content-Based Search for Cloud Files:
	.. info::
		Problem: Design a system that allows users to search files based on their content (e.g., text or metadata) instead of just file names.
		Include indexing, embedding generation, and retrieval strategies.

Predictive Maintenance for Cloud Servers:
	.. info::
		Problem: Design a system to predict potential failures in cloud servers based on historical sensor data.
		Considerations: Handling time-series data, real-time alerts, and scalability.

Usage Pattern Anomaly Detection:
	.. info::
		Problem: Design a system that detects unusual user behavior in file access patterns to prevent unauthorized access.
		Include: Model architecture (e.g., autoencoders or isolation forests) and deployment pipeline.

Data Compression System:
	.. info::
		Problem: Propose a machine learning-based system to identify optimal compression algorithms for different file types uploaded by users.
