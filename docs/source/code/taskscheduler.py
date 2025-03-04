def taskScheduler(self, tasks: List[str], n: int) -> int:
	def heap_queue():
		# greedy:
		# since the order does not matter, pick the those teasks
		# in the decreasing order of counts.
		# 
		# IMPORTANT
		# process tasks with one tick at a time
		# to do that, we can maintain two data structures
		# (1) for the tasks which can be processed right away
		# (2) tasks which are in the queue to be available for processing later
		counts = Counter(tasks)
		# we don't need to reduce counts in the original dict
		# we can totally maintain the states using the two data structures we have
		# since we don't are about the type of the task, we don't need to keep
		# the task name as well
		processing = [-count for count in counts.values()] # no need to store task name
		# to hit home the point that we won't be needing counts anymore
		counts = None
		heapq.heapify(processing)
		queue = deque() # here we need to store the count and the available time after delay

		time = 0

		while processing or queue:
			time += 1

			if processing:
				count = 1 + heapq.heappop(processing)
				if count:
					queue.append((count, time + n))
			
			if queue and queue[0][1] <= time:
				count, _ = queue.popleft()
				heapq.heappush(processing, count)

		return time

	def idle_counting():
		# visualisation:
		# 
	
	return heap_queue()