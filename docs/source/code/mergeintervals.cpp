vector<vector<int>> merge(vector<vector<int>>& intervals) {
	vector<vector<int>> res;
	if (intervals.size() == 0)
		return res;

	// cmp function in std::sort should implement std::less (i.e. a < b)
	auto cmp = [](const vector<int>& i1, const vector<int>& i2)
	{
		if (i1[0] < i2[0]) return true;
		if (i1[0] == i2[0])
			return i1[1] < i2[1];
		return false;
	};
	sort(intervals.begin(), intervals.end(), cmp);
	res.push_back(intervals[0]);
	int top = 0;
	for (int i = 0; i < intervals.size(); ++i)
	{
		if (overlaps(intervals[i], res[top]))
		{
			res[top][0] = min(res[top][0], intervals[i][0]);
			res[top][1] = max(res[top][1], intervals[i][1]);
		}
		else
		{
			res.push_back(intervals[i]);
			++top;
		}
	}

	return res;
}

bool overlaps(const vector<int>& i1, const vector<int>& i2)
{
	if (i1[0] > i2[1] || i1[1] < i2[0])
		return false;
	return true;
}