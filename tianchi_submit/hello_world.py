# coding:utf-8

# Author：雪山凌狐

import json

# 第一个为测试用
# filename = "./tcdata_test/num_list.csv"
# 这个为正式用
filename = "./tcdata/num_list.csv"

# 先完成第一个任务
result = {}
result["Q1"] = "Hello world"

def my_sum(num_list):
	"""定义一个求和函数，将列表数字求和返回"""
	total = 0
	for i in num_list:
		if i != "":
			total += int(i)
	return total

def my_sort_get(num_list, get_amount=10):
	"""定义一个排序函数，将数字从大到小排序后，获取get_amount个数字，列表返回"""
	data_list = []
	for i in num_list:
		if i != "":
			data_list.append(int(i))
	data_list.sort(reverse=True)
	return data_list[0:get_amount]

# 读取csv文件，并调用求和函数和排序函数获取赛题结果，并写入字典中
with open(filename, 'r') as f:
	num_list = f.read().split("\n")
	num_sum = my_sum(num_list)
	result["Q2"] = num_sum
	big10 = my_sort_get(num_list)
	result["Q3"] = big10

# print(result)

# 将已经ok的字典结果，打开json结果文件，写进去
with open("result.json", 'w') as f:
	f.write(json.dumps(result))