#-*-coding:utf-8-*-
import math, random

def create_dico(item_list):
	"""
	Create a dictionary of items from a list of list of items.
	"""
	assert type(item_list) is list
	dico = {}
	for items in item_list:
		for item in items:
			if type(item) == list:
				for i in item:
					if i not in dico: dico[i] = 1
					else: dico[i] += 1
			else:
				if item not in dico: dico[item] = 1
				else: dico[item] += 1
	return dico


def create_mapping(dico):
	"""
	Create a mapping (item to ID / ID to item) from a dictionary.
	Items are ordered by decreasing frequency.
	"""
	sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
	id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
	item_to_id = {v: k for k, v in id_to_item.items()}
	return item_to_id, id_to_item

class BatchManager(object):

	def __init__(self, data, batch_size, max_char_length):
		self.batch_data = self.sort_and_pad(data, batch_size, max_char_length)
		self.len_data = len(self.batch_data)

	def sort_and_pad(self, data, batch_size, max_char_length):
		num_batch = int(math.ceil(len(data) /batch_size))
		sorted_data = sorted(data, key=lambda x: len(x[0]))
		batch_data = list()
		for i in range(num_batch):
			batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size], max_char_length))
		return batch_data

	@staticmethod
	def pad_data(data, max_char_length):
		strings = []
		chars = []
		targets = []
		max_eojeol_length = max([len(sentence[0]) for sentence in data])
		for line in data:
			string, char, target = line
			eoj_padding = [0] * (max_eojeol_length - len(string))
			strings.append(string + eoj_padding)

			if len(eoj_padding) != 0:
				char.extend([[0]] * (max_eojeol_length - len(string)))

			new_chars = []
			for el in char:
				if len(el) <= max_char_length:
					new_chars.append( el + [0]*(max_char_length - len(el)) )
				else:
					new_chars.append( el[:max_char_length] )
							
			chars.append(new_chars)
			targets.append(target + eoj_padding)
			assert len(chars[-1]) == len(targets[-1])
			assert len(chars[-1]) == len(strings[-1])
		return [strings, chars, targets]

	def iter_batch(self, shuffle=False):
		if shuffle:
			random.shuffle(self.batch_data)
		for idx in range(self.len_data):
			yield self.batch_data[idx]
