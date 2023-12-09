#-*-encoding:utf-8-*-
import random

from data_utils import create_dico, create_mapping

def char_mapping(sentences, lower):
	"""
	음절 사전을 구축한다.
	"""
	if lower:
		chars = [[[char for char in word.lower()] for word in s[1]] for s in sentences]
	else:
		chars = [[[char for char in word] for word in s[1]] for s in sentences]
	dico = create_dico(chars)
	dico["<PAD>"] = 10000001
	dico['<UNK>'] = 10000000
	char_to_id, id_to_char = create_mapping(dico)
	print("Found %i unique chars" % (len(dico)))
	return dico, char_to_id, id_to_char


def tag_mapping(sentences):
	"""
	Create a dictionary and a mapping of tags, sorted by frequency.
	"""
	tags = [[tag for tag in s[2]] for s in sentences]
	dico = create_dico(tags)
	tag_to_id, id_to_tag = create_mapping(dico)
	print("Found %i unique tags" % len(dico))
	return dico, tag_to_id, id_to_tag


def prepare_dataset(dataset, char_to_id, tag_to_id, train=True, num_shuffle=0):
	"""
	데이터셋 전처리를 수행한다.
	return : list of list of dictionary
	dictionry
		- word indices
		- word char indices
		- tag indices
	"""
	data = []
	if train and num_shuffle != 0:
		for i in range(2):
			for sen in dataset:
				words = sen[1]
				chars = [[char_to_id[c if c in char_to_id else '<UNK>']
						for c in word] for word in sen[1]]
				tag_ids = [tag_to_id[l] for l in sen[2]]
			
				if i < 1:
					combined = list(zip(words, chars, tag_ids))
					for j in range(num_shuffle):
						first = random.randint(0, len(combined)-1)
						second = random.randint(0, len(combined)-1)
						while first == second:
							second = random.randint(0, len(combined)-1)
						temp = combined[first]
						combined[first] = combined[second]
						combined[second] = temp
					# combined = combined[-1::-1]
					shuffled_words, shuffled_chars, shuffled_tag_ids = zip(*combined)
					shuffled_element = [list(shuffled_words), list(shuffled_chars), list(shuffled_tag_ids)]
					data.append(shuffled_element)
				else:
					data.append([words, chars, tag_ids])
	else:
		for sen in dataset:
			words = sen[1]
			chars = [[char_to_id[c if c in char_to_id else '<UNK>'] for c in word] for word in sen[1]]
			tag_ids = [tag_to_id[l] for l in sen[2]]
			data.append([words, chars, tag_ids])
	return data

# def prepare_dataset(dataset, char_to_id, tag_to_id, dev=False, num_shuffle=0):
# 	"""
# 	데이터셋 전처리를 수행한다.
# 	return : list of list of dictionary
# 	dictionry
# 		- word indices
# 		- word char indices
# 		- tag indices
# 	"""
# 	data = []
# 	for sen in dataset:
# 		words = sen[1]
# 		chars = [[char_to_id[c if c in char_to_id else '<UNK>'] for c in word] for word in sen[1]]
# 		tag_ids = [tag_to_id[l] for l in sen[2]]

# 		# original data
# 		data.append([words, chars, tag_ids])

# 		if not dev and num_shuffle != 0:
# 			# shuffled data
# 			combined = list(zip(words, chars, tag_ids))
# 			for _ in range(num_shuffle):
# 				first = random.randint(0, len(combined)-1)
# 				second = random.randint(0, len(combined)-1)
# 				while first == second:
# 					second = random.randint(0, len(combined)-1)
# 				temp = combined[first]
# 				combined[first] = combined[second]
# 				combined[second] = temp
# 			shuffled_words, shuffled_chars, shuffled_tag_ids = zip(*combined)
# 			shuffled_element = [list(shuffled_words), list(shuffled_chars), list(shuffled_tag_ids)]
# 			data.append(shuffled_element)

# 	return data