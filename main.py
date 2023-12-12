# encoding=utf8
import os
import pickle
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from model import Model
from loader import char_mapping, tag_mapping, prepare_dataset
from model_utils import get_logger, make_path, create_model, save_model, print_config, save_config, test_srl
from data_utils import BatchManager
from data_loader import data_loader

def parse_args():
	flags = tf.app.flags
	# configurations for the model
	flags.DEFINE_integer("seg_dim",			0,		"Embedding size for segmentation, 0 if not used")
	flags.DEFINE_integer("char_dim",		100,	"Embedding size for characters")
	flags.DEFINE_integer("char_lstm_dim",	100,	"Num of hidden units in char LSTM")
	flags.DEFINE_integer("word_lstm_dim",	100,	"Num of hidden units in word LSTM")
	flags.DEFINE_integer("max_char_length",	8,		"max number of character in word")
	flags.DEFINE_integer("max_word_length",	95,		"number of word")
	flags.DEFINE_integer("num_tags",		29,		"number of tags")
	flags.DEFINE_integer("num_chars",		8000,	"number of chars")

	# configurations for training
	flags.DEFINE_float("clip",			5,			"Gradient clip")
	flags.DEFINE_float("dropout",		0.5,		"Dropout rate")
	flags.DEFINE_float("lr",			0.001,		"Initial learning rate")
	flags.DEFINE_string("optimizer",	"adam",		"Optimizer for training")
	flags.DEFINE_boolean("lower",		True,		"Wither lower case")
	flags.DEFINE_integer("batch_size",	20,			"batch size")
	flags.DEFINE_integer("patience",	5,			"Patience for the validation-based early stopping")

	flags.DEFINE_integer("max_epoch",	10,		"maximum training epochs")
	flags.DEFINE_integer("steps_check", 100,		"steps per checkpoint")
	flags.DEFINE_string("ckpt_path",	"ckpt",		 "Path to save model")
	flags.DEFINE_string("summary_path", "summary",		"Path to store summaries")
	flags.DEFINE_string("log_file",		"train.log",	"File for log")
	flags.DEFINE_string("map_file",		"maps.pkl",		"file for maps")
	flags.DEFINE_string("vocab_file",	"vocab.json",	"File for vocab")
	flags.DEFINE_string("config_file",	"config_file",	"File for config")
	flags.DEFINE_string("script",		"conlleval",	"evaluation script")
	flags.DEFINE_string("result_path",	"result",		"Path for results")
	flags.DEFINE_integer("num_shuffle",	0,				"number of shuffling")

	# dataset
	flags.DEFINE_string("DATASET_PATH",	'./data/', "path for dataset")

	FLAGS = tf.app.flags.FLAGS
	assert FLAGS.clip < 5.1, "gradient clip should't be too much"
	assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
	assert FLAGS.lr > 0, "learning rate must larger than zero"
	assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]

	return FLAGS

# config for the model
def config_model():
	config = OrderedDict()
	config["num_chars"] = FLAGS.num_chars
	config["char_dim"] = FLAGS.char_dim
	config["num_tags"] = FLAGS.num_tags
	config["seg_dim"] = FLAGS.seg_dim
	config["char_lstm_dim"] = FLAGS.char_lstm_dim
	config["word_lstm_dim"] = FLAGS.word_lstm_dim
	config["batch_size"] = FLAGS.batch_size

	config["clip"] = FLAGS.clip
	config["dropout_keep"] = 1.0 - FLAGS.dropout
	config["optimizer"] = FLAGS.optimizer
	config["lr"] = FLAGS.lr
	config["lower"] = FLAGS.lower
	config["max_char_length"] = FLAGS.max_char_length
	config["max_word_length"] = FLAGS.max_word_length

	return config

"""
매 epoch에 validation set의 평가를 수행하고 그 결과를 출력한다.
"""
def evaluate(sess, model, data, id_to_tag, logger):
	logger.info("evaluate")
	
	srl_results = model.evaluate_model(sess, data, id_to_tag)
	eval_lines = test_srl(srl_results, FLAGS.result_path)
	for line in eval_lines:
		logger.info(line.strip())
	f1 = float(eval_lines[1].strip().split()[-1])

	best_test_f1 = model.best_dev_f1.eval(session=sess)
	if f1 > best_test_f1:
		tf.assign(model.best_dev_f1, f1).eval(session=sess)
		logger.info("new best dev f1 score:{:>.3f}".format(f1))
	return f1 > best_test_f1, f1

def train(sess, _train_manager):
	with open(FLAGS.map_file, "wb") as f:
		pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)

	steps_per_epoch = _train_manager.len_data
	early_stop = 0
	sess_best = 0

	logger.info("start training")
	loss = []
	for epoch in range(FLAGS.max_epoch):
		for batch in _train_manager.iter_batch(shuffle=True):
			step, batch_loss = model.run_step(sess, True, batch)
			loss.append(batch_loss)
			if step % FLAGS.steps_check == 0:
				logger.info("Epoch:{} step:{}/{}, "
							"loss:{:>9.6f}".format(epoch+1, step%steps_per_epoch, \
									steps_per_epoch, np.mean(loss)))

		best, f1 = evaluate(sess, model, dev_manager, id_to_tag, logger)

		# early stopping
		if best:
			early_stop = 0
		else:
			early_stop += 1
			if early_stop > FLAGS.patience: break

		# save model & best f1 score
		if best:
			save_model(sess, model, FLAGS.ckpt_path, logger)
			sess_best = f1
		loss = []
	
	return sess_best

if __name__ == '__main__':
	FLAGS = parse_args()

	# tensorflow config
	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth = True
	sess = tf.Session(config=tf_config)

	# model config
	make_path(FLAGS)
	config = config_model()
	save_config(config, FLAGS.config_file)
	log_path = os.path.join("log", FLAGS.log_file)
	logger = get_logger(log_path)
	print_config(config, logger)

	# create model
	model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)

	dataset = data_loader(FLAGS.DATASET_PATH)
	train_dataset, dev_dataset = dataset[:-3000], dataset[-3000:]

	# create dictionary for word
	_, char_to_id, id_to_char = char_mapping(train_dataset, FLAGS.lower)

	# create a dictionary and a mapping for tags
	_, tag_to_id, id_to_tag = tag_mapping(train_dataset)

	# prepare data, get a collection of list containing index
	train_data = prepare_dataset(train_dataset, char_to_id, tag_to_id, num_shuffle=FLAGS.num_shuffle)
	dev_data = prepare_dataset(dev_dataset, char_to_id, tag_to_id, train=False)

	train_manager = BatchManager(train_data, FLAGS.batch_size, FLAGS.max_char_length)
	dev_manager = BatchManager(dev_data, 100, FLAGS.max_char_length)

	best_f1 = train(sess, train_manager)
	logger.info(f"Best F1-score: {best_f1}")