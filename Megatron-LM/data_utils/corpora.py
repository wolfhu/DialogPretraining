# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""several datasets with preset arguments"""
from .datasets import json_dataset, csv_dataset, other_dataset, finetune_dataset
import os

# base_path = r"\\gcrnfsw2-xiaoic\xiaoicechatexp\yuniu\dataset"
# base_path = r'data/'
base_path = r'/blob/data/processed'

class wikipedia(json_dataset):
	"""
	dataset for wikipedia with arguments configured for convenience

	command line usage: `--train-data wikipedia`
	"""
	# PATH = 'data/wikipedia/wikidump_lines.json'
	PATH = 'data/test.json'
	assert_str = "make sure to set PATH for wikipedia data_utils/corpora.py"
	def __init__(self, **kwargs):
		assert os.path.exists(wikipedia.PATH), \
                        wikipedia.assert_str
		if not kwargs:
			kwargs = {}
		kwargs['text_key'] = 'text'
		kwargs['loose_json'] = True
		super(wikipedia, self).__init__(wikipedia.PATH, **kwargs)


class webtext(json_dataset):
	"""
	dataset for webtext with arguments configured for convenience

	command line usage: `--train-data webtext`
	"""
	PATH = 'data/webtext/data.json'
	assert_str = "make sure to set PATH for webtext data_utils/corpora.py"
	def __init__(self, **kwargs):
		assert os.path.exists(webtext.PATH), \
                        webtext.assert_str
		if not kwargs:
			kwargs = {}
		kwargs['text_key'] = 'text'
		kwargs['loose_json'] = True
		super(webtext, self).__init__(webtext.PATH, **kwargs)

class chinese_dataset(other_dataset):
	"""
	dataset for chinese with arguments configured for convenience

	command line usage: `--train-data webtext`
	"""

	def __init__(self, _path = '', **kwargs):
		self._PATH = _path if _path else ''  # 'data/webtext/data.json'
		self.PATH = os.path.join(base_path, self._PATH)
		assert_str = "make sure to set PATH for chinese data_utils/corpora.py" + self.PATH

		assert os.path.exists(self.PATH), \
			assert_str
		if not kwargs:
			kwargs = {}
		super(chinese_dataset, self).__init__(self.PATH, **kwargs)

class wiki(other_dataset):
	"""
	dataset for webtext with arguments configured for convenience

	command line usage: `--train-data webtext`
	"""
	PATH = os.path.join(base_path, 'wiki/wiki.txt' )
	# PATH = os.path.join(base_path, 'wiki/wiki.clean.merge_item.clean.txt')
	assert_str = "make sure to set PATH for wiki"
	def __init__(self, **kwargs):
		# self._PATH = 'wiki/wiki.clean.merge_item.clean.txt'  # 'data/webtext/data.json'
		# self.PATH = os.path.join(base_path, self._PATH)
		assert os.path.exists(wiki.PATH), \
			wiki.assert_str
		if not kwargs:
			kwargs = {}
		super(wiki, self).__init__( wiki.PATH, **kwargs)
		print("wiki path ", self.PATH)

class baidu_baike(other_dataset):
	"""
	dataset for webtext with arguments configured for convenience

	command line usage: `--train-data webtext`
	"""
	PATH = os.path.join(base_path, 'baidu_baike/baidubaike.txt' )
	# PATH = os.path.join(base_path, 'wiki/wiki.clean.merge_item.clean.txt')
	assert_str = "make sure to set PATH for baidubaike"
	def __init__(self, **kwargs):
		# self._PATH = 'wiki/wiki.clean.merge_item.clean.txt'  # 'data/webtext/data.json'
		# self.PATH = os.path.join(base_path, self._PATH)
		assert os.path.exists(baidu_baike.PATH), \
			baidu_baike.assert_str
		if not kwargs:
			kwargs = {}
		super(baidu_baike, self).__init__( baidu_baike.PATH, **kwargs)
		print("baidu_baike path ", self.PATH)


class lyric(other_dataset):
	"""
	dataset for webtext with arguments configured for convenience

	command line usage: `--train-data webtext`
	"""
	PATH = os.path.join(base_path, 'lyric/all_music.txt' )
	assert_str = "make sure to set PATH for lyric"
	def __init__(self, **kwargs):
		# self._PATH = 'wiki/wiki.clean.merge_item.clean.txt'  # 'data/webtext/data.json'
		# self.PATH = os.path.join(base_path, self._PATH)
		assert os.path.exists(lyric.PATH), \
			lyric.assert_str
		if not kwargs:
			kwargs = {}
		super(lyric, self).__init__( lyric.PATH, **kwargs)
		print("lyric path ", self.PATH)

class news(other_dataset):
	"""
	dataset for webtext with arguments configured for convenience

	command line usage: `--train-data webtext`
	"""
	PATH = os.path.join(base_path, 'news/news.txt' )
	# PATH = os.path.join(base_path, 'wiki/wiki.clean.merge_item.clean.txt')
	assert_str = "make sure to set PATH for news"
	def __init__(self, **kwargs):
		# self._PATH = 'wiki/wiki.clean.merge_item.clean.txt'  # 'data/webtext/data.json'
		# self.PATH = os.path.join(base_path, self._PATH)
		assert os.path.exists(news.PATH), \
			news.assert_str
		if not kwargs:
			kwargs = {}
		super(news, self).__init__( news.PATH, **kwargs)
		print("news path ", self.PATH)

class zhaichaowang(other_dataset):
	"""
	dataset for webtext with arguments configured for convenience

	command line usage: `--train-data webtext`
	"""
	PATH = os.path.join(base_path, 'zhaichaowang/all.v1.txt' )
	# PATH = os.path.join(base_path, 'wiki/wiki.clean.merge_item.clean.txt')
	assert_str = "make sure to set PATH for zhaichaowang"
	def __init__(self, **kwargs):
		# self._PATH = 'wiki/wiki.clean.merge_item.clean.txt'  # 'data/webtext/data.json'
		# self.PATH = os.path.join(base_path, self._PATH)
		assert os.path.exists(zhaichaowang.PATH), \
			zhaichaowang.assert_str
		if not kwargs:
			kwargs = {}
		super(zhaichaowang, self).__init__( zhaichaowang.PATH, **kwargs)
		print("zhaichaowang path ", self.PATH)

class finetune(finetune_dataset):
	"""
	dataset for webtext with arguments configured for convenience

	command line usage: `--train-data webtext`
	"""
	PATH = os.path.join(base_path, 'finetune/douban.train.class.txt' )
	assert_str = "make sure to set PATH for finetune " + PATH
	def __init__(self, **kwargs):
		# self._PATH = 'wiki/wiki.clean.merge_item.clean.txt'  # 'data/webtext/data.json'
		# self.PATH = os.path.join(base_path, self._PATH)
		assert os.path.exists(finetune.PATH), \
			finetune.assert_str
		if not kwargs:
			kwargs = {}
		super(finetune, self).__init__(finetune.PATH, finetune = True, **kwargs)
		print("finetune path ", self.PATH)

class finetune_test(finetune_dataset):
	"""
	dataset for webtext with arguments configured for convenience

	command line usage: `--train-data webtext`
	"""
	# PATH = os.path.join(base_path, 'finetune/test.data.txt' )
	PATH = 'data/finetune/test.data.txt'
	assert_str = "make sure to set PATH for finetune_test " + PATH
	def __init__(self, **kwargs):
		# self._PATH = 'wiki/wiki.clean.merge_item.clean.txt'  # 'data/webtext/data.json'
		# self.PATH = os.path.join(base_path, self._PATH)
		assert os.path.exists(finetune.PATH), \
			finetune.assert_str
		if not kwargs:
			kwargs = {}
		super(finetune_test, self).__init__(finetune_test.PATH, finetune = True, **kwargs)
		print("finetune_test path ", self.PATH)

NAMED_CORPORA = {
	'wikipedia': wikipedia,
	'chinese': chinese_dataset,
     'webtext': webtext,
	'wiki': wiki,
	'baidu_baike':baidu_baike,
	'lyric':lyric,
	'news':news,
	'zhaichaowang':zhaichaowang,
	'finetune':finetune,
	'finetune_test':finetune_test,
}
