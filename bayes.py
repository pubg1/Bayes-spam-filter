# -*- coding: utf-8 -*-
"""
created on 2016年九月十日
@author : hwd
"""
import jieba
import collections
import os
import re
import pdb
import logging
import redis
import traceback
import yaml
import shutil
from multiprocessing.dummy import Pool

jieba.enable_parallel(4) #结巴分词开启并行分词
logger = logging.getLogger('bayes')
logger.setLevel(logging.INFO)
cmd_handler = logging.StreamHandler()
formatter = logging.Formatter("""%(asctime)s - %(module)s - %(funcName)s -%(lineno)d- %(levelname)s - %(message)s""")
cmd_handler.setFormatter(formatter)
logger.addHandler(cmd_handler)


class Error(Exception):

	def __init__(self, value):
		self.value = value

	def __str__(self):
 		return repr(self.value)


class Bayes(object):
	"""朴素贝叶斯邮件分类"""

	def __init__(self):
		self.spam_dict = collections.defaultdict(dict)
		self.ham_dict = collections.defaultdict(dict)
		self.conf = yaml.load(file("conf/conf.yaml")) 
		self.redis_connection =redis.Redis(host=self.conf["Redis"]["host"],port = self.conf["Redis"]["port"],db=self.conf["Redis"]['db'])
		self.spam_dir = self.conf["TrainingDataPath"]['spam']
		self.ham_dir = self.conf["TrainingDataPath"]['ham']
		if self.redis_connection.exists("last_increased_file_len"):
			self.last_increased_file_len = int(self.redis_connection.get("last_increased_file_len"))
		else:
			self.last_increased_file_len = 0 #上一次训练数据里新增的邮件数量
		self.total_file_len = (sum([len(x) for _,_,x in os.walk(self.spam_dir)])+
		sum([len(x) for _,_,x in os.walk(self.ham_dir)]))
		self.useless_token_list = self.get_useless_tokens()

	def __call__(self):
		self.classify(self.conf["TestDataPath"])

	def get_useless_tokens(self):
		"""加载中文停用词

		Yields:
		      返回一个包含中文停用词的列表

		Raises:
			   IOError:文件不存在
			   GeneratorExit 
		"""
		try:
			useless_token_list = []
			logger.debug("load useless tokens")
			with open(self.conf['UselessTokensPath']) as f :
				for line in f:
					useless_token_list.append(line[:len(line)-1])
					yield useless_token_list
		except:
			logger.error(traceback.print_exc())

	def email_pretreatment(self,file_path):
		"""对邮件进行预处理，过滤邮件中的英文单词和符号，进行分词

		Args:
			file_path:需要进行预处理的邮件存放路径

		Return:
			包含邮件分词后所有词的集合
			set()
		"""
		try:
			with open(file_path) as f:
				token_set=set()
				rule=re.compile('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+（')
				content = f.read()
				content = rule.sub("",content)
				content = content.replace(" ","")
				content = jieba.cut(content)
				token_set = set(token for token in content  if token>=u'\u4e00' and token<=u'\u9fa5' 
					and token not in self.useless_token_list and token.strip()!='' and token!=None)
			return token_set
		except:
			logger.error(traceback.print_exc())
		
	def email_split(self,file_path,save_classify_result=False):
		"""遍历邮件目录，返回邮件分词后的结果

		Args:
		    file_path:需要遍历的目录或者文件
		    save_classify_result:是否需要返回一个保存分类结果的字典

		Yields：
		    如果save_classify_result为true则返回一个字典
		    classify_result = {"file_path":"1.txt"，
		                        "1.txt":{"classify_result:0,
		                                  "token_set":set()}
		                                  }
		    否则返回一个包含token的集合

		Raises:
		  IOError
		  GeneratorExit

		"""
		try:
			if os.path.exists(file_path):
				if os.path.isdir(file_path):
					for f in os.listdir(file_path):
						file_name = os.path.join(file_path,f)
						next_email = self.email_split(file_name,save_classify_result)
						yield next(next_email)
				if os.path.isfile(file_path):
					if save_classify_result:
						classify_result = collections.defaultdict(dict)
						classify_result["file_path"] = file_path
						classify_result[file_path]["classify_result"]=0
						classify_result[file_path]["token_set"] = self.email_pretreatment(file_path)
						yield classify_result
					else:
						yield  self.email_pretreatment(file_path)
			else:
				raise Error("%s 不存在"% file_path)
		except GeneratorExit :
			pass
		except :
			logger.error(traceback.print_exc())

	def add_to_dict(self,token_set,token_dict):
		for token in token_set:
			if token in token_dict:
				token_dict[token] += 1
			else:
				token_dict[token]  = 1

	def train(self):
		"""对算法进行训练，保存训练之后的结果"""
		logger.info(" 开始训练")
		total_ham_file = sum([len(x) for _,_,x in os.walk(self.ham_dir)])
		total_spam_file = sum([len(x) for _,_,x in os.walk(self.spam_dir)])
		ham_tokens = self.email_split(self.ham_dir)
		#计算token在正常邮件里和垃圾邮件里出现的次数
		for ham_token_set in ham_tokens:
			self.add_to_dict(ham_token_set,self.ham_dict)
		spam_tokens = self.email_split(self.spam_dir)
		for spam_token_set in spam_tokens:
			self.add_to_dict(spam_token_set,self.spam_dict)
		token_set = set(self.ham_dict.keys()) | set(self.spam_dict.keys())
		for token in token_set:
			#对于token集合里的每个token，计算邮件包含token时是垃圾邮件的概率
			if token not in self.spam_dict and token in self.ham_dict:
				pw_s = 0.01
				pw_h = float(self.ham_dict[token]) / float(total_ham_file)
				token_possibility = pw_s/(pw_s+pw_h)
				self.redis_connection.hset("possibility",token,token_possibility)
			elif token in self.spam_dict and token not in self.ham_dict:
				pw_h = 0.01
				pw_s = float(self.spam_dict[token]) /float(total_spam_file)
				token_possibility = pw_s/(pw_s+pw_h)
				self.redis_connection.hset("possibility",token,token_possibility)
			else:
				pw_s = float(self.spam_dict[token])/float(total_spam_file)
				pw_h = float(self.ham_dict[token])/float(total_ham_file)
				token_possibility = pw_s/(pw_s+pw_h)
				self.redis_connection.hset("possibility",token,token_possibility)
		self.redis_connection.set("last_increased_file_len",0)
		self.redis_connection.save()
		logger.info("保存数据到redis")
		logger.info("训练结束")

	def cal_bayes(self,token_dict):
		"""计算一封邮件是垃圾邮件的概率"""
		ps_w=1
		ps_n=1
		file_path = token_dict["file_path"]
		for token in token_dict[file_path]['token_set']:
			token = token.encode('utf-8')
			if token in self.possibility:
				prob = float(self.possibility[token])	
			else:
				prob = 0.4 #如果在历史数据中没有出现过该词，则判定概率为0.4
			ps_w*=prob
			ps_n*=(1-prob)
		p = ps_w/(ps_n+ps_w)
		if p>0.9:
			token_dict[file_path]['classify_result'] = 1
		return token_dict

	def classify(self,file_path):
		""" 对邮件进行分类，打印结果，对测试邮件重新学习

		Args:
		    file_path: 需要分类的邮件目录或者文件
		"""
		try :
			pool =Pool(4)
			if self.redis_connection.exists("possibility"):
				logger.info("从redis加载数据")
				self.possibility = self.redis_connection.hgetall("possibility")
			else:
				self.train()
				self.possibility = self.redis_connection.hgetall('possibility')
			if os.path.exists(file_path):
				if os.path.isfile(file_path):
					classify_file_dict  = self.email_split(file_path,save_classify_result=True)
					classify_result = self.cal_bayes(next(tokens_list))	
					classify_results = [classify_result]
					self.print_result(classify_results)
					self.learner()
				else:
					tokens_list = self.email_split(file_path,save_classify_result=True)
					classify_results = pool.map(self.cal_bayes,tokens_list)
					pool.close()
					pool.join()
					self.print_result(classify_results)
					self.learner()
			else:
				raise Error("%s 不存在"%file_path)
		except :
			logger.error(traceback.print_exc())

	def print_result(self,classify_results):
		"""打印分类结果"""
		spam = 0
		for classify_result in classify_results:
			file_path = classify_result["file_path"]
			prob = classify_result[file_path]['classify_result']
			if prob == 1:
				logger.info("%s  是垃圾邮件"%file_path)
				shutil.copyfile(file_path,"".join([self.spam_dir,os.sep,os.path.basename(file_path)]))
				spam += 1
			else:
				logger.info("%s  不是垃圾邮件"%file_path)
				shutil.copyfile(file_path,"".join([self.ham_dir,os.sep,os.path.basename(file_path)]))

	def learner(self):
		"""对已经的分类的测试邮件进行学习"""
		increased_file_len = sum([len(x) for _,_,x in os.walk(self.spam_dir)])+\
		sum([len(x) for _,_,x in os.walk(self.ham_dir)])-self.total_file_len
		total_increased_file_len = self.last_increased_file_len+increased_file_len 
		if total_increased_file_len >100:
			self.train()
		else:
			self.last_increased_file_len = self.last_increased_file_len + increased_file_len
			self.redis_connection.set("last_increased_file_len",self.last_increased_file_len)
			self.redis_connection.save()
				
def main():
	bayes_classfier = Bayes()
	bayes_classfier()

if __name__ == "__main__":
	main()
	
	
	


		




