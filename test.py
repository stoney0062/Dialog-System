# _*_ coding:utf-8 _*_
import sys
import re
reload(sys)
sys.setdefaultencoding('utf8')
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
contents=u'中'


def is_chinese(uchar):
        """判断一个unicode是否是汉字"""
	if uchar >= u'/u4e00' and uchar<=u'/u9fa5':
		return True
	else:
		return False

def ReadData():
	filepath = sys.argv[1]
	file = open(filepath)
	list = ['。','！','?',';']
	#eng=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'] 
	count = 0
	while 1:
		output = ""
		result = []
		line = file.readline()
		if not line:
			break
		if len(line) < 4 :
			continue
		linelist = line.strip().split('。')
		for sentence in linelist:
			#for i in range(len(sentence)-2):
			#	if sentence[i]== " " :
			#		sentence = sentence[0:i]+sentence[i+1:len(sentence)]
			sentence.replace('\s', '')
			sentence.replace('  ', '')
			sentence.replace('   ', '')
			sentence.replace('    ', '')
			sentence.replace('     ', '')
			sentence.replace('      ', '')
			sentence.replace('       ', '')
			sentence.replace('       ', '')
			if len(sentence)<=9:
				print sentence,
			else:
				if count != 0:
					print " "
				print sentence,
		count += 1
					 
if __name__ == '__main__':
	ReadData()
