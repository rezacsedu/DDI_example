import pandas as pd
from sklearn.metrics import classification_report

def get_data(fname = 'DrugBank_Format_CRF++.csv'):
	'''
	transform the csv file provided by Sachin to the right format
	'''
	df = pd.read_csv(fname,header=0)
	lines = []
	for row_num, each in df.iterrows():
		lines.append(each.tolist())
	#may need to split data in the future
	outfile = 'crf_train'
	f = open(outfile,'w')
	for line in lines:
		f.write(str(line[0]) + ' ' + str(line[1]) + ' ' + str(line[2]) + '\n')
	f.close()

def evaluation(fname = 'rs.txt'):
	f = open('rs.txt','rb')
	lines = f.readlines()
	f.close()
	y_true = []
	y_pred = []
	for line in lines:
		if line.strip() :
			cols = line.split('\t')
			if cols[3].strip() != '.':
				y_true.append(cols[2].strip())
				y_pred.append(cols[3].strip())
	report = classification_report(y_true, y_pred)
	f = open('task1_report.txt','w')
	f.write(report)
	f.close()


if __name__ == '__main__':
	evaluation('rs.txt')
