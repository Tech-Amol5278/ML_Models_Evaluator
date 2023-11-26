import matplotlib.pyplot as plt
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import GridSearchCV
import joblib

#### check or create directory ####
cwd = os.getcwd()
reportDir = os.path.join(cwd,'reports')
mlStatsDir = os.path.join(reportDir,'ml_stats')
os.makedirs(mlStatsDir, exist_ok=True)
modelDir=os.makedirs(os.path.join(cwd,'models'), exist_ok=True)

###################### Random Forest Classifier ####################################################
class evaluator():

	def getClassName(self):
		return __class__.__name__

	def __init__(self, x_train, y_train, x_test, y_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		####### super class name ################
		self.cl_name = self.getClassName()

		####### method names ####################
		self.fn_name_rfClassifier = self.randomForestClassifer.__func__.__name__
		self.fn_name_knnClassifier = self.knnClassifier.__func__.__name__
		self.fn_name_check_accuracy = self.getClassifiersAccuracy.__func__.__name__

	###############################################################################################################
	################################# Random Forest Classifier ####################################################
	###############################################################################################################
	def randomForestClassifer(self,iters):
		'''
		:param self:
		:param iters: list of iterators
		:return: Classifier : classifiers name
				estimators : list of estimators on which the model evaluation was performed
				scores : list of Accuracy scores recorded against each estimator
				precisions : list of precisions recorded against each estimator
				recalls : list of recalls recorded against each estimator
				f1_scores : list of F1-scores recorded against each estimator
		'''

		######### fill prerquisites ################
		estimators=iters
		############################################
		scores=[]
		precisions=[]
		recalls=[]
		f1_scores=[]

		for i in estimators:
			model=RandomForestClassifier(n_estimators=i, random_state=42)
			model.fit(self.x_train,self.y_train)
			y_pred=model.predict(self.x_test)
			score=round(accuracy_score(self.y_test,y_pred),2)

			scores.append(score)
			# Calculate precision, recall, and F1-score
			report = classification_report(self.y_test, y_pred, output_dict=True)

			precisions.append(report['weighted avg']['precision'])
			recalls.append(report['weighted avg']['recall'])
			f1_scores.append(report['weighted avg']['f1-score'])

		# # confusion matrix
		# class_labels = model.classes_
		# class_labels = list(class_labels)
		# print(class_labels)
		# swapped_y_mapper = {value: key for key, value in y_mapper.items()}
		# class_labels = [swapped_y_mapper[item] for item in class_labels]
		# print(class_labels)
		# cm = confusion_matrix(self.y_test, y_pred, normalize='true')
		# disp = ConfusionMatrixDisplay(cm,display_labels=class_labels)
		# import matplotlib.pyplot as plt1
		# disp = disp.plot(cmap=plt1.cm.Blues,values_format='g')
		# plt1.tight_layout()
		# plt1.xticks(rotation=45, ha='right')
		# plt1.show()

		return self.fn_name_rfClassifier, estimators, scores, precisions, recalls, f1_scores

	###############################################################################################################
	##################################### KNN Classifier ##########################################################
	###############################################################################################################
	def knnClassifier(self, iters):
		'''
		:param self:
		:param iters: list of iterators
		:return: Classifier : classifiers name
				neighbors : list of neighbors on which the model evaluation was performed
				scores : list of Accuracy scores recorded against each neighbors
				precisions : list of precisions recorded against each neighbors
				recalls : list of recalls recorded against each neighbors
				f1_scores : list of F1-scores recorded against each neighbors
		'''
		######### fill prerquisites ################
		neighbors=iters
		############################################
		scores=[]
		precisions=[]
		recalls=[]
		f1_scores=[]

		for i in neighbors:
			model = KNeighborsClassifier(n_neighbors=i)
			model.fit(self.x_train, self.y_train)
			y_pred = model.predict(self.x_test)
			score=round(accuracy_score(self.y_test,y_pred),2)
			scores.append(score)

				# Calculate precision, recall, and F1-score
			report = classification_report(self.y_test, y_pred, output_dict=True)

			precisions.append(report['weighted avg']['precision'])
			recalls.append(report['weighted avg']['recall'])
			f1_scores.append(report['weighted avg']['f1-score'])

		# # confusion matrix
		# class_labels = model.classes_
		# class_labels = list(class_labels)
		# print(class_labels)
		# swapped_y_mapper = {value: key for key, value in y_mapper.items()}
		# class_labels = [swapped_y_mapper[item] for item in class_labels]
		# print(class_labels)
		# cm = confusion_matrix(self.y_test, y_pred, normalize='true')
		# disp = ConfusionMatrixDisplay(cm,display_labels=class_labels)
		# import matplotlib.pyplot as plt1
		# disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
		# plt.tight_layout()
		# plt.xticks(rotation=45, ha='right')
		# plt.show()

		return self.fn_name_knnClassifier, neighbors, scores, precisions, recalls, f1_scores

	###############################################################################################################
	######################## Check Accuracy #######################################################################
	###############################################################################################################
	def getClassifiersAccuracy(self, _classifier, x, scores, precisions, recalls, f1_scores):
		'''

		:param x: neighbors or estimators
		:param y:
		:param scores:
		:param precisions:
		:param recalls:
		:param f1_scores:
		:return:
		'''

		############## Environment #########################
		# graph : Estimators Vs Accuracy
		ylabel_accuracy = "Accuracy Scores"
		title_file_accuracy = "Estimators Vs Accuracy"
		title_accuracy = f"{_classifier} : {title_file_accuracy}"
		print(title_accuracy)

		# graph : Estimator vs Precision, Recall and F1-scores
		ylabel_PRF1 = "Precision, Recall, F1"
		title_file_PRF1 = "Precision, Recall and F1-score vs. Number of Estimators"
		title_PRF1 = f"{_classifier}: {title_file_PRF1}"
		print(title_PRF1)
		########### model specific enviornment ############
		if _classifier==self.fn_name_rfClassifier:
			# directory
			rf_classifierDir = os.path.join(mlStatsDir,_classifier)
			os.makedirs(rf_classifierDir, exist_ok=True)

			########### graph ###########
			x_label = "Number of Estimators"

			# graph : Estimators Vs Accuracy
			file_accuracy = os.path.join(rf_classifierDir,f"{title_file_accuracy}.png")

			# graph : Estimator vs Precision, Recall and F1-scores
			file_PRF1 = os.path.join(rf_classifierDir,f"{title_file_PRF1}.png")
		elif _classifier==self.fn_name_knnClassifier:
			# directory
			knn_classifierDir = os.path.join(mlStatsDir,_classifier)
			os.makedirs(knn_classifierDir, exist_ok=True)

			########### graph ###########
			x_label = "Number of Neighbors"

			# graph : Estimators Vs Accuracy
			file_accuracy = os.path.join(knn_classifierDir,f"{title_file_accuracy}.png")

			# graph : Estimator vs Precision, Recall and F1-scores
			file_PRF1 = os.path.join(knn_classifierDir,f"{title_file_PRF1}.png")
		else:
			pass
		####################################################
		plt.figure(figsize=(12,5))
		plt.plot(x,scores,color='green', linestyle='solid', marker='o', markerfacecolor='blue', markersize=5)
		plt.xlabel(x_label)
		plt.ylabel(ylabel_accuracy)
		plt.title(title_accuracy)
		plt.xticks(x)
		plt.show()
		try:
			plt.savefig(file_accuracy)
		except:
			print("File not found")



		# Estimator vs Precision, Recall and F1-scores
		# plt.figure(figsize=(10, 6))
		plt.figure(figsize=(12,5))
		plt.plot(x, scores,color='green', label='Accuracy', linestyle='solid', marker='o', markerfacecolor='blue', markersize=5)
		plt.plot(x, precisions, label='Precision', marker='o')
		plt.plot(x, recalls, label='Recall', marker='o')
		plt.plot(x, f1_scores, label='F1-score', marker='o')
		plt.xlabel(x_label)
		plt.ylabel(ylabel_PRF1)
		plt.title(title_PRF1)
		plt.legend()
		plt.grid(True)
		plt.show()
		try:
			plt.savefig(file_PRF1)
		except:
			print("File not found")

		return
