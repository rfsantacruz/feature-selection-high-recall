package tests;

import static org.junit.Assert.fail;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import junit.framework.Assert;

import org.junit.Test;

import problems.ClassificationProblem;
import weka.classifiers.AbstractClassifier;
import classifiers.LogisticRegressionClassifier;
import classifiers.NaiveBayesClassifier;
import classifiers.SVMLinearClassifier;

import com.google.common.collect.Sets;

import evaluation.CrossValidationOutput;
import evaluation.WekaEvaluationWrapper;
import experiment.AbstractExperimentReport;
import experiment.ClassificationExperimentReport;
import experiment.ExperimentExecutor;
import experiment.IExperimentCommand;

public class BackGroundTest {


	//test read a arff file
	@Test
	public void testReadProblems() {

		String filePath = "./TestDataSets/lsdata1.arff";
		ClassificationProblem cp;
		try {
			cp = new ClassificationProblem(filePath);

			Assert.assertNotNull("can not read the data",cp.getData());
			Assert.assertTrue("arff relation name wrong",cp.getName().equals("LinearSeparableYequalX"));
			Assert.assertTrue("path wrong",cp.getFilePath().equals(".\\TestDataSets\\lsdata1.arff"));
			Assert.assertTrue("don't read all examples",cp.getNumExamples() == 300);
			Assert.assertTrue("didn't read all atributes",cp.getNumAttributes() == 3);
			Assert.assertTrue("class index wrong",cp.getData().classIndex() == 2);

		} catch (IOException e) {
			fail("Problem to read the arrff file: " + e.getMessage());
		}
	}

	//test classifier in artificial linear sparable data
	@Test
	public void testClassifiersInArtificialData() {

		try{

			ClassificationProblem cp = new ClassificationProblem("./TestDataSets/lsdata1.arff");
			WekaEvaluationWrapper ev = new WekaEvaluationWrapper(cp);

			AbstractClassifier nb = new NaiveBayesClassifier();
			AbstractClassifier lr = new LogisticRegressionClassifier();
			AbstractClassifier svm = new SVMLinearClassifier();


			nb.buildClassifier(cp.getData());
			ev.evaluateModel(nb,cp.getData());
			double error_rate_nb = ev.errorRate();


			lr.buildClassifier(cp.getData());
			ev.evaluateModel(lr,cp.getData());
			double error_rate_lr = ev.errorRate();


			svm.buildClassifier(cp.getData());
			ev.evaluateModel(svm,cp.getData());
			double error_rate_svm = ev.errorRate();


			Assert.assertTrue("Naive byes' erro rate for linear separable data have less than 1%", error_rate_nb < 0.01 );
			Assert.assertTrue("Logistic Regression's erro rate for linear separable data have to be zer0",error_rate_lr == 0 );
			Assert.assertTrue("Linear SVM's erro rate for linear separable data have to be zero",error_rate_svm == 0 );


		}catch(IOException e){
			fail("Can't read the arff file" + e.getMessage());
		} catch (Exception e) {
			fail("Erro in build the classifier" + e.getMessage());
		}
	}

	//test classifier and cross validation in a data base with know performance 
	//TODO: No refrence to NB and LR. There is reference just to Linear SVM (near the reference 0.04 of distance) 
	@Test
	public void testClassifierInUCIData() {
		try{
			ClassificationProblem cp = new ClassificationProblem("./TestDataSets/heart-statlog.arff");
	
			AbstractClassifier nb = new NaiveBayesClassifier();
			AbstractClassifier lr = new LogisticRegressionClassifier();
			AbstractClassifier svm = new SVMLinearClassifier();

			WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);

			eval.crossValidateModel(nb, cp, 10, 10, null);
			//Assert.assertTrue("Naive Bayes performance not expected", Math.abs(eval.accuracy()-0.75) < 0.05);

			HashMap<String,Set<String>> paramLR = new HashMap<String,Set<String>>();
			paramLR.put("-C",utils.Util.generateModelsStringSettings("-C", 0.01, 0.03, 0.1, 0.3, 1, 1.3));
			paramLR.put("-B",utils.Util.generateModelsStringSettings("-B", 0.01, 0.03, 0.1, 0.3, 1, 1.3));
			eval.crossValidateModel(lr, cp, 10, 10, paramLR);
			Assert.assertTrue("Logistic Regression performance not expected", Math.abs(eval.accuracy()-0.85) < 0.05);


			HashMap<String,Set<String>> paramSVM = new HashMap<String,Set<String>>();
			paramSVM.put("-C",utils.Util.generateModelsStringSettings("-C", 0.01, 0.03, 0.1, 0.3, 1, 1.3));
			paramSVM.put("-B",utils.Util.generateModelsStringSettings("-B", 0.01, 0.03, 0.1, 0.3, 1, 1.3));
			eval.crossValidateModel(svm, cp, 10, 10, paramSVM);
			Assert.assertTrue("SVM performance not expected", Math.abs(eval.accuracy() - 0.86) < 0.05);


		}catch(IOException e){
			fail("Can't read the arff file" + e.getMessage());
		} catch (Exception e) {
			fail("Erro in build the classifier" + e.getMessage());
		}
	}

	//test run in jarfiles and files
	@Test
	public void testExperimentExecutor() {
		IExperimentCommand cmd = new IExperimentCommand() {

			@Override
			public List<AbstractExperimentReport> execute(ClassificationProblem cp) {
				ArrayList<AbstractExperimentReport> exp = new ArrayList<AbstractExperimentReport>();
				exp.add(new ClassificationExperimentReport("test", "test"));
				return exp;
			}
		};
		String dataPath = "./data/iris.data.arff";
		ClassificationExperimentReport exprep = (ClassificationExperimentReport)ExperimentExecutor.getInstance().executeCommandInFile(cmd, dataPath);
		Assert.assertTrue("", exprep.getProblemName().equals("test"));

		IExperimentCommand cmd2 = new IExperimentCommand() {

			@Override
			public List<AbstractExperimentReport> execute(ClassificationProblem cp) {
				ArrayList<AbstractExperimentReport> exp = new ArrayList<AbstractExperimentReport>();
				exp.add(new ClassificationExperimentReport(cp.getName(), cp.getName()));
				return exp;
			}
		};
		String dataPath2 = "./data";
		List<AbstractExperimentReport> exprep2 =  ExperimentExecutor.getInstance().executeCommandInFiles(cmd2, dataPath2);
		Assert.assertNotNull("Returning null in execute experiemnts in a diretory", exprep2);
		Assert.assertTrue("Retrunning zero items in execute experiemnts in a diretory", exprep2.size() > 0);

		IExperimentCommand cmd3 = new IExperimentCommand() {

			@Override
			public List<AbstractExperimentReport> execute(ClassificationProblem cp) {
				ArrayList<AbstractExperimentReport> exp = new ArrayList<AbstractExperimentReport>();
				exp.add(new ClassificationExperimentReport(cp.getName(), cp.getName()));
				return exp;
			}
		};
		String dataPath3 = "./data/datasets-UCI.jar";
		List<AbstractExperimentReport> exprep3 = ExperimentExecutor.getInstance().executeCommandInJAR(cmd3, dataPath3);
		Assert.assertNotNull("Execute Experiment in jar data set: return null", exprep3);
		Assert.assertTrue("Execute Experiment in jar data set have to get 36 reports(==36 problems)" + exprep3.size(), exprep3.size() == 36);

	}

	//test cross validation
	//TODO: check if the crossvalidation have to give the best results
	@Test
	public void testCrossValidation() {
		try{
			ClassificationProblem p = new ClassificationProblem("./data/iris.data.arff");
			AbstractClassifier lr = new LogisticRegressionClassifier();
			WekaEvaluationWrapper ev = new WekaEvaluationWrapper(p);

			//cross validation with default parameters
			CrossValidationOutput cvo = ev.crossValidateModel(lr, p, 10, 10, null);
			double accuracy = cvo.getAccuracy();
			double precision = cvo.getPrecision();
			double recall = cvo.getRecall();
			double fmeasure = cvo.getF_measure();

			HashMap<String,Set<String>> paramLR = new HashMap<String,Set<String>>();
			paramLR.put("-C",Sets.newHashSet("-C 0.1", "-C 0.3", "-C 1.0"));
			paramLR.put("-B",Sets.newHashSet("-B 0.1", "-B 0.3", "-B 1.0"));

			//return the classifier already tuned with cross validation with parameters
			CrossValidationOutput cvo2 = ev.crossValidateModel(lr, p, 10, 10, paramLR);
			double accuracy_aftercv = cvo2.getAccuracy();
			double precision_aftercv = cvo2.getPrecision();
			double recall_aftercv = cvo2.getRecall();
			double fmeasure_aftercv = cvo2.getF_measure();


			Assert.assertTrue("The cross validated classifier have to be better in accuracy", accuracy_aftercv >=  accuracy);
			Assert.assertTrue("The cross validated classifier have to be better in precision", precision_aftercv >=  precision);
			Assert.assertTrue("The cross validator classifier have to be better in recall", recall_aftercv >=  recall);
			Assert.assertTrue("The cross validator classifier have to be better in fmeasure", fmeasure_aftercv >=  fmeasure);


		}catch(IOException e){
			fail("Can't read the arff file" + e.getMessage());
		} catch (Exception e) {
			fail("Erro in build the classifier" + e.getMessage());
		}
	}

	//naive test of metrics with the artificial data where the classifier can predict right every trainig example
	// accuraccy = precision = recall = fmeasure = 1  and errorrate = a
	public void testMetrics(){
		String filePath = "./TestDataSets/lsdata1.arff";
		ClassificationProblem cp;
		try {
			
			cp = new ClassificationProblem(filePath);

			AbstractClassifier svm = new SVMLinearClassifier();
			WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);

			svm.buildClassifier(cp.getData());
			eval.evaluateModel(svm,cp.getData());
			double error_rate_nb = eval.errorRate();
			double accuracy_nb = eval.accuracy();
			double precision_nb = eval.precision();
			double recall_nb = eval.recall();
			double fmeasure_nb = eval.fMeasure();
			
			Assert.assertEquals(error_rate_nb, 0);
			Assert.assertEquals(accuracy_nb, 1);
			Assert.assertEquals(precision_nb, 1);
			Assert.assertEquals(recall_nb, 1);
			Assert.assertEquals(fmeasure_nb, 1);
			

		} catch (Exception e) {
			fail("Problem to read the arrff file: " + e.getMessage());
		}
	}

}
