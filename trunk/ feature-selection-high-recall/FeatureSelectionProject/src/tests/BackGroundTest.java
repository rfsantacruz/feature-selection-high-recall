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
import classifiers.AbstractLinearClassifier;
import classifiers.LogisticRegressionClassifier;
import classifiers.NaiveBayesClassifier;
import classifiers.SVMLinearClassifier;

import com.google.common.collect.Sets;

import evaluation.WekaEvaluationWrapper;
import experiment.ExperimentExecutor;
import experiment.ExperimentReport;
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
	//TODO:Obs naive bayes is missing in one example
	@Test
	public void testClassifiersInArtificialData() {

		try{

			ClassificationProblem cp = new ClassificationProblem("./TestDataSets/lsdata1.arff");
			WekaEvaluationWrapper ev = new WekaEvaluationWrapper(cp);

			AbstractLinearClassifier nb = new NaiveBayesClassifier();
			AbstractLinearClassifier lr = new LogisticRegressionClassifier();
			AbstractLinearClassifier svm = new SVMLinearClassifier();


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

			NaiveBayesClassifier nb = new NaiveBayesClassifier();
			LogisticRegressionClassifier lr = new LogisticRegressionClassifier();
			SVMLinearClassifier svm = new SVMLinearClassifier();

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
			public List<ExperimentReport> execute(ClassificationProblem cp) {
				ArrayList<ExperimentReport> exp = new ArrayList<ExperimentReport>();
				exp.add(new ExperimentReport("test", "test"));
				return exp;
			}
		};
		String dataPath = "./data/iris.data.arff";
		ExperimentReport exprep = ExperimentExecutor.getInstance().executeCommandInFile(cmd, dataPath);
		Assert.assertTrue("", exprep.getProblemName().equals("test"));

		IExperimentCommand cmd2 = new IExperimentCommand() {

			@Override
			public List<ExperimentReport> execute(ClassificationProblem cp) {
				ArrayList<ExperimentReport> exp = new ArrayList<ExperimentReport>();
				exp.add(new ExperimentReport(cp.getName(), cp.getName()));
				return exp;
			}
		};
		String dataPath2 = "./data";
		List<ExperimentReport> exprep2 = ExperimentExecutor.getInstance().executeCommandInFiles(cmd2, dataPath2);
		Assert.assertNotNull("Returning null in execute experiemnts in a diretory", exprep2);
		Assert.assertTrue("Retrunning zero items in execute experiemnts in a diretory", exprep2.size() > 0);

		IExperimentCommand cmd3 = new IExperimentCommand() {

			@Override
			public List<ExperimentReport> execute(ClassificationProblem cp) {
				ArrayList<ExperimentReport> exp = new ArrayList<ExperimentReport>();
				exp.add(new ExperimentReport(cp.getName(), cp.getName()));
				return exp;
			}
		};
		String dataPath3 = "./data/datasets-UCI.jar";
		List<ExperimentReport> exprep3 = ExperimentExecutor.getInstance().executeCommandInJAR(cmd3, dataPath3);
		Assert.assertNotNull("Execute Experiment in jar data set: return null", exprep3);
		Assert.assertTrue("Execute Experiment in jar data set have to get 36 reports(==36 problems)" + exprep3.size(), exprep3.size() == 36);

	}

	//test cross validation
	@Test
	public void testCrossValidation() {
		try{
			ClassificationProblem cp = new ClassificationProblem("./data/iris.data.arff");
			AbstractLinearClassifier lr = new LogisticRegressionClassifier();
			WekaEvaluationWrapper ev = new WekaEvaluationWrapper(cp);

			ev.crossValidateModel(lr, cp, 10, 10, null);
			double accuracy = 1 - ev.errorRate();
			double precision = ev.precision();
			double recall = ev.recall();
			double fmeasure = ev.fMeasure();

			HashMap<String,Set<String>> paramLR = new HashMap<String,Set<String>>();
			paramLR.put("-C",Sets.newHashSet("-C 0.1", "-C 0.3", "-C 1.0", "-C 1.3"));
			paramLR.put("-B",Sets.newHashSet("-B 0.1", "-B 0.3", "-B 1.0", "-B 1.3"));

			//return the classifier already tuned
			ev.crossValidateModel(lr, cp, 10, 10, paramLR);
			double accuracy_aftercv = 1 - ev.errorRate();
			double precision_aftercv = ev.precision();
			double recall_aftercv = ev.recall();
			double fmeasure_aftercv = ev.fMeasure();


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
			AbstractLinearClassifier svm = new SVMLinearClassifier();
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
